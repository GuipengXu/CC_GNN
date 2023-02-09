import numpy as np
import os
import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_normal_initialization
import torch.nn.functional as F


class VBPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """
    def __init__(self, config, dataloader):
        super(VBPR, self).__init__(config, dataloader)

        # load parameters info
        self.u_embedding_size = self.i_embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton
        self.cl_weight = config['cl_weight']

        # define layers and loss
        self.u_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.u_embedding_size * 2)))
        self.i_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.i_embedding_size)))
        if self.v_feat is not None and self.t_feat is not None:
            self.item_raw_features = torch.cat((self.t_feat, self.v_feat), -1)
        elif self.v_feat is not None:
            self.item_raw_features = self.v_feat
        else:
            self.item_raw_features = self.t_feat

        self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.i_embedding_size)
        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):
        r""" Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.u_embedding[user, :]

    def get_item_embedding(self, item):
        r""" Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding[item, :]

    def forward(self, dropout=0.0):
        item_embeddings = self.item_linear(self.item_raw_features)
        item_embeddings = torch.cat((self.i_embedding, item_embeddings), -1)

        user_e = F.dropout(self.u_embedding, dropout)
        item_e = F.dropout(item_embeddings, dropout)
        return user_e, item_e

    def calculate_loss(self, interaction):
        """
        loss on one batch
        :param interaction:
            batch data format: tensor(3, batch_size)
            [0]: user list; [1]: positive items; [2]: user pop; [3] item pop; [4] conf ; [5] neg items
        :return:
        """
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[7]
        conf = interaction[4]
        user_pop = interaction[2]
        item_pop = interaction[3]
        cold = interaction[5]
        warm = interaction[6]


        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        pos_e = item_embeddings[pos_item, :]
        #neg_e = self.get_item_embedding(neg_item)
        neg_e = item_embeddings[neg_item, :]
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)

        mf_loss = - torch.mean(torch.log(1e-10 + torch.sigmoid(pos_item_score - neg_item_score)) * conf)  # self.loss(pos_item_score, neg_item_score)
        reg_loss = self.reg_loss(user_e, pos_e, neg_e)
        
        u_idx = torch.unique(user, return_inverse=True, sorted=False)
        i_idx = torch.unique(pos_item, return_inverse=True, sorted=False)

        u_id = u_idx[0]
        i_id = i_idx[0]

        u_index = torch.unique(u_idx[1], sorted=False)
        i_index = torch.unique(i_idx[1], sorted=False)

        u_pop = torch.index_select(user_pop, 0, u_index).cuda()
        i_pop = torch.index_select(item_pop, 0, i_index).cuda()
        user_u_e = user_embeddings[u_id, :]
        item_u_e = item_embeddings[i_id, :]

        n_user_u_e = F.normalize(user_u_e, dim=1)
        n_item_u_e = F.normalize(item_u_e, dim=1)

        random_noise_u = torch.rand_like(user_u_e).cuda()
        n_user_u_e_1 = n_user_u_e + torch.sign(n_user_u_e) * F.normalize(random_noise_u, dim=-1) * 0.1 * (u_pop/10.0).reshape((-1,1))

        random_noise_i = torch.rand_like(item_u_e).cuda()
        n_item_u_e_1 = n_item_u_e + torch.sign(n_item_u_e) * F.normalize(random_noise_i, dim=-1) * 0.1 * (i_pop/10.0).reshape((-1,1))
        
        user_cl_loss = InfoNCE(n_user_u_e, n_user_u_e_1, 0.2)
        item_cl_loss = InfoNCE(n_item_u_e, n_item_u_e_1, 0.2)
        cl_loss = (user_cl_loss + item_cl_loss)

        cold_e = user_embeddings[cold, :]
        warm_e = item_embeddings[warm, :]

        temperature = 0.2
        n_pos_e, n_cold_e, n_warm_e = F.normalize(pos_e, dim=1), F.normalize(cold_e, dim=1), F.normalize(warm_e, dim=1)
        pos_score = (n_pos_e * n_cold_e).sum(dim=-1)

        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(n_pos_e, n_warm_e.transpose(0, 1))

        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)

        conter_loss1 = -torch.log(pos_score / (ttl_score+pos_score) + 10e-6)

        conter_loss = torch.mean(conter_loss1)

        loss = mf_loss + self.reg_weight * reg_loss + conter_loss * 0.05+  cl_loss * self.cl_weight

        return loss, mf_loss + self.reg_weight * reg_loss, cl_loss, conter_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        all_item_e = item_embeddings
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score

def InfoNCE(view1, view2, temperature=0.2, b_cos = True):
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)

    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))

    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score+10e-6)
    
    return torch.mean(cl_loss)