import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from models.BaseModel import BaseModel
from models.general.LightGCN import LightGCNBase
import random
import time

class AHNSGeneralModel(BaseModel):
    reader, runner = 'BaseReader', 'BaseRunner'

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--num_neg', type=int, default=1,
                            help='The number of negative items during training.')
        parser.add_argument('--dropout', type=float, default=0,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--test_all', type=int, default=0,
                            help='Whether testing on all the items.')
        return BaseModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.user_num = corpus.n_users
        self.item_num = corpus.n_items
        self.num_neg = args.num_neg
        self.dropout = args.dropout
        self.test_all = args.test_all

    def loss(self, out_dict: dict) -> torch.Tensor:
        """
        BPR ranking loss with optimization on multiple negative samples (a little different now to follow the paper ↓)
        "Recurrent neural networks with top-k gains for session-based recommendations"
        :param out_dict: contain prediction with [batch_size, -1], the first column for positive, the rest for negative
        :return:
        """
        predictions = out_dict['prediction']
        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
        neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
        loss = -(((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1)).clamp(min=1e-8,max=1-1e-8).log().mean()
        # neg_pred = (neg_pred * neg_softmax).sum(dim=1)
        # loss = F.softplus(-(pos_pred - neg_pred)).mean()
        # ↑ For numerical stability, use 'softplus(-x)' instead of '-log_sigmoid(x)'
        return loss

    class Dataset(BaseModel.Dataset):
        def __init__(self, model, corpus, phase):
            super().__init__(model, corpus, phase)

            self.user_neg_items = {} # {user_id:set(neg)}
            if phase == 'train':
                for i, u in enumerate(self.data['user_id']):
                    pos_index = self.corpus.train_clicked_set[u]
                    neg_index = set(range(1, self.corpus.n_items)) - pos_index
                    self.user_neg_items[u] = neg_index
        
        def _get_feed_dict(self, index):
            user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
            if self.phase != 'train' and self.model.test_all:
                neg_items = np.arange(1, self.corpus.n_items)
            else:
                neg_items = self.data['neg_items'][index]
            item_ids = np.concatenate([[target_item], neg_items]).astype(int)
            feed_dict = {
                'user_id': user_id,
                'item_id': item_ids
            }
            return feed_dict

        # Sample negative items for all the instances
        def actions_before_epoch(self):
            neg_items = np.random.randint(1, self.corpus.n_items, size=(len(self), self.model.num_neg))
            ngs = []
            # get the candidate negtive items
            for i, u in enumerate(self.data['user_id']):
                candidate_neg = self.user_neg_items[u]
                ng = random.choices(list(candidate_neg), k=self.model.K*self.model.num_neg)
                # sample_size = self.model.K * self.model.num_neg
                # ng = np.array([np.random.choice(list(candidate_neg)) for _ in range(sample_size)])
                ngs.append(ng)
            
            user_gcn_emb, item_gcn_emb = self.model.encoder.get_hop_embed()
            ngs_candidate = torch.LongTensor(ngs).to(self.model.device)
            for k in range(self.model.num_neg):
                ng_index = self.adaptive_negative_sampling(user_gcn_emb, item_gcn_emb, self.data['user_id'],
                                                           ngs_candidate[:, k * self.model.K: (k + 1) * self.model.K],
                                                           self.data['item_id'])
                ng_index = ng_index.cpu().numpy()
                ng_index = ng_index.squeeze()
                neg_items[:, k] = ng_index
            self.data['neg_items'] = neg_items
        
        def similarity(self, user_embeddings, item_embeddings):
            # [-1, n_hops, channel]
            if self.model.simi == 'ip':
                return (user_embeddings * item_embeddings).sum(dim=-1)
            elif self.model.simi == 'cos':
                return F.cosine_similarity(user_embeddings, item_embeddings, dim=-1)
            elif self.model.simi == 'ed':
                return ((user_embeddings - item_embeddings) ** 2).sum(dim=-1)
            else:  # ip
                return (user_embeddings * item_embeddings).sum(dim=-1)
        
        
        def adaptive_negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
            s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
            n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops+1, channel]
            
            s_e = s_e.mean(dim=1)  # [batch_size, channel]
            p_e = p_e.mean(dim=1)  # [batch_size, channel]
            n_e = n_e.mean(dim=2)  # [batch_size, n_negs, channel]
            
            p_scores = self.similarity(s_e, p_e).unsqueeze(dim=1) # [batch_size, 1]
            n_scores = self.similarity(s_e.unsqueeze(dim=1), n_e) # [batch_size, n_negs]
            
            scores = torch.abs(n_scores - self.model.beta * (p_scores + self.model.alpha).pow(self.model.p + 1))
            
            """adaptive negative sampling"""
            indices = torch.min(scores, dim=1)[1].detach()  # [batch_size]
            
            neg_item = torch.gather(neg_candidates, dim=1, index=indices.unsqueeze(-1)).squeeze()
            return neg_item.unsqueeze(-1)


class AHNSLightGCN(AHNSGeneralModel, LightGCNBase):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'batch_size']

    @staticmethod
    def parse_model_args(parser):
        parser = LightGCNBase.parse_model_args(parser)
        parser.add_argument('--alpha', type=float, default=0.1,
                            help='The alpha value in AHNS')
        parser.add_argument('--beta', type=float, default=1.0,
                            help='The beta value in AHNS')
        parser.add_argument('--p', type=float, default=-2,
                            help='The p value in AHNS')
        parser.add_argument('--simi', type=str, default='cos',
                            help='the way to get similarity in AHNS')
        parser.add_argument('--K', type=int, default=32,
                            help='number of candidate negative')
        return AHNSGeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        AHNSGeneralModel.__init__(self, args, corpus)
        self.alpha = args.alpha
        self.beta = args.beta
        self.p = args.p
        self.simi = args.simi
        self.K = args.K
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        out_dict = LightGCNBase.forward(self, feed_dict)
        return {'prediction': out_dict['prediction']}