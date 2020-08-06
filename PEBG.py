import torch.nn as nn
import torch.nn.functional as F

from args import *


class PEBG(nn.Module):
    def __init__(self, questions, tags, question_attr_dic, question_to_emb_idx, tag_to_emb_idx):
        super().__init__()
        self.questions = questions
        self.tags = tags
        self.question_attr_dic = question_attr_dic
        self.question_to_emb_idx = question_to_emb_idx
        self.tag_to_emb_idx = tag_to_emb_idx

        self.question_embeddings = nn.Embedding(num_embeddings=len(questions) + 1, embedding_dim=ARGS.emb_dim, padding_idx=Constants.PAD_IDX)
        self.tag_embeddings = nn.Embedding(num_embeddings=len(tags) + 1, embedding_dim=ARGS.emb_dim, padding_idx=Constants.PAD_IDX)
        self.attr_linear = nn.Linear(1, ARGS.emb_dim)
        self.W_z = nn.Linear(ARGS.emb_dim * 3, ARGS.product_layer_dim, bias=False)
        self.W_p_params = nn.Parameter(torch.zeros(ARGS.product_layer_dim, 3))
        self.bias = nn.Parameter(torch.zeros(ARGS.product_layer_dim))
        self.dropout = nn.Dropout(p=ARGS.dropout_rate)
        self.last_linear = nn.Linear(ARGS.product_layer_dim, 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(tensor=p, mean=0, std=0.1)

    def forward(self, QT_q, QT_t, QQ_q1, QQ_q2, TT_t1, TT_t2, diff_q, diff_t, diff_a):
        QT_q_emb = self.question_embeddings(QT_q)
        QT_t_emb = self.tag_embeddings(QT_t)
        QQ_q1_emb = self.question_embeddings(QQ_q1)
        QQ_q2_emb = self.question_embeddings(QQ_q2)
        TT_t1_emb = self.tag_embeddings(TT_t1)
        TT_t2_emb = self.tag_embeddings(TT_t2)
        diff_q_emb = self.question_embeddings(diff_q)
        diff_t_emb = self.question_embeddings(diff_t)
        diff_a_emb = self.attr_linear(diff_a.unsqueeze(-1))

        # question-tag relations
        QT_logit = (QT_q_emb * QT_t_emb).sum(-1)

        # question-question relations
        QQ_logit = (QQ_q1_emb * QQ_q2_emb).sum(-1)

        # tag-tag relations
        TT_logit = (TT_t1_emb * TT_t2_emb).sum(-1)

        # difficulty constraint
        # tag avg embedding
        nonzero_cnt = torch.sum(diff_t != Constants.PAD_IDX, dim=-1, keepdim=True)
        diff_t_avg_emb = torch.sum(diff_t_emb, dim=-2) / nonzero_cnt.float()
        diff_t_avg_emb = diff_t_avg_emb.masked_fill(nonzero_cnt == 0, 0)

        # linear information Z
        Z = torch.cat((diff_q_emb, diff_t_avg_emb, diff_a_emb), dim=-1)
        l_z = self.W_z(Z)

        # quadratic information P
        P11 = (diff_q_emb * diff_q_emb).sum(-1, keepdim=True)
        P12 = (diff_q_emb * diff_t_avg_emb).sum(-1, keepdim=True)
        P13 = (diff_q_emb * diff_a_emb).sum(-1, keepdim=True)
        P22 = (diff_t_avg_emb * diff_t_avg_emb).sum(-1, keepdim=True)
        P23 = (diff_t_avg_emb * diff_a_emb).sum(-1, keepdim=True)
        P33 = (diff_a_emb * diff_a_emb).sum(-1, keepdim=True)
        P = torch.cat((P11, P12, P13, P12, P22, P23, P13, P23, P33), dim=-1)
        W_p = torch.matmul(self.W_p_params.unsqueeze(-1), self.W_p_params.unsqueeze(-2)).reshape(ARGS.product_layer_dim, -1)
        l_p = torch.matmul(P, W_p.transpose(1, 0))

        # question information
        question_emb = F.relu(l_z + l_p + self.bias)
        question_emb_dropped_out = self.dropout(question_emb)
        diff_hat = self.last_linear(question_emb_dropped_out).squeeze(-1)

        return QT_logit, QQ_logit, TT_logit, diff_hat, question_emb
