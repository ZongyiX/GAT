import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttention

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, alpha, nheads, dropout):
        super(GAT, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.nheads = nheads

        self.gat1 = [GraphAttention(nfeat, nhid, concat=True, alpha=alpha) for _ in range(nheads)]
        for i, attention in enumerate(self.gat1):
            self.add_module('attention_{}'.format(i), attention)

        self.gat2 = GraphAttention(nhid * nheads, nclass, concat=False, alpha=alpha)
        self.dropout = dropout

    def forward(self, x, adj):
        # 如果先前没有dropout，那么准确率只有0.75， 加了有0.81
        x = F.dropout(x, self.dropout, training=self.training)
        # 单个输出x是2708*8的矩阵,按列（dim=1) 拼接起来, 2708*(8*nheads)
        x = torch.cat([att(x, adj) for att in self.gat1], dim=1)
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(x, adj)
        return F.log_softmax(x, dim=1)
