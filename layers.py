import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math

class GraphAttention(Module):

    def __init__(self, in_features, out_features, concat, alpha, bias=True):
        super(GraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(alpha)

        # 注意力系数
        self.a = Parameter(torch.FloatTensor(2 * out_features, 1))

        self.W = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.a.data.uniform_(stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # z = W*h, 2708*8
        z = torch.mm(input, self.W)
        # 2708*2708*16
        z_ConcatAll = self.ConcatAll(z)
        # squeeze(2): 把第三维去掉，从2708*2708*1 压到2708*2708
        e = self.leakyrelu(torch.matmul(z_ConcatAll, self.a).squeeze(2))
        # 创建2708*2708的由极小值替代0的矩阵
        zero2min = -9e15 * torch.ones_like(e)
        # 只和邻居相关的系数eij
        eij = torch.where(adj > 0, e, zero2min)

        # 得到eij后再由公式的注意力系数矩阵，2708*2708
        attention = F.softmax(eij)

        # 2708*8
        output = torch.matmul(attention, z)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


    def ConcatAll(self, z):
        # num of node
        N = z.size()[0]

        # z = [f1, f2, .... fn].T (N * nfeat) ,fi is the feature of i node

        # z_repeated_same = [f1,...,f1,f2,...f2,....,fn,...,fn].T (N**2, nfeat)
        z_repeated_same = z.repeat_interleave(N, dim=0)

        # z_repeated_alternating = [f1,...,fn,f1,...fn,....,f1,...,fn].T (N**2, nfeat)
        z_repeated_alternating = z.repeat(N,1)

        # z_repeated_concat = [[f1,...,f1,f2,...,f2,....,fn,...fn],
        #                      [f1,...fn,f1,...,fn,....,f1,...,fn]].T (N**2, 2*nfeat)
        z_repeated_concat = torch.cat([z_repeated_same, z_repeated_alternating], dim=1)


        return z_repeated_concat.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'