import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import expm

class GraphLearner(nn.Module):
    def __init__(self, input_size, hidden_size, topk=None, epsilon=0, num_pers=16, metric_type='attention', device=None):
        """
            1 形参转私有
        """
        super(GraphLearner,self).__init__()
        self.device = device
        self.topk = topk
        self.epsilon = epsilon
        self.metric_type = metric_type

        """
            2 选择评价图相似度指标
        """
        if metric_type == 'weighted_cosine':
            self.weight_tensor = torch.Tensor(num_pers,input_size)
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))

            # print('[ Multi-perspective {} GraphLearner: {}]'.format(metric_type, num_pers))
        else:
            raise ValueError('Unknown metric_type: {}'.format(metric_type))

        # print('[ Graph Learner metric type: {} ]'.format(metric_type))


    """
        GL(X) or GL(Z(t-1)) : 由graph_clf.py中的graph_learner()调用
            输入 原始特征X(epoch=1)或者中间嵌入层特征Z(t-1)
            输出 GL(X) → A(1) | GL(Z(t-1)) → A(t)
    """
    def forward(self,context, isPhy=False, dist=None): # context (batch_size=7,station_num=178,feature_num=3)
        if self.metric_type == 'weighted_cosine':
            expand_weight_tensor = self.weight_tensor.unsqueeze(1) # (per_nums=4,1,feature_num=3)
            if len(context.shape) == 3 :
                expand_weight_tensor = expand_weight_tensor.unsqueeze(1) # (per_nums=4,1,1,feature_num=3)
            self.dist = dist
            context_fc = context.unsqueeze(0) * expand_weight_tensor # (per_nums=4,batch_size=7,station_num=178,feature_num=3)
            context_norm = F.normalize(context_fc,p=2,dim=-1)
            attention = torch.matmul(context_norm,context_norm.transpose(-1,-2)).mean(0) # (batch_size=7,station_num=178,feature_num=3)
            markoff_value = 0
        else:
            print("No weighted_cosin")

        if self.epsilon is not None:
            attention = self.build_epsilon_neighbourhood(attention,self.epsilon,markoff_value) # 基于ε-邻域的图稀疏化的余弦相似度矩阵
            #print(attention)
        if self.dist is not None and isPhy is True:
            attention = self.physical_domain_knowledge(attention,self.dist)

        return attention # 每轮batch的A(t) (batch_size=7,station_num=2160,feature_num=3)

    def physical_domain_knowledge(self,attention,dist):
        k = 258.3 #热阻系数 
        u1 = 1/2*torch.exp(torch.tensor(dist/k)).to(self.device).float() #热传导距离损耗
        #u1 = torch.tensor(1/2*expm(dist/k)).to(self.device).float() 
        u2 = 0.4*torch.ones(142,142).to(self.device) #长短波辐射差值均值,取一个近似
        u = u1 + u2
        #print(u)
        for x in range(attention.shape[0]-1):
                attention[x] = torch.mul(attention[x],u)
        return attention


    def build_epsilon_neighbourhood(self,attention,epsilon,markoff_value):
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask) # markoff_value = 0
        return weighted_adjacency_matrix
