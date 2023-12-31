'''
Alipay.com Inc.
Copyright (c) 2004-2023 All Rights Reserved.
'''
from torch import nn
import torch.nn.functional as F
import torch
class dual_softmax_loss(nn.Module):
    def __init__(self,):
        super(dual_softmax_loss, self).__init__()
        
    def forward(self, sim_matrix, sim_matrix1, temp=10):
        sim_matrix = sim_matrix * F.softmax(sim_matrix1/temp, dim=0)*len(sim_matrix) #With an appropriate temperature parameter, the model achieves higher performance
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        loss = -logpt
        return loss