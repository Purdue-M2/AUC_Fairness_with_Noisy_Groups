
import torch
import torch.nn as nn
from loss.abstract_loss_func import AbstractLossClass
from utils.registry import LOSSFUNC

import torch.nn.functional as F
import numpy as np


@LOSSFUNC.register_module(module_name="balance")
class LDAMLoss(AbstractLossClass):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(
            self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

# @LOSSFUNC.register_module(module_name="balance")
# class LDAMLoss(AbstractLossClass):

#     def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
#         super().__init__()
#         self.device = torch.device('cuda:1')
#         m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
#         m_list = m_list * (max_m / np.max(m_list))
#         m_list = torch.FloatTensor(m_list).to(self.device)
#         self.m_list = m_list
#         assert s > 0
#         self.s = s
#         self.weight = weight

#     def forward(self, x, target):
#         # Move input tensors to the same device as self.device
#         x = x.to(self.device)
#         target = target.to(self.device)

#         # Create the index tensor on the same device and with boolean type
#         index = torch.zeros_like(x, dtype=torch.bool, device=self.device)
#         index.scatter_(1, target.data.view(-1, 1), 1)

#         # Convert index to float for multiplication and ensure it's on the correct device
#         index_float = index.float().to(self.device)
#         batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
#         batch_m = batch_m.view((-1, 1))
#         x_m = x - batch_m

#         output = torch.where(index, x_m, x)
#         return F.cross_entropy(self.s * output, target, weight=self.weight)
