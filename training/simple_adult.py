import torch
import torch.nn as nn
from itertools import product
import torch.nn.functional as F

class AuxBCEWithLogitLoss(nn.Module):

    def __init__(self, reduction='mean', weight=None, k=None):
        super(AuxBCEWithLogitLoss, self).__init__()
        self.r = reduction
        self.w = weight
        self.k = k

    def forward(self, y_pred, y):
        y_stab = torch.clamp(torch.sigmoid(y_pred), min=1e-6, max=1-1e-6)
        if self.w is not None:
            bce = - self.w * (y * torch.log(y_stab) + (1. - y) * torch.log(1. - y_stab))
        else:
            bce = - (y * torch.log(y_stab) + (1. - y) * torch.log(1. - y_stab))
        if self.k:
            bce = torch.topk(bce, self.k)[0]
        if self.r == 'none':
            return bce
        elif self.r == 'mean':
            return bce.mean()
        elif self.r == 'sum':
            return bce.sum()
        else:
            return
        
class pAUCLoss(nn.Module):

    def __init__(self, k=False, reduction='mean', norm=False):
        super(pAUCLoss, self).__init__()
        self.reduction = reduction
        self.norm = norm
        self.k = k

    def forward(self, score_neg, score_pos):

        y_pos = torch.reshape(score_pos, (-1, 1))
        y_neg = torch.reshape(score_neg, (-1, 1))
        yy = y_pos - torch.transpose(y_neg, 0, 1) # [n_pos, n_neg] 2d-tensor
        bce = - torch.log(torch.clamp(torch.sigmoid(yy), min=1e-6, max=1-1e-6)) # [n_pos, n_neg] 2d-tensor
        if self.k:
            bce = torch.topk(bce, self.k)[0]
        if self.reduction == 'mean':
            if self.norm:
                return bce.mean()
            else:
                return bce.sum() / (len(score_pos) * len(score_neg))
        elif self.reduction == 'sum':
            return bce.sum()
        elif self.reduction == 'none':
            return bce
        else:
            return
        
        
def loss_helper(loss_type):
    if loss_type == 'bce':
        return AuxBCEWithLogitLoss()
    elif loss_type == 'pauc':
        return pAUCLoss()
    else:
        raise ValueError('Invalid loss type!')
    
def get_auc_constraints_dro(model, train_writter, yy_group, protected_label, loss_overall, label):
    losses = []
    
    criterion = loss_helper('pauc')
    for idx, (i, j) in enumerate(product(range(train_writter.n_groups), range(train_writter.n_groups))):
        loss = criterion(yy_group[2 * idx], yy_group[2 * idx + 1])
        loss = loss.squeeze(-1)
        losses.append(loss)
    
    constraint_list = []
    for idx, (p_tilde, loss) in enumerate(zip(model.p_tildes, losses)):
        weights = p_tilde.unsqueeze(1)
        loss = weights * loss
        #Avoid division by zero if all weights are zero
        if torch.sum(weights) == 0:
            constraint = torch.tensor(0.0, requires_grad=True)
        else:
            constraint = loss.sum() / torch.sum(weights)
        new_constraint = loss_overall - constraint 
        constraint_list.append(new_constraint)
    
    # prepare loss for model parameter
    constraint = torch.stack(constraint_list)
    
    
    return constraint


def get_auc_constraints(model, train_writter, yy_group, protected_label, loss_overall, label):
    losses = []
    
    criterion = loss_helper('pauc')
    for idx, (i, j) in enumerate(product(range(train_writter.n_groups), range(train_writter.n_groups))):
        loss = criterion(yy_group[2 * idx], yy_group[2 * idx + 1])
        loss = loss.squeeze(-1)
        losses.append(loss)
        # print(loss, '11111111')
    # exit()
    
    constraint_list = []
    for idx, (p_tilde, loss) in enumerate(zip(model.p_tildes, losses)):
        weights = protected_label.unsqueeze(1)
        loss = weights * loss
        # Avoid division by zero if all weights are zero
        if torch.sum(weights) == 0:
            constraint = torch.tensor(0.0, requires_grad=True)
        else:
            constraint = loss.sum() / torch.sum(weights)
        new_constraint = loss_overall - constraint
        constraint_list.append(new_constraint)
    # prepare loss for model parameter
    # print(constraint_list)
    constraint = torch.stack(constraint_list)
    
    
    return constraint


