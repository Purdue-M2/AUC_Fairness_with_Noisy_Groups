
import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from scipy import optimize
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from simple_adult import loss_helper
from simple_adult import get_auc_constraints_dro
from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='auc_res')
class AucResDetector(AbstractDetector):
    def __init__(self):
        super().__init__()
        self.backbone = self.build_backbone()
        self.loss_func = self.build_loss()
        
        ##########dro_robustness########
        self.num_groups = 4
        self.num_data =  None
        self.constraints_slack = 1.0
        self.maximum_lambda_radius = 1.0
        self.maximum_p_radius = [0.04,0.04,0.04,0.04] #noise_ratio * 2
        # Initialize Lagrange multipliers and probability distributions
        self.lambdas = nn.Parameter(torch.zeros(self.num_groups, dtype=torch.float32, requires_grad=True))
       
       # Initialize p_tilde variables with a placeholder
        if self.num_data is not None:
            # self.p_tildes = nn.ParameterList([nn.Parameter(torch.zeros(2*self.num_data, dtype=torch.float32,requires_grad=True)) for _ in range(self.num_groups)])
            self.p_tildes = nn.ParameterList([torch.full((2*self.num_data,), 1e-5, dtype=torch.float32, requires_grad=True) for _ in range(self.num_groups)])
        else:
            self.p_tildes = None
    def initialize_p_tildes(self, num_data):
        self.num_data = num_data
        self.p_tildes = nn.ParameterList([torch.full((2*self.num_data,), 1e-5, dtype=torch.float32, requires_grad=True) for _ in range(self.num_groups)])
        # self.p_tildes = nn.ParameterList([nn.Parameter(torch.zeros(2*self.num_data, dtype=torch.float32,requires_grad=True)) for _ in range(self.num_groups)])
        # self.optimizer_p_list = [optim.SGD([p], lr=lr, momentum=0.9, weight_decay=5e-3) for p, lr in zip(self.p_tildes, self.learning_rate_p_list)]
 
    def build_backbone(self):    
        # prepare the backbone
        backbone_class = BACKBONE['resnet50']
        backbone = backbone_class({'mode': 'original',
                                   'num_classes': 1, 'inc': 3, 'dropout': False})
        # if donot load the pretrained weights, fail to get good results
        state_dict = torch.load('./pretrained/resnet50-19c8e357.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, False)
        print('Load pretrained model successfully!')
        return backbone
        
    
    def build_loss(self):
        # prepare the loss function
        loss_class = LOSSFUNC['daw_bce']
        loss_func = loss_class()
        return loss_func

    def threshplus_tensor(self, x):
        y = x.clone()
        pros = torch.nn.ReLU()
        z = pros(y)
        return z
    
    def search_func(self, losses, alpha):
        return lambda x: x + (1.0/alpha)*(self.threshplus_tensor(losses-x).mean().item())

    def searched_lamda_loss(self, losses, searched_lamda, alpha):
        return searched_lamda + ((1.0/alpha)*torch.mean(self.threshplus_tensor(losses-searched_lamda))) 
    
    def features(self, data_dict: dict) -> torch.tensor:
        return self.backbone.features(data_dict['image']) #32,3,256,256

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        # defualt 0.9
        inner_alpha = 0.9
        outer_alpha = 0.5
        label = data_dict['label']
        intersec_label = data_dict['intersec_label']
        pred = pred_dict['cls']
        outer_loss = []
        inter_index = list(torch.unique(intersec_label))
        loss_entropy = self.loss_func(pred, label)
        for index in inter_index:
            ori_inter_loss = loss_entropy[intersec_label == index]
            # print(ori_inter_loss,'ori_inter_loss')
            lamda_i_search_func = self.search_func(ori_inter_loss,inner_alpha)
            searched_lamda_i = optimize.fminbound(lamda_i_search_func, np.min(ori_inter_loss.cpu().detach().numpy()) - 1000.0, np.max(ori_inter_loss.cpu().detach().numpy()))
            inner_loss = self.searched_lamda_loss(ori_inter_loss, searched_lamda_i, inner_alpha)
            outer_loss.append(inner_loss)
        outer_loss = torch.stack(outer_loss)
        lamda_search_func = self.search_func(outer_loss, outer_alpha)
        searched_lamda = optimize.fminbound(lamda_search_func, np.min(outer_loss.cpu().detach().numpy()) - 1000.0, np.max(outer_loss.cpu().detach().numpy()))
        loss = self.searched_lamda_loss(outer_loss, searched_lamda, outer_alpha)
        loss_dict = {'overall': loss}
        return loss_dict
    
    
    def get_auc_dro_loss(self, data_dict: dict, pred_dict: dict, device) -> dict:
        label = data_dict['label']
        intersec_label = data_dict['intersec_label'].detach().clone()
        intersec_label[intersec_label == -1] = 0
        pred = pred_dict['cls']
        criterion = loss_helper('pauc')
        
        # Separate yy_pred based on the ground truth labels in yy_batch
        yy_pos = pred[label == 1]  # Predictions for positive labels
        yy_neg = pred[label == 0]  # Predictions for negative labels
        overall_loss = criterion(yy_neg, yy_pos)
        # print(loss,'2222222')
        
        
        
        losses_dro = []
        preds = torch.chunk(pred, 4, dim=0)  # Split into 4 equal parts along the first dimension (batch dimension)

        #0: fake_male, 1: fake_female, 2: real_male, 3: real_female
        loss_dro_1 = criterion(preds[2], preds[0])
        loss_dro_1 = loss_dro_1.squeeze(-1)
        losses_dro.append(loss_dro_1)
        loss_dro_2 = criterion(preds[3], preds[0])
        loss_dro_2 = loss_dro_2.squeeze(-1) 
        losses_dro.append(loss_dro_2)
        loss_dro_3 = criterion(preds[2], preds[1])
        loss_dro_3 = loss_dro_3.squeeze(-1)
        losses_dro.append(loss_dro_3)
        loss_dro_4 = criterion(preds[3], preds[1])
        loss_dro_4 = loss_dro_4.squeeze(-1)
        losses_dro.append(loss_dro_4)
        mean_loss_dro = torch.mean(torch.stack(losses_dro))
        loss_dict = {'overall': mean_loss_dro}
        
        constraint_list = []
        for idx, (p_tilde, loss) in enumerate(zip(self.p_tildes, losses_dro)):
            p_tilde = p_tilde.to(device)
            weights = p_tilde.unsqueeze(1)
            loss = weights * loss
            #Avoid division by zero if all weights are zero
            if torch.sum(weights) == 0:
                constraint = torch.tensor(0.0, requires_grad=True)
            else:
                constraint = loss.sum() / torch.sum(weights)
            new_constraint = overall_loss - constraint
            constraint_list.append(new_constraint)
        # prepare loss for model parameter
        constraint = torch.stack(constraint_list)
        return loss_dict, constraint
    
    def get_auc_loss(self, data_dict: dict, pred_dict: dict, device) -> dict:
        label = data_dict['label']
        intersec_label = data_dict['intersec_label'].detach().clone()
        intersec_label[intersec_label == -1] = 0
        pred = pred_dict['cls']
        criterion = loss_helper('pauc')
        
        yy_pos = pred[label == 1]  # Predictions for positive labels
        yy_neg = pred[label == 0]  # Predictions for negative labels
        loss = criterion(yy_neg, yy_pos)
        
        
        
        losses_dro = []
        preds = torch.chunk(pred, 4, dim=0)  

        #0: fake_male, 1: fake_female, 2: real_male, 3: real_female
        loss_dro_1 = criterion(preds[2], preds[0])
        loss_dro_1 = loss_dro_1.squeeze(-1)
        losses_dro.append(loss_dro_1)
        loss_dro_2 = criterion(preds[3], preds[0])
        loss_dro_2 = loss_dro_2.squeeze(-1) 
        losses_dro.append(loss_dro_2)
        loss_dro_3 = criterion(preds[2], preds[1])
        loss_dro_3 = loss_dro_3.squeeze(-1)
        losses_dro.append(loss_dro_3)
        loss_dro_4 = criterion(preds[3], preds[1])
        loss_dro_4 = loss_dro_4.squeeze(-1)
        losses_dro.append(loss_dro_4)
        mean_loss_dro = torch.mean(torch.stack(losses_dro))
        loss_dict = {'overall': mean_loss_dro}
        
        
        constraint_list = []
        for idx, (p_tilde, constraint) in enumerate(zip(self.p_tildes, losses_dro)):
            p_tilde = p_tilde.to(device)
            new_constraint = loss - ( intersec_label * constraint).mean()
            constraint_list.append(new_constraint)
        # prepare loss for model parameter
        constraint = torch.stack(constraint_list)
        return loss_dict, constraint
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        pred = pred.squeeze(1)
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    
    def get_test_metrics(self):
        pass


    def forward(self, data_dict: dict, inference=False) -> dict:
        # Update self.num_data based on the input data
        if self.p_tildes is None:
            self.initialize_p_tildes(len(data_dict['label']))
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)

        pred_dict = {'cls': pred}

        return pred_dict
    
    def project_ptilde(self, phats, ptilde, idx):
        # print(len(phats[0]))
        current_batch_size = phats.shape[1]
        phat = phats[idx, :current_batch_size].to(ptilde.device)
        # phat = phat.view(-1)
        ptilde_sliced = ptilde[:current_batch_size] # Slice ptilde to match the current batch size
        projected_ptilde = self.project_multipliers_to_L1_ball(ptilde_sliced, phat, self.maximum_p_radius[idx])
        return projected_ptilde
    
    def project_multipliers_to_L1_ball(self, multipliers, center, radius):
        """Projects its argument onto the feasible region.
        The feasible region is the set of all vectors in the L1 ball with the given center multipliers and given radius.
        Args:
            multipliers: rank-1 `Tensor`, the Lagrange multipliers to project.
            radius: float, the radius of the feasible region.
            center: rank-1 `Tensor`, the Lagrange multipliers as the center.
        Returns:
            The rank-1 `Tensor` that results from projecting "multipliers" onto a L1 norm ball w.r.t. the Euclidean norm.
            The returned rank-1 `Tensor` is in a simplex.
        Raises:
            TypeError: if the "multipliers" `Tensor` is not floating-point.
            ValueError: if the "multipliers" `Tensor` does not have a fully-known shape,
            or is not one-dimensional.
        """
        assert radius >= 0
        # Compute the offset from the center and the distance
        offset = multipliers - center
        # print()
        dist = torch.abs(offset)
        
        # Project multipliers on the simplex
        new_dist = self.project_multipliers_wrt_euclidean_norm(dist, radius)
        signs = torch.sign(offset)
        new_offset = signs * new_dist
        projection = center + new_offset
        projection = torch.clamp(projection, min=0.0)
        
        return projection
    
    def project_multipliers_wrt_euclidean_norm(self, multipliers, radius):
        """Projects its argument onto the feasible region.
        The feasible region is the set of all vectors with nonnegative elements that
        sum to at most "radius".
        Args:
            multipliers: rank-1 `Tensor`, the Lagrange multipliers to project.
            radius: float, the radius of the feasible region.
        Returns:
            The rank-1 `Tensor` that results from projecting "multipliers" onto the
            feasible region w.r.t. the Euclidean norm.
        Raises:
            TypeError: if the "multipliers" `Tensor` is not floating-point.
            ValueError: if the "multipliers" `Tensor` does not have a fully-known shape,
            or is not one-dimensional.
        """
        if not torch.is_floating_point(multipliers):
            raise TypeError("multipliers must have a floating-point dtype")
        multipliers_dims = multipliers.shape
        if multipliers_dims is None:
            raise ValueError("multipliers must have a known rank")
        if len(multipliers_dims) != 1:
            raise ValueError("multipliers must be rank 1 (it is rank %d)" % len(multipliers_dims))
        dimension = multipliers_dims[0]
        if dimension is None:
            raise ValueError("multipliers must have a fully-known shape")

        iteration = 0
        inactive = torch.ones_like(multipliers, dtype=multipliers.dtype)
        old_inactive = torch.zeros_like(multipliers, dtype=multipliers.dtype)

        while True:
            iteration += 1
            scale = min(0.0, (radius - torch.sum(multipliers)).item() /
                        max(1.0, torch.sum(inactive) if isinstance(torch.sum(inactive), torch.Tensor) else torch.sum(inactive)))
            multipliers = multipliers + (scale * inactive)
            new_inactive = (multipliers > 0).to(multipliers.dtype)
            multipliers = multipliers * new_inactive

            not_done = (iteration < dimension)
            not_converged = torch.any(inactive != old_inactive)

            if not (not_done and not_converged):
                break

            old_inactive = inactive
            inactive = new_inactive

        return multipliers
    
    
    def project_lambdas(self, lambdas):
        """Projects the Lagrange multipliers onto the feasible region."""
        if self.maximum_lambda_radius:
            projected_lambdas = self.project_multipliers_wrt_euclidean_norm_handlefloat(
                lambdas, self.maximum_lambda_radius)
        else:
            projected_lambdas = torch.clamp(lambdas, min=0.0)
        return projected_lambdas   
    
    def project_multipliers_wrt_euclidean_norm_handlefloat(self, multipliers, radius):
        """Projects its argument onto the feasible region.
        The feasible region is the set of all vectors with nonnegative elements that
        sum to at most "radius".
        Args:
            multipliers: rank-1 `Tensor`, the Lagrange multipliers to project.
            radius: float, the radius of the feasible region.
        Returns:
            The rank-1 `Tensor` that results from projecting "multipliers" onto the
            feasible region w.r.t. the Euclidean norm.
        Raises:
            TypeError: if the "multipliers" `Tensor` is not floating-point.
            ValueError: if the "multipliers" `Tensor` does not have a fully-known shape,
            or is not one-dimensional.
        """
        if not torch.is_floating_point(multipliers):
            raise TypeError("multipliers must have a floating-point dtype")
        multipliers_dims = multipliers.shape
        if multipliers_dims is None:
            raise ValueError("multipliers must have a known rank")
        if len(multipliers_dims) != 1:
            raise ValueError("multipliers must be rank 1 (it is rank %d)" % len(multipliers_dims))
        dimension = multipliers_dims[0]
        if dimension is None:
            raise ValueError("multipliers must have a fully-known shape")

        iteration = 0
        inactive = torch.ones_like(multipliers, dtype=multipliers.dtype)
        old_inactive = torch.zeros_like(multipliers, dtype=multipliers.dtype)

        while True:
            iteration += 1
            scale = min(0.0, (radius - torch.sum(multipliers)) /
                        max(1.0, torch.sum(inactive)))
            multipliers = multipliers + (scale * inactive)
            new_inactive = (multipliers > 0).to(multipliers.dtype)
            multipliers = multipliers * new_inactive

            not_done = (iteration < dimension)
            not_converged = torch.any(inactive != old_inactive)

            if not (not_done and not_converged):
                break

            old_inactive = inactive
            inactive = new_inactive

        return multipliers