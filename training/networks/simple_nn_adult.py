import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from training.dataset.data_utils import project_multipliers_wrt_euclidean_norm

class SimpleNN(nn.Module):
    def __init__(self,input_size, num_groups, batch_size, method):
        super(SimpleNN, self).__init__()
        # Define a simple feedforward network with two hidden layers
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)  # Output layer for binary classification
        
        # Initialize lambdas and p_tildes as trainable parameters
        self.num_groups = num_groups
        if method == 'auc':
            self.lambdas = nn.Parameter(torch.zeros(self.num_groups, dtype=torch.float32, requires_grad=True))
        else:
            self.lambdas = nn.Parameter(torch.zeros(self.num_groups, dtype=torch.float32, requires_grad=True))
        # Initialize p_tildes based on the length of data_iter
        if batch_size is not None:
            self.num_data = 2 * batch_size
            # self.p_tildes = nn.ParameterList([torch.full((self.num_data,), 1e-5, dtype=torch.float32, requires_grad=True) for _ in range(self.num_groups)])
            self.p_tildes = nn.ParameterList([nn.Parameter(torch.zeros(self.num_data, dtype=torch.float32, requires_grad=True)) for _ in range(self.num_groups)])
        else: 
            self.p_tildes = None

        self.maximum_lambda_radius = 1.0
        # self.maximum_p_radius = [0.8,0.8,0.8,0.8] #noise_ratio (\gamma) * 2
        self.maximum_p_radius = [1.0,1.0,1.0,1.0] #noise_ratio (\gamma) * 2
        
           
        
    def forward(self, x):
        self.input_size = len(x)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output(x)
        
        return x
    

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
    
    def get_train_metrics(self, predictions, labels):
        """ Error rate for predictions and binary labels.

        Args:
            predictions: numpy array of floats representing predictions. Predictions are treated 
            as positive classification if value is >0, and negative classification if value is <= 0.
            labels: numpy array of floats representing labels. labels are also treated as positive 
            classification if value is >0, and negative classification if value is <= 0.

        Returns: float, error rate of predictions classifications compared to label classifications.
        """
        error_rates=[]
        for i in range(len(predictions)):
            # predictions = torch.sigmoid(predictions[i])
            # labels = torch.sigmoid(labels[i])
            signed_labels = (
                (labels[i] > 0).float() - (labels[i] <= 0).float())
            numerator = (torch.mul(signed_labels, predictions[i]) <= 0).sum().item()
            denominator = predictions[i].shape[0]
            error_rate = float(numerator) / float(denominator)
            error_rates.append(error_rate)
        
        error_rate = sum(error_rates) / len(error_rates)
        return error_rate