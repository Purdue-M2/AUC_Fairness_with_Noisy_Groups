import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from abc import ABC
import scipy
import numpy as np
import torch.optim as optim
from itertools import product

class Linear(nn.Module):
    def __init__(self, n_feat):
        super(Linear, self).__init__()
        self.n_feat = n_feat
        self.fc = nn.Linear(self.n_feat, 1, bias=True)
        # nn.init.xavier_normal_(self.fc.weight)
    def forward(self, x):
        out = self.fc(x)
        return out

class MLP(nn.Module):
    """
    A feedforward NN in pytorch using ReLU activiation functions between all layers
    The output layer uses BatchNorm
    Supports an arbitrary number of hidden layers
    """

    def __init__(self, h_sizes):
        """
        :param h_sizes: input sizes for each hidden layer (including the first)
        :param out_size: defaults to 1 for binary and represents the (positive class probability?)
        :param task: 'classification' or 'regression'
        """
        super(MLP, self).__init__()

        # Hidden layers
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1], bias=True))

        # Output layer
        self.out = nn.Linear(h_sizes[-1], 1, bias=True)
        self.relu = torch.nn.ReLU()
        self.bn = nn.BatchNorm1d(1)

    def forward(self, x):
        # Feedforward
        for layer in self.hidden:
            x = self.relu(layer(x))
        output = self.bn(self.out(x))
        return output

class SummaryWritter:
    '''
    For writting stat and result by experiments
    '''
    def __init__(self, X_train, y_train, z_train):

        # save the input
        try:
            self.n_samples = X_train.shape[0]
            self.n_features = X_train.shape[1]
        except:
            self.n_samples = len(y_train)
            self.n_features = 0

        self.n_labels = len(np.unique(y_train))
        assert self.n_labels == 2, "Only support binary output!"
        self.n_groups = len(np.unique(z_train))
        # get the number of positive and negative samples
        self.n_neg = len(y_train[(y_train < 0.5)])
        self.n_pos = len(y_train[(y_train > 0.5)])
        # get the number of each grouplabel
        self.n_grouplabel = []
        # order: group neg pos
        for i in range(self.n_groups):
            self.n_grouplabel.append(len(z_train[(y_train < 0.5) & (z_train == i)]))
            self.n_grouplabel.append(len(z_train[(y_train > 0.5) & (z_train == i)]))
        # reshaped index
        n_grouplabel_cs = np.cumsum(self.n_grouplabel).tolist()
        n_grouplabel_cs = [0] + n_grouplabel_cs
        self.grouplabel_ind = []
        self.group_ind = []
        self.label_ind = []
        neg_ind = []
        pos_ind = []
        # order: group neg pos
        for i in range(self.n_groups):
            group_neg_ind = list(range(n_grouplabel_cs[2*i] , n_grouplabel_cs[2*i+1]))
            group_pos_ind = list(range(n_grouplabel_cs[2*i+1] , n_grouplabel_cs[2*i+2]))
            neg_ind += group_neg_ind
            pos_ind += group_pos_ind
            self.grouplabel_ind.append(group_neg_ind)
            self.grouplabel_ind.append(group_pos_ind)
            self.group_ind.append(group_neg_ind + group_pos_ind)
        self.label_ind.append(neg_ind)
        self.label_ind.append(pos_ind)

    def grouplabelsplitter(self, X_train, y_train, z_train):

        X_train_list = []

        for i in range(self.n_groups):
            X_train_group_neg = X_train[(y_train < 0.5) & (z_train == i)]
            X_train_group_pos = X_train[(y_train > 0.5) & (z_train == i)]
            X_train_list.append(X_train_group_neg)
            X_train_list.append(X_train_group_pos)
        
        return X_train_list
    
def model_helper(size):
    if isinstance(size, int):
        return Linear(size)
    elif isinstance(size, list):
        return MLP(size)
    else:
        raise ValueError('Invalid size type!')
    
def compute_phats(X_train_list, train_writter):
    X_train = torch.cat(X_train_list)
    num_groups = len(X_train_list)
    num_datapoints = X_train.shape[0]
    
    neg_group = []
    pos_group = []
    for i in range(num_groups):
        if (i+2) % 2 == 0:
            neg_group.append(X_train_list[i])
        else:
            pos_group.append(X_train_list[i])
    
    phats=[]
    X_phats_list = []
    for (i, j) in product(range(train_writter.n_groups), range(train_writter.n_groups)):
        # load the data from ith group and jth group
        X_group_neg = neg_group[i]
        X_group_pos = pos_group[j]
        X_single_phat = torch.cat((X_group_neg, X_group_pos), dim=0)
        X_phats_list.append(X_single_phat)
        phat = float(1 / (X_single_phat.shape[0]))
        phats.append(phat)
        
    return phats
    
def np2torch(X_train, train_writter, batchsize, device, get_loader=True):

        X_train_list = []
        group_neg_loaders = []
        group_pos_loaders = []
        
        # num_group_list = []

        # batch_size_gender = [3249, 6751] #adult
        # batch_size_gender = [442, 9558] #1598 34570 batch_size_gender = [3980, 6020] #bank 14398 21770 36168 
        # batch_size_gender = [651, 349] #5304 2849 batch_size_gender = [198, 802] #compas 1613 6540 8153
        batch_size_gender = [6038, 3962] #default 14490 9510 24000
        for i in range(train_writter.n_groups):

            try:
                X_train_neg = torch.from_numpy(X_train[2*i]).to(device) #(num_female/male_neg, 102)
                X_train_pos = torch.from_numpy(X_train[2*i+1]).to(device)#(num_female/male_pos, 102)
            except TypeError:
                values = X_train[2*i].data
                indices = np.vstack((X_train[2*i].row, X_train[2*i].col))
                ind = torch.LongTensor(indices)
                v = torch.FloatTensor(values)
                shape = X_train[2*i].shape
                X_train_neg = torch.sparse.FloatTensor(ind, v, torch.Size(shape)).to(device)
                
                values = X_train[2*i+1].data
                indices = np.vstack((X_train[2*i+1].row, X_train[2*i+1].col))
                ind = torch.LongTensor(indices)
                v = torch.FloatTensor(values)
                shape = X_train[2*i+1].shape
                X_train_pos = torch.sparse.FloatTensor(ind, v, torch.Size(shape)).to(device)

            X_train_list.append(X_train_neg)
            X_train_list.append(X_train_pos)
            

            if get_loader:
                
                group_neg_train_set = TensorDataset(X_train_neg)
                group_pos_train_set = TensorDataset(X_train_pos)
                
                
                group_neg_loaders.append(DataLoader(group_neg_train_set, batch_size=max(1, int(batchsize * train_writter.n_grouplabel[2*i] / train_writter.n_samples)), drop_last=True))
                group_pos_loaders.append(DataLoader(group_pos_train_set, batch_size=max(1, int(batch_size_gender[i] - int(batchsize * train_writter.n_grouplabel[2*i] / train_writter.n_samples))), drop_last=True))
                

        #len(X_train_list) == 4
        #X_train: female_neg, female_pos, male_neg, male_pos
        #compute phats
        phats = compute_phats(X_train_list, train_writter)
        
        
        X_train = torch.cat(X_train_list)
        
        # get label because of bce
        with torch.no_grad():
            model = model_helper(X_train.shape[1]).to(device)
            model.eval()
            y_pred_train = model.forward(X_train).view(-1)
            y_train = torch.zeros_like(y_pred_train)
            y_train[train_writter.label_ind[1]] = 1.
        # print(X_train_list[0].shape, '0')
        # print(X_train_list[1].shape, '1')
        # print(X_train_list[2].shape, '2')
        # print(X_train_list[3].shape, '3')
        # exit()

        return X_train, y_train, group_neg_loaders, group_pos_loaders, phats, X_train_list

def np2torch_uniform(X_train, train_writter, batchsize, device, get_loader=True):
    X_train_list = []
    group_neg_loaders = []
    group_pos_loaders = []

    for i in range(train_writter.n_groups):
        try:
            # Convert group data to PyTorch tensors
            X_train_neg = torch.from_numpy(X_train[2 * i]).to(device)
            X_train_pos = torch.from_numpy(X_train[2 * i + 1]).to(device)
        except TypeError:
            # Handle sparse data
            values = X_train[2 * i].data
            indices = np.vstack((X_train[2 * i].row, X_train[2 * i].col))
            ind = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = X_train[2 * i].shape
            X_train_neg = torch.sparse.FloatTensor(ind, v, torch.Size(shape)).to(device)
            
            values = X_train[2 * i + 1].data
            indices = np.vstack((X_train[2 * i + 1].row, X_train[2 * i + 1].col))
            ind = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = X_train[2 * i + 1].shape
            X_train_pos = torch.sparse.FloatTensor(ind, v, torch.Size(shape)).to(device)

        X_train_list.append(X_train_neg)
        X_train_list.append(X_train_pos)

        # Create uniform data loaders (ignoring group proportions)
        if X_train_neg.size(0) > 0:  # Only create DataLoader if group has samples
            group_neg_train_set = TensorDataset(X_train_neg)
            group_neg_loaders.append(DataLoader(group_neg_train_set, batch_size=batchsize, shuffle=True, drop_last=False))
        else:
            print(f"Group {i} (negative samples) has no data.")
        
        if X_train_pos.size(0) > 0:  # Only create DataLoader if group has samples
            group_pos_train_set = TensorDataset(X_train_pos)
            group_pos_loaders.append(DataLoader(group_pos_train_set, batch_size=batchsize, shuffle=True, drop_last=False))
        else:
            print(f"Group {i} (positive samples) has no data.")

    # Combine data for phats calculation
    phats = compute_phats(X_train_list, train_writter)
    X_train = torch.cat(X_train_list)

    # Generate labels if required for BCE loss
    with torch.no_grad():
        model = model_helper(X_train.shape[1]).to(device)
        model.eval()
        y_pred_train = model.forward(X_train).view(-1)
        y_train = torch.zeros_like(y_pred_train)
        y_train[train_writter.label_ind[1]] = 1.
    # print(len(group_neg_loaders), '00000')
    # print(len(group_pos_loaders), '111111')
    # exit()
    return X_train, y_train, group_neg_loaders, group_pos_loaders, phats, X_train_list

    
def project_multipliers_wrt_euclidean_norm(multipliers, radius):
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
                        max(1.0, torch.sum(inactive)).item())
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