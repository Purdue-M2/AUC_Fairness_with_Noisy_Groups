import sys
from detectors import DETECTOR
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from PIL import Image
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import os.path as osp
from log_utils import Logger
import torch.backends.cudnn as cudnn
from dataset.pair_dataset import  pairDatasetDro, FakeMaleDataset, FakeFemaleDataset, RealMaleDataset, RealFemaleDataset, ExactRatioOrderedBatchSampler, RatioWithReplacementBatchSampler
from dataset.datasets_train import ImageDataset_Train, ImageDataset_Test, ImageDataset_Test_Dro
import csv
import argparse
from tqdm import tqdm
import time
import os
from sam import SAM
from utils.bypass_bn import enable_running_stats, disable_running_stats
from metrics.base_metrics_class import calculate_metrics_for_train
from fairness_metrics import acc_fairness_dro
from transform import get_albumentations_transforms

parser = argparse.ArgumentParser("Example")

parser.add_argument('--lr', type=float, default=0.0005,
                    help="learning rate for training")
parser.add_argument('--train_batchsize', type=int, default=16, help="batch size")
parser.add_argument('--test_batchsize', type=int, default=32, help="test batch size")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--datapath', type=str,
                    default='../dataset/deepfakes/ff++_ori/')
parser.add_argument("--model", type=str, default='auc_effinet',
                    help="detector name[auc_effinet, auc_res,...]")
parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument("--dataset_type", type=str, default='pair',
                    help="detector name[pair,no_pair]")
parser.add_argument("--num_groups", type=int,
                        default='4')
parser.add_argument("--method", type=str,
                        default='auc')
parser.add_argument("--backbone", type=str,
                        default='efficientnetb4', help=['resnet50', 'xception', 'efficientnetb4'])
parser.add_argument('--pho', type=float, default=0.5,
                    help="learning parameter for SAM")
parser.add_argument('--ratio', type=float, default=0.02,
                    help="noise ratio")
#################################test##############################

parser.add_argument("--inter_attribute", type=str,
                    default='male-female')
parser.add_argument("--label_attribute", type=str,
                    default='pos-neg')
parser.add_argument("--single_attribute", type=str,
                    default='young-middle-senior-ageothers')
parser.add_argument("--test_datapath", type=str,
                        default='../dataset/deepfakes/ff++_ori/test.csv', help="test data path")
parser.add_argument("--savepath", type=str,
                        default='../results')

args = parser.parse_args()


###### import data transform #######
from transform import fair_df_default_data_transforms as data_transforms
test_transforms = get_albumentations_transforms([''])
###### load data ######
if args.dataset_type == 'pair':
    train_dataset = pairDatasetDro(args.datapath + 'faketrain_male.csv', args.datapath + 'faketrain_female.csv',args.datapath + 'realtrain_male.csv', args.datapath + 'realtrain_female.csv', data_transforms['train'])
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.train_batchsize, shuffle=True, num_workers=16, pin_memory=True, collate_fn=train_dataset.collate_fn)
    train_dataset_size = len(train_dataset)
    
else:
    train_dataset = ImageDataset_Train(args.datapath + 'train.csv', data_transforms['train'])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.train_batchsize, shuffle=True, num_workers=16, pin_memory=True)

    train_dataset_size = len(train_dataset)


device = args.device

# prepare the model (detector)
model_class = DETECTOR[args.model]


def cleanup_npy_files(directory):
    """
    Deletes all .npy files in the given directory.
    :param directory: The directory to clean up .npy files in.
    """
    for item in os.listdir(directory):
        if item.endswith(".npy"):
            os.remove(os.path.join(directory, item))
    print("Cleaned up .npy files in directory:", directory)


# train and evaluation
def train(model, optimizer_theta, optimizer_lambda, optimizer_p_list, scheduler, device, num_epochs, start_epoch):

    # Initialize best metric variables before the training loop
    best_auc = 0.0
    best_eer = 1.0  # Assuming lower EER is better
    best_accuracy = 0.0
    best_esauc = 0.0
    best_violation = 1.0
    best_min_over_max_auc = 0.0
    
    # Optionally, keep track of the epochs when the best metrics occurred
    best_auc_epoch = None
    best_eer_epoch = None
    best_accuracy_epoch = None
    best_ap_epoch = None
    best_esauc_epoch = None
    best_violation_epoch = None
    best_min_over_max_auc_epoch = None
    
    
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        phase = 'train'
        model.train()
        num_groups = 4

        total_loss = 0.0

        for idx, data_dict in enumerate(tqdm(train_dataloader)):
            
            # Zero out all gradients
            optimizer_theta.zero_grad()
            optimizer_lambda.zero_grad()
            for optimizer_p in optimizer_p_list:
                optimizer_p.zero_grad()

            imgs, labels, intersec_labels = data_dict['image'], data_dict[
                'label'], data_dict['intersec_label']
            data_dict['image'], data_dict['label'], data_dict['intersec_label'] = imgs.to(
                device), labels.to(device), intersec_labels.to(device),

            

            phat_list = []
            step = 2*args.train_batchsize
            start=0
            end = step
            group_num = [33549, 34919, 41220, 42590]
            for i in range(num_groups):
                phat = torch.zeros(2*data_dict['label'].size(0), device=args.device)
                phat[start : end] = float(1 / (group_num[i]))
                phat_list.append(phat)
                start += step
                end += step
            phats = torch.stack(phat_list, dim=0).to(args.device)
            
            with torch.set_grad_enabled(phase == 'train'):

                enable_running_stats(model)
                preds = model(data_dict)
                criterion = nn.BCELoss()
                # Calculate individual losses
                # Initialize p_tildes if not already initialized
                if model.p_tildes is None:
                    model.initialize_p_tildes(len(data_dict['label']))
                
                if args.method == 'naive':
                    if isinstance(model, torch.nn.DataParallel):
                        loss, constraints = model.module.get_auc_loss(data_dict, preds, device)
                    else:
                        loss, constraints = model.get_auc_loss(data_dict, preds, device)
                elif args.method == 'auc':
                    if isinstance(model, torch.nn.DataParallel):
                        loss, constraints = model.module.get_auc_dro_loss(data_dict, preds, device)
                    else:
                        loss, constraints = model.get_auc_dro_loss(data_dict, preds, device)
                ce = criterion(torch.sigmoid(preds['cls'].squeeze(1)), data_dict['label'].float())
                loss = loss['overall']
                lagrangian_loss = torch.dot(model.lambdas, constraints)
                # Total loss
                total_loss_batch = ce + loss + (lagrangian_loss)
                
                # Main backward pass
                total_loss_batch.backward(retain_graph=True)
                
                # Update main model parameters
                optimizer_theta.first_step(zero_grad=True)
                
                #second forward and backward
                disable_running_stats(model) 
                
                preds = model(data_dict)
                
                # Calculate individual losses
                
                if args.method == 'naive':
                    if isinstance(model, torch.nn.DataParallel):
                        loss, constraints = model.module.get_auc_loss(data_dict, preds, device)
                    else:
                        loss, constraints = model.get_auc_loss(data_dict, preds, device)
                elif args.method == 'auc':
                    if isinstance(model, torch.nn.DataParallel):
                        loss, constraints = model.module.get_auc_dro_loss(data_dict, preds, device)
                    else:
                        loss, constraints = model.get_auc_dro_loss(data_dict, preds, device)
                ce = criterion(torch.sigmoid(preds['cls'].squeeze(1)), data_dict['label'].float())
                loss = loss['overall']
                lagrangian_loss = torch.dot(model.lambdas, constraints)
                # Total loss
                total_loss_batch = ce + loss +  (lagrangian_loss)
                
                # Main backward pass
                total_loss_batch.backward(retain_graph=True)
                optimizer_theta.second_step(zero_grad=True)
                
                
                # Update lambda by accent
                lambda_grads = torch.autograd.grad(
                    torch.dot(model.lambdas, constraints), 
                    model.lambdas, 
                    retain_graph=True, 
                    allow_unused=True
                )[0]
                
                
                if lambda_grads is not None:
                    model.lambdas.grad = -lambda_grads
                    optimizer_lambda.step()
                    model.lambdas.data = model.project_lambdas(model.lambdas.data)
                
                # Reverse gradients for p and lambda to perform gradient ascent
                
                if args.method == 'auc' or 'fpr':
                    for p in model.p_tildes.parameters():
                        if p.grad is not None:
                            p.grad = -p.grad
                
                    for i, optimizer_p in enumerate(optimizer_p_list):
                        current_batch_size = phats[0].shape[0]
                        optimizer_p.step()
                        model.p_tildes[i].data[:current_batch_size] = model.project_ptilde(phats, model.p_tildes[i].data[:current_batch_size], i)
                        
            


            if idx % 200 == 0:
                # compute training metric for each batch data
                if isinstance(model, torch.nn.DataParallel):
                    batch_metrics = model.module.get_train_metrics(data_dict, preds)
                else:
                    batch_metrics = model.get_train_metrics(data_dict, preds)


                print('#{} batch_metric{}'.format(idx, batch_metrics))

            total_loss += total_loss_batch.item() * imgs.size(0)

        epoch_loss = total_loss / train_dataset_size
        print('Epoch: {} Loss: {:.4f}'.format(epoch, epoch_loss))

        # update learning rate
        if phase == 'train':
            scheduler.step()

        # evaluation

        if (epoch+1) % 1 == 0:

            savepath = 'YOUR_PATH_HERE'


            temp_model = savepath+"/"+args.method+str(epoch)+'.pth'
            os.makedirs(os.path.dirname(temp_model), exist_ok=True)
            torch.save(model.state_dict(), temp_model)

            print()
            print('-' * 10)

            phase = 'test'
            model.eval()
            
            print('Testing: ')
            print('-' * 10)

            interattributes = args.inter_attribute.split('-')
            label_attributes = args.label_attribute.split('-')
            for eachatt in interattributes:
                for label_attribute in label_attributes:
                    test_dataset = ImageDataset_Test_Dro(args.test_datapath,eachatt, label_attribute, test_transforms)
                    test_dataset_size = len(test_dataset)

                    test_dataloader = DataLoader(
                        test_dataset, batch_size=args.test_batchsize, shuffle=False,num_workers=32, pin_memory=False)
                    print('Testing: ')
                    print('-' * 10)
                    pred_list = []
                    label_list = []
                    intersec_label_list = []
                    for idx, data_dict in enumerate(tqdm(test_dataloader)):
                        imgs, labels, intersec_labels = data_dict['image'], data_dict['label'], data_dict['intersec_label']
                        data_dict['image'], data_dict['label'], data_dict['intersec_label'] = imgs.to(
                            device), labels.to(device), intersec_labels.to(device)
                        output = model(data_dict, inference=True)
                        pred = output['cls']
                        pred = pred.cpu().data.numpy().tolist()

                        pred_list += pred
                        label_list += labels.cpu().data.numpy().tolist()
                        intersec_label_list += intersec_labels.cpu().data.numpy().tolist()
                    label_list = np.array(label_list)
                    pred_list = np.array(pred_list)
                    intersec_label_list = np.array(intersec_label_list)
                    
                    savepath = args.savepath + '/' + eachatt + '_' + label_attribute
                    os.makedirs(savepath, exist_ok=True)  # Create directory if it doesn't exist
                    np.save(savepath + '/labels.npy', label_list)
                    np.save(savepath + '/predictions.npy', pred_list)
                    np.save(savepath + '/intersec_labels.npy', intersec_label_list)
                
            auc, eer, fpr, accuracy,  violation, esauc, min_over_max_auc = acc_fairness_dro(args.savepath + '/', [['male_pos', 'male_neg', 'female_pos', 'female_neg']])        
            cleanup_npy_files(args.savepath)
            # auc_list.append(auc)
            print('Epoch: {} AUC: {:.6f} Violation: {:.6f} ES-AUC: {:.6f} Min/Max: {:.6f}'.format(epoch, auc, violation, esauc, min_over_max_auc))
            
            # Update best metrics if current metrics are better
            if auc > best_auc:
                best_auc = auc
                best_auc_epoch = epoch
            if eer < best_eer:
                best_eer = eer
                best_eer_epoch = epoch
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_accuracy_epoch = epoch
            if esauc > best_esauc:
                best_esauc = esauc
                best_esauc_epoch = epoch 
            if violation < best_violation:
                best_violation = violation
                best_violation_epoch = epoch  
            if min_over_max_auc > best_min_over_max_auc:
                best_min_over_max_auc = min_over_max_auc
                best_min_over_max_auc_epoch = epoch  
                    
    print(f"Best AUC: {best_auc:.4f} at epoch {best_auc_epoch}")
    print(f"Best EER: {best_eer:.4f} at epoch {best_eer_epoch}")
    print(f"Best Accuracy: {best_accuracy:.4f} at epoch {best_accuracy_epoch}")
    print(f"Best ES-AUC: {best_esauc:.4f} at epoch {best_esauc_epoch}")
    print(f"Best Violation: {best_violation:.4f} at epoch {best_violation_epoch}")
    print(f"MIN/MAX: {best_min_over_max_auc:.4f} at epoch {best_min_over_max_auc_epoch}")
    
    return model, epoch


def main():

    torch.manual_seed(args.seed)
    use_gpu = torch.cuda.is_available()


    sys.stdout = Logger(osp.join('YOUR_PATH_HERE'))


    if use_gpu:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    model = model_class()
    model.to(device)

    start_epoch = 0
    # if args.continue_train and args.checkpoints != '':
    #     state_dict = torch.load(args.checkpoints)
    #     model.load_state_dict(state_dict)
    #     start_epoch = 0
        

    # optimize
    # Optimizer for model parameters (theta)
    theta_params = [param for name, param in model.named_parameters() if name not in ['lambdas', 'p_tildes']]
    base_optimizer = torch.optim.SGD
    optimizer_theta = SAM(theta_params, base_optimizer, lr=args.lr, momentum=0.9, weight_decay=5e-3)
    
    # Optimizer for lambda and p_tildes
    learning_rate_lambda = 0.0005
    learning_rate_p_list = [0.0005] * args.num_groups
    # Ensure p_tildes are initialized
    dummy_data_dict = next(iter(train_dataloader))
    dummy_data_dict = {key: value.to(device) for key, value in dummy_data_dict.items()}  # Move data to the appropriate device
    model(dummy_data_dict)  # This will initialize p_tildes if not already initialized
    
    optimizer_lambda = optim.SGD([model.lambdas], lr=learning_rate_lambda, momentum=0.9, weight_decay=5e-3)
    optimizer_p_list = [optim.SGD([p], lr=lr, momentum=0.9, weight_decay=5e-3) for p, lr in zip(model.p_tildes, learning_rate_p_list)]
    
    

    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_theta, step_size=60, gamma=0.9)

    model, epoch = train(model, optimizer_theta, optimizer_lambda, optimizer_p_list,
                         exp_lr_scheduler, device, num_epochs=200, start_epoch=start_epoch)

    if epoch == 199:
        print("training finished!")
        exit()


if __name__ == '__main__':
    main()
