from email.policy import strict
import sys
from detectors import DETECTOR
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import os.path as osp
from log_utils import Logger
import torch.backends.cudnn as cudnn
from dataset.pair_dataset import pairDataset, pairDatasetDro
from dataset.datasets_train import ImageDataset_Test_Dro
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

parser.add_argument("--checkpoints", type=str, default='',
                    help="continue train model path")
parser.add_argument("--model", type=str, default='auc_effinet',
                    help="detector name[auc_effinet, auc_res,...]")
parser.add_argument('--device', default='cuda:5', help='device id (i.e. 0 or 0,1 or cpu)')

#################################test##############################

parser.add_argument("--inter_attribute", type=str,
                    default='male-female')
parser.add_argument("--label_attribute", type=str,
                    default='pos-neg')
parser.add_argument("--test_datapath", type=str,
                        default='../dataset/deepfakes/ff++_ori/test.csv', help="test data path")
parser.add_argument("--savepath", type=str,
                        default='../results')
parser.add_argument("--backbone", type=str,
                        default='efficientnetb4', help=['resnet50', 'xception', 'efficientnetb4'])
parser.add_argument('--test_batchsize', type=int, default=32, help="test batch size")
args = parser.parse_args()

###### import data transform #######
from transform import fair_df_default_data_transforms as data_transforms
test_transforms = get_albumentations_transforms([''])
def cleanup_npy_files(directory):
    """
    Deletes all .npy files in the given directory.
    :param directory: The directory to clean up .npy files in.
    """
    for item in os.listdir(directory):
        if item.endswith(".npy"):
            os.remove(os.path.join(directory, item))
    print("Cleaned up .npy files in directory:", directory)
device = args.device
# prepare the model (detector)
model_class = DETECTOR[args.model]
model = model_class()
model.to(device)

# Load the checkpoint
checkpoint_path = args.checkpoints
checkpoint = torch.load(checkpoint_path, map_location=device)
new_state_dict = {}
for k, v in checkpoint.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v  # remove 'module.' prefix
    else:
        new_state_dict[k] = v

# Load the state dict into the model
model.load_state_dict(new_state_dict, strict=False)
model.eval()  # Set the model to evaluation mode
interattributes = args.inter_attribute.split('-')

preds_list=[]
labels_list = []
intersec_labels_list = []
for eachatt in interattributes:
    test_dataset = ImageDataset_Test_Dro(args.test_datapath,eachatt, test_transforms)
    test_dataset_size = len(test_dataset)

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batchsize, shuffle=False,num_workers=32, pin_memory=False)
    total_auc = 0
    total_violation = 0
    auc_list = []
    print('Testing: ')
    print('-' * 10)
    # print('%d batches int total' % len(test_dataloader))
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
            # print('%d batches int total' % len(test_dataloader))
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
    print('AUC: {:.6f} Violation: {:.6f} ES-AUC: {:.6f} Min/Max: {:.6f}'.format(auc, violation, esauc, min_over_max_auc))

