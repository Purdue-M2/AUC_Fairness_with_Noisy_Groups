import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import os.path as osp
from log_utils import Logger
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import os
from sam import SAM
import torch.backends.cudnn as cudnn
import argparse
from utils.bypass_bn import enable_running_stats, disable_running_stats
# import wandb
# from fairness_metrics import acc_fairness

from dataset.data_processor import load_adult_data, load_bank_data, load_compas_data, load_default_data
from dataset.data_utils import np2torch
from networks.simple_nn_adult import SimpleNN
from simple_adult import get_auc_constraints_dro, get_equal_tpr_and_fpr_constraints, get_equal_tpr_and_fpr_constraints_dro, get_auc_constraints
from simple_adult import loss_helper
from itertools import product
 
parser = argparse.ArgumentParser("Example")

parser.add_argument('--lr', type=float, default=0.0005,
                    help="learning rate for training")
parser.add_argument('--train_batchsize', type=int, default=10000, help="batch size")
parser.add_argument('--test_batchsize', type=int, default=16, help="test batch size")
parser.add_argument('--seeds', type=list, default=[5, 6,7,8,9])
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--fake_datapath', type=str,
                    default='dataset/')
parser.add_argument('--real_datapath', type=str,
                    default='dataset/')
# parser.add_argument('--datapath', type=str,
#                     default='../dataset/deepfakes/ff++_cvpr/')
parser.add_argument('--datapath', type=str,
                    default='../dataset/deepfakes/ff++_ori/')
parser.add_argument("--continue_train", default=False, action='store_true')
parser.add_argument("--checkpoints_simple_adult_010101", type=str, default='',
                    help="continue train model path")
parser.add_argument("--model", type=str, default='fair_df_detector_inverted_cvar',
                    help="detector name[xception, fair_df_detector,daw_fdd, efficientnetb4]")

parser.add_argument("--dataset_type", type=str, default='pair',
                    help="detector name[pair,no_pair]")


# 不要改该参数，系统会自动分配
parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')
 # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
parser.add_argument('--world-size', default=5, type=int,
                        help='number of distributed processes')
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

#################################test##############################

parser.add_argument("--inter_attribute", type=str,
                    default='male,asian-male,white-male,black-male,others-nonmale,asian-nonmale,white-nonmale,black-nonmale,others-young-middle-senior-ageothers')
parser.add_argument("--single_attribute", type=str,
                    default='young-middle-senior-ageothers')
parser.add_argument("--test_datapath", type=str,
                        default='../dataset/deepfakes/ff++_cvpr/test_noise.csv', help="test data path")
parser.add_argument("--savepath", type=str,
                        default='../results_cvpr25_simple_adult')
parser.add_argument('--pho', type=float, default=0.0005,
                    help="sam factor")
parser.add_argument("--method", type=str,
                        default='auc')
parser.add_argument("--dataset", type=str,
                        default='default')
parser.add_argument("--run_name", type=str,
                        default='simple_default_auc_auc_sam_08_0005')
parser.add_argument('--ratio', type=float, default=0.5,
                    help="noise ratio")

args = parser.parse_args()

if torch.cuda.is_available() is False:
    raise EnvironmentError("not find GPU device for training.")


device = args.device

# Initialize wandb
# wandb.init(project="auc_optimization", entity="mywu981", name=args.run_name)
# wandb.init(project="auc_optimization", entity="mywu981")

###### load data ######
if args.dataset == 'adult':
    (X_train_n, train_writter), (X_test, test_writter) = load_adult_data('../dataset/adult/processed_data.csv')
elif args.dataset == 'bank':
    (X_train_n, train_writter), (X_test, test_writter) = load_bank_data('../dataset/bank/processed_data.csv')
elif args.dataset == 'compas':
    (X_train_n, train_writter), (X_test, test_writter) = load_compas_data('../dataset/compas-analysis-master/processed_data.csv')
elif args.dataset == 'default':
    (X_train_n, train_writter), (X_test, test_writter) = load_default_data('../dataset/default/processed_data.csv')
X_train, y_train, group_neg_loaders, group_pos_loaders, phats, _ = np2torch(X_train_n, train_writter, args.train_batchsize,args.device, get_loader=True)



def calculate_metrics_for_train(label, output):
    # Assuming output is already squeezed and contains raw scores
    # Apply sigmoid to convert scores to probabilities
    prob = torch.sigmoid(output)

    # Convert probabilities to binary predictions
    prediction = (prob > 0.5).long()  # Using 0.5 as the threshold for binary classification    
    correct = (prediction == label).sum().item()
    accuracy = correct / prediction.size(0)

    # Preparing data for AUC, EER, and Average Precision calculations
    label_np = label.cpu().numpy()
    prob_np = prob.cpu().detach().numpy()
    prediction_np = prediction.cpu().numpy()
    if np.isnan(label_np).any() or np.isnan(prob_np).any():
        print("NaN values found in inputs")
        print("Labels: ", label_np)
        print("Probabilities: ", prob_np)
        raise ValueError("Input contains NaN.")
    # Calculate AUC and EER
    fpr, tpr, thresholds = metrics.roc_curve(label_np, prob_np, pos_label=1)
    if np.isnan(fpr[0]) or np.isnan(tpr[0]):
        auc, eer = -1, -1  # Handling the case where AUC/EER calculation is not possible
    else:
        auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    # Calculate Average Precision
    ap = metrics.average_precision_score(label_np, prob_np)
    
    tn, fp, fn, tp = confusion_matrix(label_np, prediction_np).ravel()

    # Calculate TPR and FPR
    tpr_05 = tp / (tp + fn)  # True Positive Rate
    fpr_05 = fp / (fp + tn)  # False Positive Rate


    return auc, eer, accuracy, ap, fpr_05, tpr_05



# train and evaluation
def train(model, optimizer_theta, optimizer_lambda, optimizer_p_list, scheduler, X_train_n, train_writter, X_test, test_writter, seed, num_epochs, start_epoch):
    
    # Initialize best metric variables before the training loop
    best_auc = 0.0
    best_eer = 1.0  # Assuming lower EER is better
    best_accuracy = 0.0
    best_ap = 0.0
    best_fpr = 1.0
    best_violation = 1.0
    best_min_over_max_auc = 0.0
    best_minimax = 0.0
    
    # Optionally, keep track of the epochs when the best metrics occurred
    best_auc_epoch = None
    best_eer_epoch = None
    best_accuracy_epoch = None
    best_ap_epoch = None
    best_fpr_epoch = None
    best_violation_epoch = None
    best_min_over_max_auc_epoch = None 
    best_minimax_epoch = None
    
    auc_list = []
    fpr_list = []
    violation_list = []
    X_train, y_train, group_neg_loaders, group_pos_loaders, phats, _ = np2torch(X_train_n, train_writter, args.train_batchsize,args.device, get_loader=True)
    X_val, y_val,  _, _, phats_val, X_val_list = np2torch(X_test, test_writter, args.train_batchsize,args.device, get_loader=False)
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        phase = 'train'
        model.train()

        total_loss = 0.0

        criterion = loss_helper('pauc')
        
        
        # print(len(group_neg_loaders), '000000')
        # print(len(group_pos_loaders), '5555555')
        for idx, (group_neg_loader, group_pos_loader) in enumerate(zip(zip(*group_neg_loaders), zip(*group_pos_loaders))):
            # print('111111')
            # exit()
            # Zero out all gradients
            optimizer_theta.zero_grad()
            optimizer_lambda.zero_grad()
            for optimizer_p in optimizer_p_list:
                optimizer_p.zero_grad()
            
            X_batch_list = []
            n_batch_list = []
            X_phats_list = []

            # loop through groups
            
            for (i, j) in product(range(train_writter.n_groups), range(train_writter.n_groups)):               
                # Access the preloaded data
                X_group_neg, = group_neg_loader[i]  # Get the data tensor from the tuple
                X_group_pos, = group_pos_loader[j]  # Get the data tensor from the tuple
                

                X_batch_list.append(X_group_neg)
                X_batch_list.append(X_group_pos)
                n_batch_list.append(X_group_neg.shape[0])
                n_batch_list.append(X_group_pos.shape[0])
                
                X_single_phat = torch.cat((X_group_neg, X_group_pos), dim=0)
                X_phats_list.append(X_single_phat)
            X_batch = torch.cat(X_batch_list)
            current_batch_size = X_batch.shape[0]
            # print(current_batch_size,'5555555')
            gender_label = torch.zeros(current_batch_size, device=args.device)
            start_1=0
            for i in range(3):
                start_1 += n_batch_list[i]
            end_1 = start_1 + n_batch_list[3]+n_batch_list[4]
            gender_label[start_1:end_1] = 1
            start_2 = end_1 + n_batch_list[5]
            end_2 = start_2+n_batch_list[6] + n_batch_list[7]
            gender_label[start_1:end_2] = 1
            
            phat_list=[]
            protected_labels_list = []
            # Calculate start and end indices, and assign values to phat
            start_idx = 0
            
            for w in range(len(X_phats_list)):
                value = phats[w]
                length = X_phats_list[w].shape[0]
                end_idx = start_idx + length
                # Initialize phat tensor
                phat = torch.zeros(current_batch_size, device=args.device)  # Ensure it matches the device of 'value'
                protected_label = torch.zeros(current_batch_size, device=args.device)
                
                # Use slicing to assign the value to the specified range
                phat[start_idx:end_idx] = value
                protected_label[start_idx:end_idx] = 1
                
                # Update the start index for the next batch
                start_idx = end_idx
                
                # Append the current phat to the list
                phat_list.append(phat)
                protected_labels_list.append(protected_label)
            phats_final = torch.stack(phat_list, dim=0).to(args.device)
            protected_labels = torch.stack(protected_labels_list, dim=0).to(args.device)
            
            
            yy_pred = model(X_batch).view(-1)
            yy_group = torch.split(yy_pred, n_batch_list)
    
            losses = []
            for idx, (i, j) in enumerate(product(range(train_writter.n_groups), range(train_writter.n_groups))):
                loss_group = criterion(yy_group[2 * idx], yy_group[2 * idx + 1])
                loss_group = loss_group.squeeze(-1)
                losses.append(loss_group)

            # prepare loss for model parameter
            loss = torch.stack(losses).mean()
            # print(loss)
    
            yy_batch_list = []  # To store ground truth labels for all groups
            # now get the groud truth by looping groups
            for idx in range(len(yy_group)):
                yy_batch = torch.ones_like(yy_group[idx])
                if (idx+2) % 2 == 0:
                    yy_batch[:] = 0.
                yy_batch_list.append(yy_batch)
            
            yy_batch = torch.cat(yy_batch_list)
            # print(yy_batch.shape)
            
            # Separate yy_pred based on the ground truth labels in yy_batch
            yy_pos = yy_pred[yy_batch == 1]  # Predictions for positive labels
            yy_neg = yy_pred[yy_batch == 0]  # Predictions for negative labels
            overall_loss = criterion(yy_neg, yy_pos)
            # print(loss, '????????????????')
    
            #compute constraint
            if args.method == 'auc':
                constraints = get_auc_constraints_dro(model, train_writter, yy_group, gender_label, overall_loss, yy_batch)# 4,
                
            elif args.method == 'fpr':
                constraints = get_equal_tpr_and_fpr_constraints_dro(model, yy_batch, yy_pred)
                
            elif args.method == 'naive':
                constraints = get_auc_constraints(model, train_writter, yy_group, gender_label, overall_loss, yy_batch)
            

            lagrangian_loss = torch.dot(model.lambdas, constraints)
            # print(lagrangian_loss, '!!!!!!!!!!!!!!!!!!!!!')

            # Total loss
            total_loss_batch = loss + lagrangian_loss
            
            
            
            # Main backward pass
            total_loss_batch.backward(retain_graph=True)
            
            # Update main model parameters
            optimizer_theta.step()
            
            
            # Update lambda by accent
            lambda_grads = torch.autograd.grad(
                torch.dot(model.lambdas, constraints), 
                model.lambdas, 
                retain_graph=True, 
                allow_unused=True
            )[0]
            
            
            if lambda_grads is not None:
                model.lambdas.grad = -lambda_grads
                # print('llllllllggggggg',lambda_grads)
                # print('Before: lambda', model.lambdas.data)
                optimizer_lambda.step()
                # print('After step: lambda', model.lambdas.data)
                model.lambdas.data = model.project_lambdas(model.lambdas.data)
                # print('After: lambda', model.lambdas.data)
            
            # Reverse gradients for p and lambda to perform gradient ascent
            
            if args.method == 'auc' or 'fpr':
                for p in model.p_tildes.parameters():
                    if p.grad is not None:
                        p.grad = -p.grad
                        # print('ppppppppgggggggg',p.grad)
            

            
            
            
                for i, optimizer_p in enumerate(optimizer_p_list):
                    # print('Before p: ', model.p_tildes[i].data)
                    optimizer_p.step()
                    # print('after step p: ', model.p_tildes[i].data)
                    model.p_tildes[i].data[:current_batch_size] = model.project_ptilde(phats_final, model.p_tildes[i].data[:current_batch_size], i)
                    # print('after p: ', model.p_tildes[i].data)
            
            
                
            
            total_loss += total_loss_batch.item() 
        # print('22222')
        # exit()
        epoch_loss = total_loss / ((X_train.shape[0])*2)
        print('Epoch: {} Loss: {:.9f}'.format(epoch, epoch_loss))

        # update learning rate
        if phase == 'train':
            scheduler.step()

        # evaluation

        if (epoch+1) % 1 == 0:
            
            if args.method == 'auc':
                savepath = 'YOUR_PATH_HERE'
            elif args.method == 'naive':
                savepath = 'YOUR_PATH_HERE'


            temp_model = savepath+"/"+args.model+str(epoch)+'.pth'
            os.makedirs(os.path.dirname(temp_model), exist_ok=True)
            torch.save(model.state_dict(), temp_model)

            # print()
            print('-' * 10)

            phase = 'test'
            model.eval()
            #Initialize fairness metric variables before the training loop
            
            yy_pred_val = model(X_val).view(-1)
            group_num = X_val_list[0].shape[0] + X_val_list[1].shape[0]
            auc, eer, accuracy, ap, fpr, tpr = calculate_metrics_for_train(y_val, yy_pred_val)
            auc_groups = []
            for (i, j) in product(range(test_writter.n_groups), range(test_writter.n_groups)):
                # Extract the corresponding indices for the two groups
                indices_group_i = test_writter.grouplabel_ind[2 * i]
                indices_group_j = test_writter.grouplabel_ind[2 * j + 1]
                
                # Extract true labels and predictions for the two groups
                y_group_i = y_val[indices_group_i]
                y_group_j = y_val[indices_group_j]
                yy_pred_group_i = yy_pred_val[indices_group_i]
                yy_pred_group_j = yy_pred_val[indices_group_j]
                
                # Combine the two groups
                y_combined = torch.cat([y_group_i, y_group_j])
                yy_pred_combined = torch.cat([yy_pred_group_i, yy_pred_group_j])
                
                # Calculate metrics for the combined group
                auc_group, _, _, _, _, _ = calculate_metrics_for_train(y_combined, yy_pred_combined)
                auc_groups.append(auc_group)
            print(auc_groups, '000000')
            # auc_female, _, _, _, fpr_female, tpr_female = calculate_metrics_for_train(y_val[:group_num], yy_pred_val[:group_num])
            # auc_male, _, _, _, fpr_male, tpr_male = calculate_metrics_for_train(y_val[group_num:], yy_pred_val[group_num:])
            
            # if args.method == 'auc':
            violation = max(abs(auc - auc_groups[0]), abs(auc - auc_groups[1]), abs(auc - auc_groups[2]), abs(auc - auc_groups[3]))
            numer = min(auc_groups)
            dem = max(auc_groups)
            minimax = numer/dem
            
            # compute fairness metric 
            n_steps_auc = len(auc_groups)
            auc_errors = torch.tensor(auc_groups, dtype=torch.float32)
            
            agg_weights_auc = torch.arange(1, 1 + n_steps_auc, dtype=torch.float32) ** 2 
            cumulative_weights_auc = torch.cumsum(agg_weights_auc, dim=0)
            
            weighted_errors_auc = auc_errors * agg_weights_auc  # element-wise multiplication
            cumulative_weighted_errors_auc = torch.cumsum(weighted_errors_auc, dim=0)
            agg_auc_errors = cumulative_weighted_errors_auc / cumulative_weights_auc
            min_over_max_auc = torch.min(agg_auc_errors) / torch.max(agg_auc_errors)
            # else:
            # violation = torch.max(torch.tensor([tpr - tpr_female, tpr - tpr_male, fpr - fpr_female, fpr - fpr_male]))
            # violation = torch.max(torch.tensor([tpr - tpr_female, tpr - tpr_male]))
            
            violation_list.append(violation)
            # Log metrics to wandb
            # wandb.log({
            #     'Loss/epoch_loss': epoch_loss
            # })

            
            print(f"Epoch {epoch}: AUC={auc}, EER={eer}, Accuracy={accuracy}, AP={ap}, FPR={fpr}, Violation={violation}, Min/Max={minimax}, Min/Max_normal={min_over_max_auc}")
            # Update best metrics if current metrics are better
            if epoch > 600:
                if auc > best_auc:
                    best_auc = auc
                    best_auc_epoch = epoch
                if eer < best_eer:
                    best_eer = eer
                    best_eer_epoch = epoch
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_accuracy_epoch = epoch
                if ap > best_ap:
                    best_ap = ap
                    best_ap_epoch = epoch
                if fpr < best_fpr:
                    best_fpr = fpr
                    best_fpr_epoch = epoch    
                if violation < best_violation:
                    best_violation = violation
                    best_violation_epoch = epoch  
                if min_over_max_auc > best_min_over_max_auc:
                    best_min_over_max_auc = min_over_max_auc
                    best_min_over_max_auc_epoch = epoch
                if minimax > best_minimax:
                    best_minimax = minimax
                    best_minimax_epoch = epoch

  
    # print("\nTraining complete.")
    print(f"Best AUC: {best_auc:.6f} at epoch {best_auc_epoch}")
    print(f"Best EER: {best_eer:.6f} at epoch {best_eer_epoch}")
    print(f"Best Accuracy: {best_accuracy:.6f} at epoch {best_accuracy_epoch}")
    print(f"Best AP: {best_ap:.6f} at epoch {best_ap_epoch}")
    print(f"Best FPR: {best_fpr:.6f} at epoch {best_fpr_epoch}")
    print(f"Best Violation: {best_violation:.6f} at epoch {best_violation_epoch}")
    print(f"MIN/MAX_normal: {best_min_over_max_auc:.6f} at epoch {best_min_over_max_auc_epoch}")
    print(f"MIN/MAX: {best_minimax:.6f} at epoch {best_minimax_epoch}")

    return model, epoch, best_auc, best_eer, best_accuracy, best_ap, best_min_over_max_auc, best_violation, best_minimax


def main():
    # Initialize lists to store results for each run
    best_auc_list = []
    best_eer_list = []
    best_accuracy_list = []
    best_ap_list = []
    best_min_over_max_auc_list = []
    min_over_max_fpr_list = []
    best_violation_list = []
    best_minimax_list = []

    
    for seed in args.seeds:
        torch.manual_seed(seed)
        use_gpu = torch.cuda.is_available()

        if args.method == 'auc':
            # pass
            sys.stdout = Logger(osp.join('YOUR_PATH_HERE'))
        elif args.method == 'naive':
            sys.stdout = Logger(osp.join('YOUR_PATH_HERE'))
        else:
            raise EnvironmentError("method is not found.")

        if use_gpu:
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(seed)
        else:
            print("Currently using CPU")


        # Define the model
        input_size = X_train.shape[1]  # Number of features in the dataset
        # Define hyperparameters for lambda and p_tildes
        num_groups = train_writter.n_groups ** 2
        learning_rate_lambda = 0.001
        learning_rate_p_list = [0.001] * num_groups
    
        model = SimpleNN(input_size, num_groups, args.train_batchsize, args.method)
        # Optimizer for model parameters (theta)
        theta_params = [param for name, param in model.named_parameters() if name not in ['lambdas', 'p_tildes']]
        optimizer_theta = optim.SGD(theta_params, lr=0.01, momentum=0.9, weight_decay=5e-3)

        # Learning rate scheduler
        scheduler = lr_scheduler.StepLR(optimizer_theta, step_size=0.1, gamma=0.9)
        

        # Optimizer for lambda and p_tildes
        optimizer_lambda = optim.SGD([model.lambdas], lr=learning_rate_lambda, momentum=0.9, weight_decay=0.01)
        optimizer_p_list = [optim.SGD([p], lr=lr, momentum=0.9, weight_decay=0.01) for p, lr in zip(model.p_tildes, learning_rate_p_list)]


        # Training loop
        model.to(args.device)

        start_epoch = 0
        if args.continue_train and args.checkpoints_simple_adult_010101 != '':
            state_dict = torch.load(args.checkpoints_simple_adult_010101)
            model.load_state_dict(state_dict, strict=False)
            start_epoch = 59
            print(start_epoch)
            

        # Train the model
        model, epoch, best_auc, best_eer, best_accuracy, best_ap, best_min_over_max_auc, best_violation, best_minimax = train(model, optimizer_theta, optimizer_lambda, optimizer_p_list, scheduler, X_train_n, train_writter, X_test, test_writter, seed, num_epochs=1000, start_epoch=start_epoch)

        # #log results to wandb
        # wandb.log({
        #         'train_metric/auc': best_auc,
        #         'train_metric/eer': best_eer,
        #         'train_metric/accuracy': best_accuracy,
        #         'train_metric/ap': best_ap,
        #         'train_metric/min_over_max_auc': best_min_over_max_auc,
        #         'seed': seed
        #     }, step=seed)
        
        # Append results for each metric
        best_auc_list.append(best_auc)
        best_eer_list.append(best_eer)
        best_accuracy_list.append(best_accuracy)
        best_ap_list.append(best_ap)
        best_min_over_max_auc_list.append(best_min_over_max_auc)
        best_minimax_list.append(best_minimax)
        best_violation_list.append(best_violation)
        
        print('---------------------------------------')
        print(f"Experiment Seed {seed}: AUC={best_auc}, EER={best_eer}, Accuracy={best_accuracy}, AP={best_ap}, min_over_max_auc={best_min_over_max_auc}, minimax={best_minimax}")
        print('---------------------------------------')

    # Calculate mean and standard deviation for each metric after all runs
    mean_best_auc = np.mean(best_auc_list)
    std_best_auc = np.std(best_auc_list)

    mean_best_eer = np.mean(best_eer_list)
    std_best_eer = np.std(best_eer_list)

    mean_best_accuracy = np.mean(best_accuracy_list)
    std_best_accuracy = np.std(best_accuracy_list)

    mean_best_ap = np.mean(best_ap_list)
    std_best_ap = np.std(best_ap_list)

    mean_min_over_max_auc = np.mean(best_min_over_max_auc_list)
    std_min_over_max_auc = np.std(best_min_over_max_auc_list)
    
    minimax = np.mean(best_minimax_list)
    std_minimax = np.std(best_minimax_list)
    
    mean_best_violation = np.mean(best_violation_list)
    std_best_violation = np.std(best_violation_list)

    # Print or log the results
    print('============================================')
    print('============================================')
    print(f"Best AUC - Mean: {mean_best_auc}, Std: {std_best_auc}")
    print(f"Best EER - Mean: {mean_best_eer}, Std: {std_best_eer}")
    print(f"Best Accuracy - Mean: {mean_best_accuracy}, Std: {std_best_accuracy}")
    print(f"Best AP - Mean: {mean_best_ap}, Std: {std_best_ap}")
    print(f"Min over Max AUC normal - Mean: {mean_min_over_max_auc}, Std: {std_min_over_max_auc}")
    print(f"Min over Max AUC - Mean: {minimax}, Std: {std_minimax}")
    print(f"violation - Mean: {mean_best_violation}, Std: {std_best_violation}")
        
    # # Finish wandb run
    # wandb.finish()    
    
    if epoch == 999:
        print("training finished!")
        exit()


if __name__ == '__main__':
    main()




