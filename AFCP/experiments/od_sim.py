import os, sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import torch.optim as optim
import shutil

from scipy.stats.mstats import mquantiles
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split

sys.path.append("../")

from third_party.utils import *
from SelectiveCI_fairness.methods import Marginal_Selection, Exhaustive_Selection, Partial_Selection, Adaptive_Selection
from SelectiveCI_fairness.black_box import Blackbox
from SelectiveCI_fairness.networks import BinaryClassification
from SelectiveCI_fairness.evals import eval_pvalues, eval_by_att_od
from experiments.data_gen import data_model





#########################
# Experiment parameters #
#########################
# Parse input arguments
print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))
# if len(sys.argv) != 5:
#     print("Error: incorrect number of parameters.")
#     quit()

n_train_calib = int(sys.argv[1])
lr = float(sys.argv[2])
n_epoch = int(sys.argv[3])
seed = int(sys.argv[4])
ttest_delta = float(sys.argv[5])
delta0 = float(sys.argv[6])
delta1 = float(sys.argv[7])
perc1 = float(sys.argv[8])

# else: # Default parameters
#     n_train_calib_list = 1000
#     lr = 0.0001
#     n_epoch = 500
#     seed = 2023
#     delta0 = 0.1 
#     delta1 = 0.9
#     ttest_delta = 0

# Fixed experiment parameters
num_workers = 0
batch_size = 25
beta = 0 # frequency-threshold, set to 0 if using simulated data or having merged minor groups in data-preoprocessing steps
alpha = 0.1 # nominal miscoverage level
n_exp = 30
n_test = 500

p = 10 # dimension of the feature for simulated data
K = 2 # number of classes
idx_list = [p-1, p-2, p-3] # list of attributes to investigate over

group_perc = [1-perc1, perc1] # easy-to-classify, hard-to-classify

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    # Make CuDNN Determinist
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)


###############
# Output file #
###############
outdir = "results/outlier_detection/"
os.makedirs(outdir, exist_ok=True)
outfile_name = "ndata"+str(n_train_calib) + "_lr" + str(lr) + "_delta1" + str(delta1) +\
                "_delta"+str(ttest_delta) + "_seed" + str(seed)
outfile = outdir + outfile_name + ".txt"
print("Output file: {:s}".format(outfile), end="\n")

modeldir = "models/outlier_detection/"+outfile_name



#########################
# Auxiliary functions #
#########################

colors = [[31, 120, 180], [51, 160, 44], [250,159,181]]
colors = [(r / 255, g / 255, b / 255) for (r, g, b) in colors]

def plot_loss(train_loss, val_loss):
    x = np.arange(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(x, train_loss, color=colors[0], label="Training loss", linewidth=2)
    plt.plot(x, val_loss, color=colors[1], label="Validation loss", linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title("Evolution of the training, validation and test loss")

    plt.show()
    
def criterion(outputs, inputs, targets):
    # return Loss(outputs, inputs)
    return Loss(outputs, targets.unsqueeze(1))
    
    
    
    
    
#################
# Download/Simulate Data #
#################
data_model = data_model(K = K, p = p, delta0 = delta0, delta1 = delta1, group_perc = group_perc, seed = seed)
    
    
n_train = int(n_train_calib*0.75)
X_train = data_model.sample_X(n_train)
y_train = data_model.sample_Y(X_train)

n_calib = 0
while n_calib < int(n_train_calib*0.25):
    X_new = data_model.sample_X(int(n_train_calib*0.25))
    Y_new = data_model.sample_Y(X_new)
    inliers_idx = np.where(Y_new == 0)
    X_new = X_new[inliers_idx]
    Y_new = Y_new[inliers_idx]
    if n_calib == 0:
        X_calib = X_new
        y_calib = Y_new
    else:
        X_calib = np.append(X_calib, X_new, axis = 0)
        y_calib = np.append(y_calib, Y_new, axis = 0)
    n_calib = X_calib.shape[0]

if n_calib > int(n_train_calib*0.25):
    random_indices = np.random.choice(X_calib.shape[0],
                                      size=int(n_train_calib*0.25),
                                      replace=False)
    X_calib = X_calib[random_indices]
    y_calib = y_calib[random_indices]



X_test = data_model.sample_X(n_test)
y_test = data_model.sample_Y(X_test)

print('total number of available training data is: {:d}, in which {:d} are inliers, {:d} are outlier'.\
      format(X_train.shape[0],sum(y_train==0), sum(y_train==1)))
print('total number of available calibration data is: {:d}, in which {:d} are inliers, {:d} are outliers'.\
      format(X_calib.shape[0],sum(y_calib==0), sum(y_calib==1)))
print('total number of test data is {:d}, in which {:d} are inliers, {:d} are outliers.'\
      .format(X_test.shape[0], sum(y_test==0), sum(y_test==1)))

train_dataset = data_utils.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
calib_dataset = data_utils.TensorDataset(torch.Tensor(X_calib), torch.Tensor(y_calib))
test_dataset = data_utils.TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))  

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=n_test, num_workers=num_workers, shuffle= False)


################
# Train models #
################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Is CUDA available? {}".format(torch.cuda.is_available()))

# Define the model parameters
input_shape = X_train.shape[1]
net = BinaryClassification(input_shape = input_shape, device = device) # AutoEncoder(input_shape= input_shape)


Loss = nn.BCEWithLogitsLoss() # nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

bbox_od = Blackbox(net, device, train_loader, batch_size=batch_size, max_epoch=n_epoch,
                        learning_rate=lr, val_loader=calib_loader, criterion=criterion, optimizer=optimizer, verbose = False)
# Train the model and save snapshots regularly
bbox_od_stats = bbox_od.full_train(save_dir = modeldir)

dataiter = iter(test_loader)
inputs, labels = next(dataiter)
# C_PVals_od = Conformal_PVals(net = net, device = device, cal_loader = calib_loader, model_path = './od/od_old_adult'+'_'+str(seed), verbose = False)
# pvals_od = C_PVals_od.compute_pvals(inputs)

# result_by_att = eval_by_att(X_test, labels, idx_list, pvals_od, [alpha])
# worst_group = result_by_att.iloc[np.argmax(result_by_att['Fixed-FPR']),-2]

################
#   Inference  #
################

def complete_df(df, pvals, method = None, is_marg_results = False, is_selective = False, k_hat = None):
    # assert method is not None, 'need to input the method name!'
    df["n_data"] = n_train_calib
    df["seed"] = seed
    df["lr"] = lr
    df["batch_size"] = batch_size
    df['alpha'] = alpha
    df['delta1'] = delta1
    df['method'] = method
    df['perc_1'] = perc1
    if beta>0:
        df['beta_threshold'] = beta*len(X_calib)
    if is_marg_results:
        df['n_count'] = len(pvals)
        df['attribute_idx'] = -1
        df['attribute_level'] = -1
    if is_selective == False:
        df['selected_7'] = None
        df['selected_8'] = None
        df['selected_9'] = None
    elif is_selective: 
      assert k_hat is not None, "input selected attribute list"
      df['selected_7'] = np.mean([1 if x == [7] else 0 for x in k_hat])
      df['selected_8'] = np.mean([1 if x == [8] else 0 for x in k_hat])
      df['selected_9'] = np.mean([1 if x == [9] else 0 for x in k_hat])

    return df


results_full = pd.DataFrame()


## Adaptive
Selective_method = Adaptive_Selection(alpha = alpha, random_state = seed, ttest_delta = ttest_delta) 
pvals_adaptive, k_hat = Selective_method.outlier_detection(X_calib,
                                                            X_test,
                                                            bbox_od, 
                                                            idx_list)

results_marg = eval_pvalues(pvals_adaptive,  labels, [alpha])
results_marg = complete_df(results_marg, pvals = pvals_adaptive, method = 'Adaptive', is_marg_results = True, is_selective = True, k_hat = k_hat)
results_cond = eval_by_att_od(X_test, labels, idx_list, pvals_adaptive, [alpha])
results_cond = complete_df(results_cond, pvals = pvals_adaptive, method = 'Adaptive', is_selective = True, k_hat = k_hat)
results_full = pd.concat([results_full, results_marg, results_cond])


## Adaptive1
Selective_method = Adaptive_Selection(alpha = alpha, random_state = seed)
pvals_adaptive_1, k_hat_1 = Selective_method.outlier_detection(X_calib,
                                                                  X_test,
                                                                  bbox_od,
                                                                  idx_list)

results_marg = eval_pvalues(pvals_adaptive_1,  labels, [alpha])
results_marg = complete_df(results_marg, pvals = pvals_adaptive_1, method = 'Adaptive1', is_marg_results = True, is_selective = True, k_hat = k_hat_1)
results_cond = eval_by_att_od(X_test,  labels, idx_list, pvals_adaptive_1, [alpha])
results_cond = complete_df(results_cond, pvals = pvals_adaptive_1, method = 'Adaptive1', is_selective = True, k_hat = k_hat_1)
results_full = pd.concat([results_full, results_marg, results_cond])

## Marginal
Selective_method = Marginal_Selection(alpha = alpha, random_state = seed)
pvals_marginal = Selective_method.outlier_detection(X_test,
                                                    X_calib,
                                                    bbox_od = bbox_od, 
                                                    left_tail = False)
# marginal results, on all combinations
results_marg = eval_pvalues(pvals_marginal, labels, [alpha])
results_marg = complete_df(results_marg, pvals = pvals_marginal, method = 'Marginal', is_marg_results = True)
results_cond = eval_by_att_od(X_test, labels, idx_list, pvals_marginal, [alpha])
results_cond = complete_df(results_cond, pvals = pvals_marginal, method = 'Marginal')
results_full = pd.concat([results_full, results_marg, results_cond])

## Partial
Selective_method = Partial_Selection(alpha = alpha, random_state = seed)
pvals_partial = Selective_method.outlier_detection(X_test,
                                                    X_calib,
                                                    bbox_od = bbox_od,
                                                    left_tail = False,
                                                    sensitive_atts_idx = idx_list, 
                                                   )      
results_marg = eval_pvalues(pvals_partial, labels, [alpha])
results_marg = complete_df(results_marg, pvals = pvals_partial, method = 'Partial', is_marg_results = True)
results_cond = eval_by_att_od(X_test, labels, idx_list, pvals_partial, [alpha])
results_cond = complete_df(results_cond, pvals = pvals_partial, method = 'Partial')
results_full = pd.concat([results_full, results_marg, results_cond])

## Exhaustive 
Selective_method = Exhaustive_Selection(alpha = alpha, random_state = seed)
pvals_exhaustive = Selective_method.outlier_detection(X_test,
                                                        X_calib,
                                                        bbox_od = bbox_od,
                                                        left_tail = False,
                                                        sensitive_atts_idx = idx_list
                                                        )
results_marg = eval_pvalues(pvals_exhaustive,labels, [alpha])
results_marg = complete_df(results_marg, pvals = pvals_exhaustive, method = 'Exhaustive', is_marg_results = True)
results_cond = eval_by_att_od(X_test, labels, idx_list, pvals_exhaustive, [alpha])
results_cond = complete_df(results_cond, pvals = pvals_exhaustive, method = 'Exhaustive')
results_full = pd.concat([results_full, results_marg, results_cond])



################
# Save Results #
################
results_full.to_csv(outfile, index=False)
print("\nResults written to {:s}\n".format(outfile))
sys.stdout.flush()

# Clean up temp model directory to free up disk space
shutil.rmtree(modeldir, ignore_errors=True)