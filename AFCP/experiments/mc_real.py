import os, sys
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import shutil

import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.append("../")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from third_party.utils import *
from SelectiveCI_fairness.methods import Marginal_Selection, Exhaustive_Selection, Partial_Selection, Adaptive_Selection
from SelectiveCI_fairness.black_box import Blackbox
from SelectiveCI_fairness.networks import ClassNNet
from SelectiveCI_fairness.evals import eval_psets, eval_by_att_mc 

from ucimlrepo import fetch_ucirepo 


#########################
# Experiment parameters #
#########################
# if True: # Input parameters
# Parse input arguments
print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))
if len(sys.argv) != 7:
    print("Error: incorrect number of parameters.")
    quit()

n_train_calib = int(sys.argv[1])
lr = float(sys.argv[2])
n_epoch = int(sys.argv[3])
seed = int(sys.argv[4])
ttest_delta = float(sys.argv[5])
conditional_ = eval(sys.argv[6])

# else: # Default parameters
#     n_train_calib = 1000
#     lr = 0.0001
#     n_epoch = 500
#     seed = 2023
#     ttest_delta = 0


# Fixed experiment parameters
num_workers = 0
batch_size = 25
beta = 0 # frequency-threshold, set to 0 if using simulated data or having merged minor groups in data-preoprocessing steps
alpha = 0.1 # nominal miscoverage level
n_test = 500

###############
# Output file #
###############
outdir = "results/multiclass/real/"
os.makedirs(outdir, exist_ok=True)
outfile_name = "ndata"+str(n_train_calib) + "_lr" + str(lr) + "_ttestdelta" + str(ttest_delta) + "_conditional" + str(conditional_) + "_seed" + str(seed) 

outfile = outdir + outfile_name + ".txt"
print("Output file: {:s}".format(outfile), end="\n")

modeldir = "models/multiClass/real/"+outfile_name

idx_list = [0, 3, 5, 6, 7] # list of attributes to investigate over

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    # Make CuDNN Determinist
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)




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
    targets = targets.to(torch.long)
    return Loss(outputs, targets)

    
    
    
    
#################
# Download/Simulate Data #
#################

# fetch dataset 
nursery = fetch_ucirepo(id=76) 
  
# data (as pandas dataframes) 
X = nursery.data.features 
y = nursery.data.targets 
colnames_ = X.columns.values

# remove the class label with only 2 samples
mask = (y != "recommend").all(axis=1)
y = y[mask]
X = X[mask]

# Encoding X and Y
encoder = LabelEncoder()
y = encoder.fit_transform(y)
encoding_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print("Encoding mapping:", encoding_mapping)
X = np.array(X)
for col in range(X.shape[1]):
  encoder = LabelEncoder()
  X[:,col] = encoder.fit_transform(X[:,col])
  encoding_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
  print('column:',colnames_[col])
  print("Encoding mapping:", encoding_mapping)
X = X.astype('float64')


# Adding prediction complexity
Z = np.hstack((X, y.reshape(-1,1)))
np.random.shuffle(Z)
rows_0 = Z[Z[:, 0] == 0]
rows_2 = Z[Z[:, 0] == 2]
rows_1 = Z[Z[:, 0] == 1]

np.random.shuffle(rows_1)  
rows_1_sampled = rows_1[:500] 
noise = np.random.uniform(-4, 4, rows_1_sampled.shape[0])
rows_1_sampled[:, -1] = np.clip(np.round(rows_1_sampled[:, -1] + noise), 0, 3)# .astype(int)

Z = np.vstack((rows_0, rows_2, rows_1_sampled))
X = Z[:,:-1]
y = Z[:,-1]
print(np.unique(X[:,0], return_counts =True))


random_indices = random.sample(range(X.shape[0]), n_train_calib+n_test)
X_full = X[random_indices,:]
y_full = y[random_indices]

X_train_calib, X_test, y_train_calib, y_test = train_test_split(X_full, y_full, train_size = n_train_calib, random_state = seed)

X_train, X_calib, y_train, y_calib = train_test_split(X_train_calib, y_train_calib, train_size = 0.5, random_state = seed)

# convert to dataset and dataloader
train_dataset = data_utils.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
calib_dataset = data_utils.TensorDataset(torch.Tensor(X_calib), torch.Tensor(y_calib))
test_dataset = data_utils.TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

batch_size = 25
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=batch_size, num_workers=num_workers, shuffle = False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=n_test, num_workers=num_workers, shuffle= False)

################
# Train models #
################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Is CUDA available? {}".format(torch.cuda.is_available()))


# Define the model parameters
net = ClassNNet(num_features = X_train.shape[1], num_classes = len(set(y_train)), device = device, use_dropout=False)

Loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
# Training the model
bbox_mc = Blackbox(net, device, train_loader, batch_size=batch_size, max_epoch=n_epoch,
                   learning_rate=lr, val_loader=calib_loader, criterion=criterion, optimizer=optimizer, verbose = False)

bbox_mc_stats = bbox_mc.full_train(save_dir = modeldir)

################
#   Inference  #
################

def complete_df_real(df, method = None, is_marg_results = False, is_selective = False, k_hat = None):
    # assert method is not None, 'need to input the method name!'
    df["n_data"] = n_train_calib
    df["seed"] = seed
    df["lr"] = lr
    df["batch_size"] = batch_size
    df['alpha'] = alpha
    df['method'] = method
    df['conditional'] = conditional_
    if beta>0:
        df['beta_threshold'] = beta*len(X_calib)
    if is_marg_results:
        df['n_count'] = len(y_test)
        df['attribute_idx'] = -1
        df['attribute_level'] = -1
    if is_selective == False:
        df['selected_0'] = None
        df['selected_3'] = None
        df['selected_5'] = None
        df['selected_6'] = None
        df['selected_7'] = None
    elif is_selective:
      assert k_hat is not None, "input selected attribute list"
      df['selected_0'] = np.mean([1 if x == [0] else 0 for x in k_hat])
      df['selected_3'] = np.mean([1 if x == [0] else 0 for x in k_hat])
      df['selected_5'] = np.mean([1 if x == [5] else 0 for x in k_hat])
      df['selected_6'] = np.mean([1 if x == [0] else 0 for x in k_hat])
      df['selected_7'] = np.mean([1 if x == [7] else 0 for x in k_hat])

    return df




results_full = pd.DataFrame()


## Adaptive
Selective_method = Adaptive_Selection(alpha = alpha, random_state = seed, ttest_delta = ttest_delta) 
C_sets_adaptive, k_hat = Selective_method.multiclass_classification(X_calib,
                                                                    y_calib,
                                                                    X_test,
                                                                    bbox_mc, 
                                                                    idx_list, 
                                                                    conditional = conditional_)

results_marg = eval_psets(C_sets_adaptive, y_test)
results_marg = complete_df_real(results_marg, method = 'Adaptive', is_marg_results = True, is_selective = True, k_hat = k_hat)
results_cond = eval_by_att_mc(X_test, y_test, idx_list, C_sets_adaptive)
results_cond = complete_df_real(results_cond, method = 'Adaptive', is_selective = True, k_hat = k_hat)
results_full = pd.concat([results_full, results_marg, results_cond])

## Adaptive1
Selective_method = Adaptive_Selection(alpha = alpha, random_state = seed)
C_sets_adaptive_1, k_hat_1 = Selective_method.multiclass_classification(X_calib,
                                                                  y_calib,
                                                                  X_test,
                                                                  bbox_mc,
                                                                  idx_list,
                                                                  conditional = conditional_)

results_marg = eval_psets(C_sets_adaptive_1, y_test)
results_marg = complete_df_real(results_marg, method = 'Adaptive1', is_marg_results = True, is_selective = True, k_hat = k_hat_1)
results_cond = eval_by_att_mc(X_test, y_test, idx_list, C_sets_adaptive_1)
results_cond = complete_df_real(results_cond, method = 'Adaptive1', is_selective = True, k_hat = k_hat_1)
results_full = pd.concat([results_full, results_marg, results_cond])


## Marginal
Selective_method = Marginal_Selection(alpha = alpha, random_state = seed)
C_sets_marginal = Selective_method.multiclass_classification(X_test,
                                                                      X_calib,
                                                                      y_calib,
                                                                      left_tail = False,
                                                                      bbox_mc = bbox_mc,
                                                                      conditional = conditional_)
# marginal results, on all combinations
results_marg = eval_psets(C_sets_marginal, y_test)
results_marg = complete_df_real(results_marg, method = 'Marginal', is_marg_results = True)
results_cond = eval_by_att_mc(X_test, y_test, idx_list, C_sets_marginal)
results_cond = complete_df_real(results_cond, method = 'Marginal')
results_full = pd.concat([results_full, results_marg, results_cond])

## Partial
Selective_method = Partial_Selection(alpha = alpha, random_state = seed)
C_sets_partial = Selective_method.multiclass_classification(X_test,
                                                                    X_calib,
                                                                    y_calib,
                                                                    left_tail = False,
                                                                    sensitive_atts_idx = idx_list, 
                                                                    bbox_mc = bbox_mc, 
                                                                    conditional = conditional_)      
results_marg = eval_psets(C_sets_partial, y_test)
results_marg = complete_df_real(results_marg, method = 'Partial', is_marg_results = True)
results_cond = eval_by_att_mc(X_test, y_test, idx_list, C_sets_partial)
results_cond = complete_df_real(results_cond, method = 'Partial')
results_full = pd.concat([results_full, results_marg, results_cond])

## Exhaustive 
Selective_method = Exhaustive_Selection(alpha = alpha, random_state = seed)
C_sets_exhuastive = Selective_method.multiclass_classification(X_test,
                                                                    X_calib,
                                                                    y_calib,
                                                                    left_tail = False,
                                                                    sensitive_atts_idx = idx_list, 
                                                                    bbox_mc = bbox_mc, 
                                                                    conditional = conditional_)
results_marg = eval_psets(C_sets_exhuastive, y_test)
results_marg = complete_df_real(results_marg, method = 'Exhaustive', is_marg_results = True)
results_cond = eval_by_att_mc(X_test, y_test, idx_list, C_sets_exhuastive)
results_cond = complete_df_real(results_cond, method = 'Exhaustive')
results_full = pd.concat([results_full, results_marg, results_cond])



################
# Save Results #
################
results_full.to_csv(outfile, index=False)
print("\nResults written to {:s}\n".format(outfile))
sys.stdout.flush()

# Clean up temp model directory to free up disk space
shutil.rmtree(modeldir, ignore_errors=True)