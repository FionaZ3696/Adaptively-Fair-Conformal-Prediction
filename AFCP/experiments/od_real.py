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
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

sys.path.append("../")

from third_party.utils import *
from SelectiveCI_fairness.methods import Marginal_Selection, Exhaustive_Selection, Partial_Selection, Adaptive_Selection
from SelectiveCI_fairness.black_box import Blackbox
from SelectiveCI_fairness.networks import BinaryClassification
from SelectiveCI_fairness.evals import eval_pvalues, eval_by_att_od



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
full_data_train = pd.read_csv(
    "adult.data",
    names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python', skiprows=1,
        na_values="?", dtype={0:int, 1:str, 2:int, 3:str, 4:int, 5: str, 6:str , 7:str ,8:str ,9: str, 10:int, 11:int, 12:int, 13:str,14: str})

print('Training dataset size: ', full_data_train.shape[0])

full_data_test = pd.read_csv(
    "adult.test",
    names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python', skiprows=1,
        na_values="?", dtype={0:int, 1:str, 2:int, 3:str, 4:int, 5: str, 6:str , 7:str ,8:str ,9: str, 10:int, 11:int, 12:int, 13:str,14: str})

print('Testing dataset size: ', full_data_test.shape[0])

print('Training dataset size before removing missing values: ', full_data_train.shape[0])
full_data_train.replace('nan', np.nan, inplace=True)
full_data_train.dropna(inplace=True)
print('Training dataset size after removing missing values: ', full_data_train.shape[0])

print('Testing dataset size before removing missing values: ', full_data_test.shape[0])
full_data_test.replace('nan', np.nan, inplace=True)
full_data_test.dropna(inplace=True)
print('Testing dataset size after removing missing values: ', full_data_test.shape[0])

full_data_train.reset_index(drop = True, inplace = True)
full_data_test.reset_index(drop = True, inplace = True)

print("Training data target distribution: \n", full_data_train.Target.value_counts())
print("="*60)
print("Testing data target distribution: \n", full_data_test.Target.value_counts())

# set <= 50K as inlier, >50 K as outliers
full_data_train.Target.replace('<=50K', 0, inplace = True)
full_data_train.Target.replace('>50K', 1, inplace = True)

full_data_test.Target.replace('<=50K.', 0, inplace = True)
full_data_test.Target.replace('>50K.', 1, inplace = True)

# next deal with the categorical variables
cat_col = ['Workclass','Education','Education-Num','Marital Status','Occupation','Relationship','Race','Sex','Country']

def condense_category(col_name, df_train, df_test, min_freq=0.1, new_name='Others'):
    series = pd.value_counts(df_train.loc[:,col])
    mask = (series/series.sum()).lt(min_freq)
    train_col = pd.Series(np.where(df_train.loc[:,col].isin(series[mask].index), new_name, df_train.loc[:,col]))
    test_col = pd.Series(np.where(df_test.loc[:,col].isin(series[mask].index), new_name, df_test.loc[:,col]))
    return train_col, test_col

for col in cat_col:
  train_col_new, test_col_new = condense_category(col, df_train = full_data_train, df_test = full_data_test)
  full_data_train.loc[:,col] = train_col_new
  full_data_test.loc[:,col] = test_col_new

index_Sex = full_data_test.columns.get_loc("Sex")
index_Country = full_data_test.columns.get_loc("Country")
index_Race = full_data_test.columns.get_loc("Race")
index_MaritalStatus = full_data_test.columns.get_loc("Marital Status")
index_Education = full_data_test.columns.get_loc("Education")
index_Workclass = full_data_test.columns.get_loc("Workclass")
index_Occupation = full_data_test.columns.get_loc("Occupation")
index_Relationship = full_data_test.columns.get_loc("Relationship")


# Do label encoding on the cat_data
for col in cat_col:
    full_data_train[col] = LabelEncoder().fit_transform(full_data_train[col])
    full_data_test[col] = LabelEncoder().fit_transform(full_data_test[col])
# # normalize the continuous variables
cont_col = ['Age','fnlwgt', 'Capital Gain','Capital Loss','Hours per week']
full_data_train[cont_col] = MinMaxScaler().fit_transform(full_data_train[cont_col])
full_data_test[cont_col] = MinMaxScaler().fit_transform(full_data_test[cont_col])

full_inlier_train = np.array(full_data_train)[np.where(full_data_train['Target'] == 0)[0]]
full_outlier_train = np.array(full_data_train)[np.where(full_data_train['Target'] == 1)[0]]

full_inlier_test = np.array(full_data_test)[np.where(full_data_test['Target'] == 0)[0]]
full_outlier_test = np.array(full_data_test)[np.where(full_data_test['Target'] == 1)[0]]

X_train, X_, y_train, y_ = train_test_split(np.array(full_data_train)[:,:-1], np.array(full_data_train)[:,-1],
                                                                    train_size = int(n_train_calib*0.75),
                                                                    random_state=seed)

X_calib, _, y_calib, _ = train_test_split(X_[np.where(y_ == 0)], y_[np.where(y_ == 0)],
                                          train_size = int(n_train_calib*0.25),
                                          random_state=seed)

X_test_0, _, y_test_0, _ = train_test_split(full_inlier_test[:,:-1], full_inlier_test[:,-1],
                                                                    train_size=int(n_test/2),
                                                                    random_state=seed)
X_test_1, _, y_test_1, _ = train_test_split(full_outlier_test[:,:-1], full_outlier_test[:,-1],
                                                                    train_size=int(n_test/2),
                                                                    random_state=seed)
X_test = np.concatenate((X_test_0, X_test_1), axis=0)
y_test =  np.concatenate((y_test_0, y_test_1), axis=0)

print('total number of available training data is: {:d}.'.format(len(X_train)))
print('total number of available calibration data is: {:d}.'.format(len(X_calib)))
print('total number of test data is {:d} in which {:d} are label 0 test data, {:d} are label 1 test data.'\
      .format(len(X_test), len(X_test_0), len(X_test_1)))


# convert to dataset and dataloader
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
net = BinaryClassification(input_shape = input_shape) # AutoEncoder(input_shape= input_shape)


Loss = nn.BCEWithLogitsLoss() # nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

bbox_od = Blackbox(net, device, train_loader, batch_size=batch_size, max_epoch=n_epoch,
                        learning_rate=lr, val_loader=calib_loader, criterion=criterion, optimizer=optimizer, verbose = False)
# Train the model and save snapshots regularly
bbox_od_stats = bbox_od.full_train(save_dir = '/content/od/', model_name = 'od_old_adult'+'_'+str(seed))

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
        df['selected_1'] = None
        df['selected_3'] = None
        df['selected_5'] = None
        df['selected_6'] = None
        df['selected_7'] = None
        df['selected_8'] = None
        df['selected_9'] = None
        df['selected_13'] = None
    elif is_selective: 
      assert k_hat is not None, "input selected attribute list"
      df['selected_1'] = np.mean([1 if x == [1] else 0 for x in k_hat])
      df['selected_3'] = np.mean([1 if x == [3] else 0 for x in k_hat])
      df['selected_5'] = np.mean([1 if x == [5] else 0 for x in k_hat])
      df['selected_6'] = np.mean([1 if x == [6] else 0 for x in k_hat])
      df['selected_7'] = np.mean([1 if x == [7] else 0 for x in k_hat])
      df['selected_8'] = np.mean([1 if x == [8] else 0 for x in k_hat])
      df['selected_9'] = np.mean([1 if x == [9] else 0 for x in k_hat])
      df['selected_13'] = np.mean([1 if x == [13] else 0 for x in k_hat])

    return df

idx_list = [index_Sex, index_Race, index_Country, index_Education, index_MaritalStatus, index_Occupation, index_Workclass, index_Relationship]

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

## Adaptive+
Selective_method = Adaptive_Selection(alpha = alpha, random_state = seed)
pvals_adaptive_p, k_hat_p = Selective_method.outlier_detection(X_calib,
                                                                  X_test,
                                                                  bbox_od,
                                                                  idx_list, 
                                                                  select_multiple_att = True)

results_marg = eval_pvalues(pvals_adaptive_p,  labels, [alpha])
results_marg = complete_df(results_marg, pvals = pvals_adaptive_p, method = 'Adaptive+', is_marg_results = True, is_selective = True, k_hat = k_hat_p)
results_cond = eval_by_att_od(X_test,  labels, idx_list, pvals_adaptive_p, [alpha])
results_cond = complete_df(results_cond, pvals = pvals_adaptive_p, method = 'Adaptive+', is_selective = True, k_hat = k_hat_p)
results_full = pd.concat([results_full, results_marg, results_cond])


## Marginal
Selective_method = Marginal_Selection(alpha = alpha, random_state = seed)
pvals_marginal = Selective_method.outlier_detection(X_test,
                                                    X_calib,
                                                    bbox_od = bbox_od,
                                                    left_tail = False
                                                    )
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
                                                    sensitive_atts_idx = idx_list 
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