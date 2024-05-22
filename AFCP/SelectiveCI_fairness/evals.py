import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests


###################################################################################################################
#                                             Evaluation                                                          #
###################################################################################################################



def eval_psets(S, y):
    coverage = np.mean([y[i] in S[i] for i in range(len(y))])
    length = np.mean([len(S[i]) for i in range(len(y))])
    idx_cover = np.where([y[i] in S[i] for i in range(len(y))])[0]
    length_cover = np.mean([len(S[i]) for i in idx_cover])
    result = pd.DataFrame({'Coverage': [coverage], 'Length': [length], 'Length_cover': [length_cover]})
    return result

def eval_by_att_mc(X_test, y_test, att_idx_full, C_sets):
  '''Input a np.array as test features,
  alpha is a list'''
  results = pd.DataFrame()
  for att_idx in att_idx_full:
    for level in set(X_test[:,att_idx]):
      group_ = np.where(X_test[:,att_idx]==level)
      result_ = eval_psets(np.array(C_sets, dtype = 'object')[group_], y_test[group_])
      result_['n_count'] = sum(X_test[:,att_idx] == level)
      # result_['attribute_level'] = str(att_idx) + str('_') + str(level)
      result_['attribute_idx'] = att_idx
      result_['attribute_level'] = level
      results = pd.concat([results, result_])

  return results

def eval_by_att_label(X_test, y_test, att_idx_full, C_sets):
  '''Input a np.array as test features,
  alpha is a list'''
  results = pd.DataFrame()
  for att_idx in att_idx_full:
    for level in set(X_test[:,att_idx]):
      for label in set(y_test):
        group_ = np.where((X_test[:,att_idx]==level) & (y_test == label))
        result_ = eval_psets(np.array(C_sets, dtype = 'object')[group_], y_test[group_])
        result_['n_count'] = sum(X_test[:,att_idx] == level)
        # result_['attribute_level'] = str(att_idx) + str('_') + str(level)
        result_['attribute_idx'] = att_idx
        result_['attribute_level'] = level
        result_['label'] = label
        results = pd.concat([results, result_])

  return results

def eval_by_label(y_test, C_sets):
  results = pd.DataFrame()
  for label in set(y_test):
    group_ = np.where(y_test == label)
    y = y_test[group_]
    S = np.array(C_sets, dtype = 'object')[group_]
    coverage = np.mean([y[i] in S[i] for i in range(len(y))])
    length = np.mean([len(S[i]) for i in range(len(y))])
    idx_cover = np.where([y[i] in S[i] for i in range(len(y))])[0]
    result_ = eval_psets(np.array(C_sets, dtype = 'object')[group_], y_test[group_])
    result_['n_count'] = sum(y_test == label)
    result_['label'] = label
    results = pd.concat([results, result_])
  return results



def eval_by_att_mc(X_test, y_test, att_idx_full, C_sets):
  '''Input a np.array as test features,
  alpha is a list'''
  results = pd.DataFrame()
  for att_idx in att_idx_full:
    for level in set(X_test[:,att_idx]):
      group_ = np.where(X_test[:,att_idx]==level)
      result_ = eval_psets(np.array(C_sets, dtype = 'object')[group_], y_test[group_])
      result_['n_count'] = sum(X_test[:,att_idx] == level)
      # result_['attribute_level'] = str(att_idx) + str('_') + str(level)
      result_['attribute_idx'] = att_idx
      result_['attribute_level'] = level
      results = pd.concat([results, result_])

  return results

def eval_by_att_od(test_feats, test_labels, att_idx_full, pvals, alpha):
  '''Input a np.array as test features,
  alpha is a list'''
  results = pd.DataFrame()
  for att_idx in att_idx_full:
    for level in set(test_feats[:,att_idx]):
      group_ = np.where(test_feats[:,att_idx]==level)
      # u_hat_mean = np.mean(pvals[group_])
      # u_hat_sd = np.std(pvals[group_])
      result_ = eval_pvalues(np.array(pvals, dtype = 'object')[group_], test_labels[group_], alpha)
      result_['n_count'] = sum(test_feats[:,att_idx] == level)
      # result_['attribute_level'] = str(att_idx) + str('_') + str(level)
      result_['attribute_idx'] = att_idx
      result_['attribute_level'] = level
      # if return_u_hat:
      #   result_['u_hat'] = [pvals[group_]]
      #   result_['u_hat_mean'] = u_hat_mean
      #   result_['u_hat_std'] = u_hat_sd

      results = pd.concat([results, result_])

  return results

def eval_pvalues(pvals, Y, alpha_list):
    # make sure pvals and Y are numpy arrays
    pvals = np.array(pvals)
    Y = np.array(Y)

    # Evaluate with BH and Storey-BH
    fdp_list = -np.ones((len(alpha_list),1))
    power_list = -np.ones((len(alpha_list),1))
    rejections_list = -np.ones((len(alpha_list),1))
    fdp_storey_list = -np.ones((len(alpha_list),1))
    power_storey_list = -np.ones((len(alpha_list),1))
    rejections_storey_list = -np.ones((len(alpha_list),1))
    for alpha_idx in range(len(alpha_list)):
        alpha = alpha_list[alpha_idx]
        rejections_list[alpha_idx], fdp_list[alpha_idx], power_list[alpha_idx] = filter_BH(pvals, alpha, Y)
        rejections_storey_list[alpha_idx], fdp_storey_list[alpha_idx], power_storey_list[alpha_idx] = filter_StoreyBH(pvals, alpha, Y)
    results_tmp = pd.DataFrame({})
    results_tmp["Alpha"] = alpha_list
    results_tmp["BH-Rejections"] = rejections_list
    results_tmp["BH-FDP"] = fdp_list
    results_tmp["BH-Power"] = power_list
    results_tmp["Storey-BH-Rejections"] = rejections_storey_list
    results_tmp["Storey-BH-FDP"] = fdp_storey_list
    results_tmp["Storey-BH-Power"] = power_storey_list
    # Evaluate with fixed threshold
    fpr_list = -np.ones((len(alpha_list),1))
    tpr_list = -np.ones((len(alpha_list),1))
    rejections_list = -np.ones((len(alpha_list),1))
    for alpha_idx in range(len(alpha_list)):
        alpha = alpha_list[alpha_idx]
        rejections_list[alpha_idx], fpr_list[alpha_idx], tpr_list[alpha_idx] = filter_fixed(pvals, alpha, Y)
    results_tmp["Fixed-Rejections"] = rejections_list
    results_tmp["Fixed-FPR"] = fpr_list
    results_tmp["Fixed-TPR"] = tpr_list
    return results_tmp

def filter_StoreyBH(pvals, alpha, Y, lamb=0.5):
    n = len(pvals)
    R = np.sum(pvals<=lamb)
    pi = (1+n-R) / (n*(1.0 - lamb))
    pvals[pvals>lamb] = 1
    return filter_BH(pvals, alpha/pi, Y)

def filter_fixed(pvals, alpha, Y):
    is_nonnull = (Y==1)
    reject = (pvals<=alpha)
    rejections = np.sum(reject)
    if rejections>0:
        if np.sum(Y==0)>0:
            fpr = np.mean(reject[np.where(Y==0)[0]])
        else:
            fpr = 0
        if np.sum(Y==1)>0:
            tpr = np.mean(reject[np.where(Y==1)[0]])
        else:
            tpr = 0
    else:
        fpr = 0
        tpr = 0
    return rejections, fpr, tpr


def filter_BH(pvals, alpha, Y):
    is_nonnull = (Y==1)
    reject, pvals_adj, _, _ = multipletests(pvals, alpha, method="fdr_bh")
    rejections = np.sum(reject)
    if rejections>0:
        fdp = 1-np.mean(is_nonnull[np.where(reject)[0]])
        power = np.sum(is_nonnull[np.where(reject)[0]]) / np.sum(is_nonnull)
    else:
        fdp = 0
        power = 0
    return rejections, fdp, power
