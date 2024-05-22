import numpy as np 
import pandas as pd 
import torch
import sys
from scipy.stats.mstats import mquantiles
from statsmodels.stats.multitest import multipletests

###################################################################################################################
#                                             Aux functions                                                       #
###################################################################################################################

def nonconf_scores_od(inputs, bbox_od):
    inputs = torch.from_numpy(inputs).float()
    # scores: for autoencoder is MSE loss, outlier larger
    scores = bbox_od.net.get_anomaly_scores(inputs)
    return scores

def compute_conf_pvals(test_score, calib_scores, left_tail = False):
  # notice that if larger scores stand for outliers, we should use right tail: calib scores >= test scores.
  n_cal = len(calib_scores)
  if left_tail:
    pval = (1.0 + np.sum(np.array(calib_scores) <= np.array(test_score))) / (1.0 + n_cal)
  else:
    pval = (1.0 + np.sum(np.array(calib_scores) >= np.array(test_score))) / (1.0 + n_cal)
  return pval

def nonconf_scores_mc(X_cal, Y_cal, bbox_mc, alpha = 0.1, random_state = 2023):
  X_cal = torch.from_numpy(X_cal).float()
  Y_cal = torch.from_numpy(Y_cal)
  p_hat_calib = bbox_mc.net.predict_prob(X_cal)
  n_cal = X_cal.shape[0]
  grey_box = ProbAccum(p_hat_calib)
  rng = np.random.default_rng(random_state)
  epsilon = rng.uniform(low=0.0, high=1.0, size=n_cal)
  alpha_max = grey_box.calibrate_scores(Y_cal, epsilon=epsilon)
  scores = alpha - alpha_max
  return scores

def union_rowwise(arr):
    # Convert each numpy array to a set and then calculate the union
    return [list(set.union(*[set(subitem) for subitem in row])) for row in arr]

class arc:
    def __init__(self, alpha, random_state=2024):
        self.alpha = alpha
        self.random_state = random_state

    def calibrate(self, calib_scores):

        # n_cal = X_calib.shape[0]
        # self.bbox_mc = bbox_mc
        # cal_scores = nonconf_scores_mc(X_calib, Y_calib, self.bbox_mc, alpha = self.alpha, random_state = self.random_state)
        n_cal = len(calib_scores)
        level_adjusted = 1 if n_cal == 0 else (1.0-self.alpha)*(1.0+1.0/float(n_cal))

        calib_scores = np.append(calib_scores, np.inf)
        alpha_correction = mquantiles(calib_scores, prob=level_adjusted)

        # Store calibrate level
        self.alpha_calibrated = self.alpha - alpha_correction

    def predict(self, X_test, bbox_mc, allow_empty = True):
        X_test = torch.from_numpy(X_test).float()
        n_test = X_test.shape[0]
        rng = np.random.default_rng(self.random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n_test)
        p_hat = bbox_mc.net.predict_prob(X_test)
        grey_box = ProbAccum(p_hat)
        S_hat = grey_box.predict_sets(self.alpha_calibrated, epsilon=epsilon, allow_empty=allow_empty)
        return S_hat

def arc_wrapper(calib_scores, X_test, bbox_mc, alpha = 0.1, random_state = 2024):
    method = arc(alpha, random_state)
    method.calibrate(calib_scores)
    C_sets = method.predict(X_test, bbox_mc)
    return C_sets


def calibrate_alpha(scores, alpha = 0.1):
  n_cal = len(scores)
  if n_cal == 0:
    return alpha
  level_adjusted = (1.0 - alpha)*(1.0 + 1.0/float(n_cal))
  alpha_correction = mquantiles(scores, prob=level_adjusted)
  alpha_calibrated = alpha - alpha_correction
  return alpha_calibrated

def predict_set(X_test, bbox_mc, alpha_calibrated, allow_empty = True, random_state = 2023):
  X_test = torch.from_numpy(X_test).float()
  n = X_test.shape[0]
  rng = np.random.default_rng(random_state)
  epsilon = rng.uniform(low=0.0, high=1.0, size=n)
  p_hat = bbox_mc.net.predict_prob(X_test)
  grey_box = ProbAccum(p_hat)
  S_hat = grey_box.predict_sets(alpha_calibrated, epsilon=epsilon, allow_empty = allow_empty)
  return S_hat

def predict_set_wrapper(calib_scores, X_test, bbox_mc, allow_empty, alpha, random_state):
    alpha_adjusted = calibrate_alpha(calib_scores, alpha = alpha)
    C = predict_set(X_test[None,:], bbox_mc, alpha_adjusted, allow_empty, random_state)
    return C




class ProbAccum:
    def __init__(self, prob):
        self.n, self.K = prob.shape
        # the label corresponding to sorted probability (from the largest to the smallest)
        self.order = np.argsort(-prob, axis=1)
        # find the rank of each label based on the sorted probability
        self.ranks = np.empty_like(self.order)
        for i in range(self.n):
            self.ranks[i, self.order[i]] = np.arange(len(self.order[i]))
        # sort the predicted probabilities
        self.prob_sort = -np.sort(-prob, axis=1)
        # cumsum on the sorted probability
        self.Z = np.round(self.prob_sort.cumsum(axis=1),9)

    def predict_sets(self, alpha, epsilon=None, allow_empty=True):
        if alpha>0:
            # if positive alpha, return L[i] as the largest index that has cumprob greater than 1-alpha
            L = np.argmax(self.Z >= 1.0-alpha, axis=1).flatten()
        else:
            # if alpha = 0, return L[i] be the largest index so that S will be the entire set
            L = (self.Z.shape[1]-1)*np.ones((self.Z.shape[0],)).astype(int)
        if epsilon is not None:
            # Corresponding to the V and U<= V part in equation (5) in the paper
            Z_excess = np.array([ self.Z[i, L[i]] for i in range(self.n) ]) - (1.0-alpha)
            p_remove = Z_excess / np.array([ self.prob_sort[i, L[i]] for i in range(self.n) ])
            remove = epsilon <= p_remove
            for i in np.where(remove)[0]:
                if not allow_empty:
                    # L[i] corresponding to the maximum index (later add probability up to) for the sample i
                    L[i] = np.maximum(0, L[i] - 1)  # Note: avoid returning empty sets
                else:
                    L[i] = L[i] - 1

        # Return prediction set
        S = [ self.order[i,np.arange(0, L[i]+1)] for i in range(self.n) ]
        return(S)

    def calibrate_scores(self, Y, epsilon=None):
        Y = np.atleast_1d(Y)
        if isinstance(Y, int) == False:
          Y = list(map(int, Y))
        n2 = len(Y)
        ranks = np.array([ self.ranks[i,Y[i]] for i in range(n2) ])
        # the cumulative probabilities up to the rank of the true label
        prob_cum = np.array([ self.Z[i,ranks[i]] for i in range(n2) ])
        # the predicted prob for the true label
        prob = np.array([ self.prob_sort[i,ranks[i]] for i in range(n2) ])
        alpha_max = 1.0 - prob_cum
        if epsilon is not None:
            alpha_max += np.multiply(prob, epsilon)
        else:
            alpha_max += prob
        alpha_max = np.minimum(alpha_max, 1)
        return alpha_max


class Conformal_PVals:
    '''
    Class for computing conformal p-values for any test set
    '''
    def __init__(self, net, device, cal_loader, model_path,
                 verbose = True, random_state = 2023) -> None:
        self.device = device

        self.net = net
        saved_stats = torch.load(model_path, map_location = self.device)
        self.net.load_state_dict(saved_stats['model_state'])
        self.cal_loader = cal_loader
        self.verbose = verbose

        self.compute_scores()
        if self.verbose:
            print('Initialization done!')
            sys.stdout.flush()

    def compute_scores(self):
        self.cal_scores = []
        for inputs, _ in self.cal_loader:
            inputs = inputs.to(self.device)
            scores = self.net.get_anomaly_scores(inputs)
            self.cal_scores += scores

    def _compute_pval_single(self, test_input, left_tail):
        '''
        Calculate the conformal p-value for a single test point
        '''
        test_input = test_input.to(self.device)
        ## for experiment with image data
        # test_score = self.net.get_anomaly_scores(test_input)
        test_score = self.net.get_anomaly_scores(test_input.reshape([1,len(test_input)]))
        n_cal = len(self.cal_scores)
        if left_tail:
            pval = (1.0 + np.sum(np.array(self.cal_scores) <= np.array(test_score))) / (1.0 + n_cal)
        else:
            pval = (1.0 + np.sum(np.array(self.cal_scores) >= np.array(test_score))) / (1.0 + n_cal)
        return pval

    def compute_pvals(self, test_inputs, left_tail = False):
        """ Compute the conformal p-values for test points using a calibration set
        """
        test_inputs = test_inputs.to(self.device)
        n_test = len(test_inputs)
        pvals = -np.zeros(n_test)

        for i in range(n_test):
            pvals[i] = self._compute_pval_single(test_inputs[i], left_tail)

        if self.verbose:
            print("Finished computing p-values for {} test points.".format(n_test))
            sys.stdout.flush()
        return pvals


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