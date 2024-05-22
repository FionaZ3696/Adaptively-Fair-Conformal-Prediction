import numpy as np 
from statsmodels.stats.multitest import multipletests 
from scipy import stats 
import sys 
from tqdm import tqdm

sys.path.append("../")
from third_party.utils import compute_conf_pvals, nonconf_scores_od, nonconf_scores_mc, arc_wrapper, union_rowwise

###################################################################################################################
#                                             Benchmark methods                                                   #
###################################################################################################################

class Exhaustive_Selection:
    def __init__(self, alpha, random_state = 2023):
        self.alpha = alpha
        self.random_state = random_state

    def outlier_detection(self, X_tests, X_calib, bbox_od, left_tail = False, sensitive_atts_idx = None):
        cal_scores = nonconf_scores_od(X_calib, bbox_od)
        test_scores = nonconf_scores_od(X_tests, bbox_od)
        conf_pvals = []
        for idx, X_test in enumerate(X_tests):
            test_score = np.array(test_scores)[idx]
            mask = np.prod(X_calib[:, sensitive_atts_idx] == X_test[sensitive_atts_idx], axis = 1)
            selected_cal_scores = np.array(cal_scores)[mask == 1]
            conf_pval = compute_conf_pvals(test_score, selected_cal_scores, left_tail)
            conf_pvals.append(conf_pval)
        return conf_pvals

    def multiclass_classification(self, X_tests, X_calib, Y_calib, bbox_mc, left_tail = False, 
                                  sensitive_atts_idx = None, conditional = True):

        cal_scores = nonconf_scores_mc(X_calib, Y_calib, bbox_mc, alpha = self.alpha, random_state = self.random_state)
        C_full = []
        n_test = X_tests.shape[0]
        conf_pval = np.full((n_test, len(set(Y_calib))), -np.Inf)

        for idx, y in enumerate(set(Y_calib)):
            scores_test_y = nonconf_scores_mc(X_tests,
                                                np.repeat(y, n_test),
                                                bbox_mc,
                                                alpha = self.alpha,
                                                random_state = self.random_state+1)

            idx_y = [np.where(Y_calib == y) if conditional else np.arange(len(Y_calib))][0]
            X_calib_y = X_calib[idx_y]
            Y_calib_y = Y_calib[idx_y]
            cal_scores_y = cal_scores[idx_y]

            for i, X_test in enumerate(X_tests):
                mask = np.prod(X_calib_y[:, sensitive_atts_idx] == X_test[sensitive_atts_idx], axis = 1)
                selected_cal_scores = np.array(cal_scores_y)[mask == 1]
                test_score = scores_test_y[i]
                conf_pval[i, idx] = compute_conf_pvals(test_score, selected_cal_scores, left_tail = left_tail)

        for i in np.arange(n_test):
            C_set = np.where(np.array(conf_pval[i]) >= self.alpha)[0].tolist()
            C_full.append(C_set)

        return C_full
            
    

class Marginal_Selection:
    def __init__(self, alpha, random_state = 2023):
        self.alpha = alpha
        self.random_state = random_state
        
    def outlier_detection(self, X_tests, X_calib, bbox_od, left_tail = False):
        cal_scores = nonconf_scores_od(X_calib, bbox_od)
        test_scores = nonconf_scores_od(X_tests, bbox_od)
        conf_pvals = []
        for idx, X_test in enumerate(X_tests):
            test_score = np.array(test_scores)[idx]
            conf_pval = compute_conf_pvals(test_score, cal_scores, left_tail)
            conf_pvals.append(conf_pval)
        return conf_pvals        

    def multiclass_classification(self, X_tests, X_calib, Y_calib, bbox_mc, left_tail = False, conditional = True):
        cal_scores = nonconf_scores_mc(X_calib, Y_calib, bbox_mc, alpha = self.alpha, random_state = self.random_state)
        C_full = []
        n_test = X_tests.shape[0]
        conf_pval = np.full((n_test, len(set(Y_calib))), -np.Inf)
        
        for idx, y in enumerate(set(Y_calib)):
            scores_test_y = nonconf_scores_mc(X_tests,
                                                np.repeat(y, n_test),
                                                bbox_mc,
                                                alpha = self.alpha,
                                                random_state = self.random_state+1)

            idx_y = [np.where(Y_calib == y) if conditional else np.arange(len(Y_calib))][0]
            cal_scores_y = cal_scores[idx_y]
            
            for i, X_test in enumerate(X_tests):
                test_score = scores_test_y[i]
                conf_pval[i, idx] = compute_conf_pvals(test_score, cal_scores_y, left_tail = left_tail)

                
        for i in np.arange(n_test):
            C_set = np.where(np.array(conf_pval[i]) >= self.alpha)[0].tolist()
            C_full.append(C_set)
            
        return C_full
            
            
            
class Partial_Selection:
    def __init__(self, alpha, random_state = 2023):
        self.alpha = alpha
        self.random_state = random_state

    def outlier_detection(self, X_tests, X_calib, bbox_od, left_tail = False, sensitive_atts_idx = None):

        cal_scores = nonconf_scores_od(X_calib, bbox_od)
        test_scores = nonconf_scores_od(X_tests, bbox_od)

        conf_pvals = []

        for idx, X_test in enumerate(X_tests):
            u_hat = []
            test_score = np.array(test_scores)[idx]
            for att_idx in sensitive_atts_idx:
                mask = np.where(X_calib[:,att_idx] == X_test[att_idx])
                selected_cal_scores = np.array(cal_scores)[mask]
                u_ = compute_conf_pvals(test_score, selected_cal_scores, left_tail)
                u_hat.append(u_)

            conf_pval = max(u_hat)
            conf_pvals.append(conf_pval)
        return conf_pvals

    def multiclass_classification(self, X_tests, X_calib, Y_calib, bbox_mc, left_tail = False, 
                                           sensitive_atts_idx = None, conditional = True):
        cal_scores = nonconf_scores_mc(X_calib, Y_calib, bbox_mc, alpha = self.alpha, random_state = self.random_state)
        C_full = []
        n_test = X_tests.shape[0]
        conf_pval = np.full((n_test, len(set(Y_calib))), -np.Inf)
        
        for idx, y in enumerate(set(Y_calib)):
            scores_test_y = nonconf_scores_mc(X_tests,
                                                np.repeat(y, n_test),
                                                bbox_mc,
                                                alpha = self.alpha,
                                                random_state = self.random_state+1)

            idx_y = [np.where(Y_calib == y) if conditional else np.arange(len(Y_calib))][0]
            X_calib_y = X_calib[idx_y]
            cal_scores_y = cal_scores[idx_y]

            for i, X_test in enumerate(X_tests):
                test_score = scores_test_y[i]
                conf_pvals = []
                for att_idx in sensitive_atts_idx:
                    mask = np.where(X_calib_y[:,att_idx] == X_test[att_idx])
                    selected_cal_scores = np.array(cal_scores_y)[mask]
                    conf_pval_ = compute_conf_pvals(test_score, selected_cal_scores, left_tail = left_tail)
                    conf_pvals.append(conf_pval_)

                conf_pval[i, idx] = np.max(conf_pvals) # axis = 0

                
        for i in np.arange(n_test):
            C_set = np.where(np.array(conf_pval[i]) >= self.alpha)[0].tolist()
            C_full.append(C_set)
        
        return C_full


###################################################################################################################
#                                             Selective conformal                                                 #
###################################################################################################################
class Adaptive_Selection:

    def __init__(self, alpha, ttest_delta = None, random_state = 2024):
        self.alpha = alpha
        self.random_state = random_state
        self.beta = 0 # filter out low-frequency events
        self.ttest_delta = ttest_delta # H0: worst miscoverage rate <= alpha + ttest_delta
        self.sig_level = 0.1 # significance level to reject t test

    def augment_data(self, X_test, y, X_calib, Y_calib):
        X = np.vstack([X_calib, X_test])
        Y = np.append(Y_calib, y)
        return [X, Y]

    def error_func_groupwise(self, phi_k, E):
        """Return the maximum error made by the attribute
        Args:
            phi_k (list): values of the function phi projected from the investigated attributes
        Returns:
            maximum error of the attribute (over all levels)
        """
        max_miscov = 0
        for m in set(phi_k):
            phi_grouped = phi_k == m
            if sum(phi_grouped)/len(phi_grouped) < self.beta:
                continue
            miscov_prop = sum(phi_grouped * E)/sum(phi_grouped)
            if miscov_prop >= max_miscov:
                max_miscov = miscov_prop
                miscov_ind = np.array(E)[np.where(phi_grouped)[0]]

        return max_miscov, miscov_ind


    def select_the_worst_group(self, err_func, att_idx, E, X_aug):
          max_max_miscov = 0

          for att in att_idx:
              max_cov_temp, miscov_ind_temp = err_func(phi_k = X_aug[:, att], E = E)
              if max_cov_temp >= max_max_miscov:
                  max_max_miscov = max_cov_temp
                  miscov_ind = miscov_ind_temp
                  k_hat = att

          if self.ttest_delta is not None:
            test_result = stats.ttest_1samp(miscov_ind, self.alpha + self.ttest_delta, alternative='greater')
            return k_hat if test_result.pvalue < self.sig_level else []

          return k_hat

    def select_multiple_groups(self, err_func, att_idx, E, X_aug):

        k_hat_s = []

        for att in att_idx.items():
            max_cov_temp, miscov_ind_temp = err_func(phi_k = X_aug[:, att], E = E)
            
            if self.ttest_delta is not None:
                test_result = stats.ttest_1samp(miscov_ind_temp, self.alpha + self.ttest_delta, alternative='greater')
            if test_result.pvalue < self.sig_level:
                k_hat_s.append(att)
        
        return k_hat_s

    def outlier_detection(self, X_calib, X_test, bbox_od, att_idx, 
                          return_khat = True, left_tail = False, select_multiple_att = False):
        conf_pval_full = []
        k_hat_full = []

        for i, X in tqdm(enumerate(X_test)):

            X_aug = np.vstack([X_calib, X])
            scores = nonconf_scores_od(X_aug, bbox_od)
            if left_tail:
                ranks = np.array(scores).argsort().argsort() + 1 #argsort return rank starting from 0, we need start from 1
            else:
                ranks = (-np.array(scores)).argsort().argsort()[:len(np.array(scores))] +1
            n_union = len(X_aug)
            u_hat = ranks/n_union
            E = [1 if val <= self.alpha else 0 for val in u_hat]
            
            if select_multiple_att: 
                k_hat = self.select_multiple_groups(self.error_func_groupwise, att_idx, E, X_aug)
            else:
                k_hat = self.select_the_worst_group(self.error_func_groupwise, att_idx, E, X_aug)

            if k_hat == []:
              k_hat_full.append(set({}))
              calib_scores = scores[:-1]

            else:
              k_hat_full.append(set({k_hat}))
              mask = np.where(X_calib[:,k_hat] == X[k_hat])
              # mask = np.prod(X_calib_y[:, k_hat] == X[k_hat], axis = 1)
              calib_scores = np.array(scores[:-1])[mask]

            conf_pval = compute_conf_pvals(scores[-1], calib_scores, left_tail = left_tail)
            conf_pval_full.append(conf_pval)     
        
        return conf_pval_full, k_hat_full


    def multiclass_classification(self, X_calib, Y_calib, X_test, bbox_mc, att_idx,
                                  return_khat = True, conditional = False, left_tail = False):

        n_test = X_test.shape[0]
        labels = np.array(list(set(Y_calib)))

        k_final = []
        C_sets_final = []

        calib_scores = nonconf_scores_mc(X_calib, Y_calib, bbox_mc, alpha = self.alpha, random_state = self.random_state)

        test_scores = np.full((n_test, len(labels)), -np.inf)
        conf_pval_y = np.full((n_test, len(labels)), -np.Inf)
        conf_pval_add = np.full((n_test, len(labels)), -np.Inf)

        for idx, y in enumerate(labels):
          test_scores[:, idx] = nonconf_scores_mc(X_test, np.repeat(y,n_test), bbox_mc, alpha = self.alpha, random_state = self.random_state)

        for i, X in tqdm(enumerate(X_test)):

          k_hat_i = []

          for idx, y in enumerate(labels):

            idx_y = [np.where(Y_calib == y) if conditional else np.arange(len(Y_calib))][0]
            X_calib_y = X_calib[idx_y]
            Y_calib_y = Y_calib[idx_y]

            X_aug, Y_aug = self.augment_data(X_calib_y, Y_calib_y, X, y)
            scores_aug = np.append(calib_scores[idx_y], test_scores[i, idx])

            n_union = X_aug.shape[0]

            # Evaluate the miscoverage indicator
            E = []
            for j in range(n_union):
              not_j = [i for i in range(n_union) if i != j]
              C_temp = arc_wrapper(scores_aug[not_j], X_aug[j][None,:], bbox_mc, self.alpha, self.random_state)
              E_temp = Y_aug[j] not in C_temp[0]
              E.append(E_temp)

            # Select the protected attribute
            k_hat = self.select_the_worst_group(self.error_func_groupwise, att_idx, E, X_aug)
            # print(k_hat)
            # print("For the {}th sample, the selected group for label {} is {}".format(i, y, k_hat))

            # construct prediction set
            if k_hat == []:
              k_hat_i.append(set({}))
              calib_scores_selected = scores_aug[:-1]

            else:
              k_hat_i.append(set({k_hat}))
              mask = np.where(X_calib_y[:,k_hat] == X[k_hat])
              # mask = np.prod(X_calib_y[:, k_hat] == X[k_hat], axis = 1)
              calib_scores_selected = np.array(scores_aug[:-1])[mask]

            conf_pval_y[i, idx] = compute_conf_pvals(scores_aug[-1], calib_scores_selected, left_tail = left_tail)
            conf_pval_add[i, idx] = compute_conf_pvals(scores_aug[-1], scores_aug[:-1], left_tail = left_tail)
          
          # The prediction sets constructed using the place-holder label y
          C_set_y = set(labels[np.where(np.array(conf_pval_y[i]) >= self.alpha)[0]])
          # The prediction set with label-conditional or marginal coverage
          C_set_add = set(labels[np.where(np.array(conf_pval_add[i]) >= self.alpha)[0]])
          # Take the union
          C_set_ = C_set_y.union(C_set_add)
          C_sets_final.append(list(C_set_))

          # k_hat_intersect = list(set.intersection(*k_hat_y[i,:]))
          # k_hat_full.append(k_hat_intersect)


          # C_hat_marg = arc_wrapper(calib_scores, X[None,:], bbox_mc, alpha = self.alpha, random_state = self.random_state)
          # C_hat_marg = set(C_hat_marg[0])

          # C_hat = list(set.union(*C_hat_i).union(C_hat_marg))
          # C_sets_final.append(C_hat)
          k_hat = list(set.intersection(*k_hat_i))
          k_final.append(k_hat)
          print("For the {}th sample, the selected attribute is {}".format(i, k_hat))

        return C_sets_final, k_final



# class Adaptive_Conformal_Selection_old:
#     def __init__(self, beta, alpha, ttest_delta = None, multiple_ttest_delta = None, random_state = 2023):
#         """
#         Args:
#            beta (float): threshold for 'frequent event'
#            alpha (float): nominal covrage level
#            ttest_delta (float): for adpative+ method, reject the null hypothesis that the maximum miscoverage rate is not greater than alpha + delta for small p value
#            strong_selective (bool): for adaptive++ method, select all groups that reject the null hypothesis
#         """
#         self.beta = beta
#         self.alpha = alpha
#         self.ttest_delta = ttest_delta
#         self.multiple_ttest_delta = multiple_ttest_delta
#         self.random_state = random_state

#     def error_func_groupwise(self, att_idx, phi_k, E):
#         """Return the maximum error made by the attribute (form 1 in the paper)
#         Args:
#             att_idx (float): index of the investigated attribute
#             phi_k (list): values of the function phi projected from the investigated attributes
#         Returns:
#             maximum error of the attribute (over all levels)
#         """
#         max_miscov = 0
#         for m in set(phi_k):
#             phi_grouped = phi_k == m
#             if sum(phi_grouped)/len(phi_grouped) < self.beta:
#                 continue
#             miscov_prop = sum(phi_grouped * E)/sum(phi_grouped)
#             if miscov_prop >= max_miscov:
#                 max_miscov = miscov_prop
#                 if (self.ttest_delta is not None) or (self.multiple_ttest_delta is not None):
#                     # return the miscoverage events for the subgroup
#                     miscov_ind = np.array(E)[np.where(phi_grouped)[0]]

#         if (self.ttest_delta is not None) or (self.multiple_ttest_delta is not None):
#             return max_miscov, miscov_ind
#         else:
#             return max_miscov

#     def select_the_worst_group(self, err_func, sensitive_att_dict, E):
#         """Return the worst attribute index
#         Args:
#             err_func (func): function to compute error for attributes
#             sensitive_att_dict (dict): {att_idx: projected value of this attribute}
#             E (array): rejection indicator of (calib union test)
#         Returns:
#             worst attribute index (the attribute that contributes to the highest error)
#         """
#         max_max_miscov = 0

#         for key, value in sensitive_att_dict.items():
#             if self.ttest_delta is not None:
#                 max_cov_temp, miscov_ind_temp = err_func(att_idx = key, phi_k = value, E = E)
#                 if max_cov_temp >= max_max_miscov:
#                     max_max_miscov = max_cov_temp
#                     k_hat = key
#                     miscov_ind = miscov_ind_temp

#             else:
#                 max_cov_temp = err_func(att_idx = key, phi_k = value, E = E)
#                 if max_cov_temp >= max_max_miscov:
#                     max_max_miscov = max_cov_temp
#                     k_hat = key
#         # print('max_max_miscoverage rate is {}'.format(max_max_miscov))

#         if self.ttest_delta is not None:
#           _, t_pvalue = stats.ttest_1samp(miscov_ind, self.alpha + self.ttest_delta, alternative='greater')

#           return {k_hat} if t_pvalue < 0.05 else set({}) # 0.2

#         return {k_hat}

#     def select_multiple_groups(self, err_func, sensitive_att_dict, E):
#         """
#         The empirical version of selecting any number of attributes, no theoretical coverage yet
#         """
#         k_hats = set({})
#         for key, value in sensitive_att_dict.items():
#             max_cov_temp, miscov_ind_temp = err_func(att_idx = key, phi_k = value, E = E)
#             _, t_pvalue = stats.ttest_1samp(miscov_ind_temp, self.alpha + self.multiple_ttest_delta, alternative='greater')
#             if t_pvalue < 0.05: #0.2
#                 k_hats.add(key)
#         if len(k_hats) == 0:
#             return set({})
#         return k_hats

#     def select_two_groups(self, err_func, sensitive_att_dict, E):
#         k_hats = set({})
#         max_miscov, miscov_ind, keys = [], [], []

#         for key, value in sensitive_att_dict.items():
#             max_cov_temp, miscov_ind_temp = err_func(att_idx = key, phi_k = value, E = E)
#             max_miscov.append(max_cov_temp)
#             miscov_ind.append(miscov_ind_temp)
#             keys.append(key)

#         sorted_miscov = sorted(enumerate(max_miscov), key=lambda x: x[1], reverse=True)
#         idx1, max1 = sorted_miscov[0]
#         max_idx1 = miscov_ind[idx1]
#         # print('max_max_miscoverage rate is {}'.format(max1))

#         _, t_pvalue = stats.ttest_1samp(max_idx1, self.alpha + self.multiple_ttest_delta, alternative='greater')
#         if t_pvalue < 0.05: # 0.2
#           k_hats.add(keys[idx1])
#           idx2, max2 = sorted_miscov[1]
#           max_idx2 = miscov_ind[idx2]
#           # print('second_max_miscoverage rate is {}'.format(max2))
#           _, t_pvalue = stats.ttest_1samp(max_idx2, self.alpha + self.multiple_ttest_delta, alternative='greater')
#           if t_pvalue < 0.05: #0.2
#             k_hats.add(keys[idx2])

#         return set({}) if len(k_hats) == 0 else k_hats

#     def augment_data(self, X_test, y, X_calib, Y_calib):
#         X = np.vstack([X_calib, X_test])
#         Y = np.append(Y_calib, y)
#         return [X, Y]


#     def multiclass_classification_homogeneous(self, X_test, X_calib, Y_calib, bbox_mc,left_tail = False,
#                                            sensitive_atts_dict = None, sensitive_atts_idx = None,
#                                            return_khat = False, conditional = True, allow_empty = False):
#         n_test = X_test.shape[0]
#         labels = np.array(list(set(Y_calib)))

#         C_full = []
#         k_hat_full = []

#         k_hat_y = np.full((n_test, len(labels)), set({}))
#         conf_pval_y = np.full((n_test, len(labels)), -np.Inf)
#         conf_pval_add = np.full((n_test, len(labels)), -np.Inf)

#         scores_calib = nonconf_scores_mc(X_calib, Y_calib, bbox_mc, alpha = self.alpha, random_state = self.random_state)

#         for idx, y in enumerate(labels):

#             scores_test_y = nonconf_scores_mc(X_test, np.repeat(y, n_test), bbox_mc, alpha = self.alpha, random_state = self.random_state)
#             idx_y = [np.where(Y_calib == y) if conditional else np.arange(len(Y_calib))][0]
#             X_calib_y = X_calib[idx_y]
#             Y_calib_y = Y_calib[idx_y]

#             # print("The place holder label is {}".format(y))

#             for i in np.arange(len(X_test)):

#                 E = []
#                 X_aug, Y_aug = self.augment_data(X_calib_y, Y_calib_y, X_test[i], y)
#                 scores_aug = np.append(scores_calib[idx_y], scores_test_y[i])
#                 n_union = X_aug.shape[0]

#                 for j in range(n_union):
#                     not_j = [i for i in range(n_union) if i != j]

#                     C_temp = predict_set_wrapper(scores_aug[not_j], X_aug[j], bbox_mc, allow_empty, self.alpha, self.random_state)
#                     E_temp = Y_aug[j] not in C_temp[0]
#                     E.append(E_temp)

#                 if sensitive_atts_dict is None:
#                     assert sensitive_atts_idx is not None, "Need to input the attribute index"
#                     sensitive_att_dict = {}
#                     for att_idx in sensitive_atts_idx:
#                         sensitive_att_dict[att_idx] = X_aug[:, att_idx]

#                 if self.multiple_ttest_delta is not None:
#                     k_hat = self.select_two_groups(self.error_func_groupwise, sensitive_att_dict, E)
#                     # k_hat = self.select_multiple_groups(self.error_func_groupwise, sensitive_att_dict, E)
#                 else:
#                     k_hat = self.select_the_worst_group(self.error_func_groupwise, sensitive_att_dict, E)

#                 print("For the {}th sample, the selected group for label {} is {}".format(i, y, k_hat))
#                 k_hat_y[i, idx] = k_hat

#                 if k_hat == {}:
#                     calib_scores_selected = scores_aug[:-1]
#                 else:
#                     mask = np.prod(X_calib_y[:, list(k_hat)] == X_test[i][list(k_hat)], axis = 1)
#                     calib_scores_selected = np.array(scores_aug[:-1])[mask == 1]

#                 conf_pval_y[i, idx] = compute_conf_pvals(scores_aug[-1], calib_scores_selected, left_tail = left_tail)
#                 conf_pval_add[i, idx] = compute_conf_pvals(scores_aug[-1], scores_aug[:-1], left_tail = left_tail)


#         for i in np.arange(n_test):

#             # The prediction sets constructed using the place-holder label y
#             C_set_y = set(labels[np.where(np.array(conf_pval_y[i]) >= self.alpha)[0]])
#             # The prediction set with label-conditional or marginal coverage
#             C_set_add = set(labels[np.where(np.array(conf_pval_add[i]) >= self.alpha)[0]])
#             # Take the union
#             C_set_ = C_set_y.union(C_set_add)
#             C_full.append(list(C_set_))

#             k_hat_intersect = list(set.intersection(*k_hat_y[i,:]))
#             k_hat_full.append(k_hat_intersect)


#             if self.multiple_ttest_delta:

#                 k_hat_union = list(set.union(*k_hat_y[i,:]))

#                 for l in k_hat_union:
#                     mask = np.prod(X_calib[:, l] == X_test[i][l], axis = 1)

#                     for y in labels:
#                         idx_y = [np.where(Y_calib == y) if conditional else np.arange(len(Y_calib))][0]
#                         calib_scores_selected = np.array(scores_calib)[mask == 1]




#         return [C_full, k_hat_full] if return_khat else C_full



#     def multiclass_classification_adpative(self, X_test, X_calib, Y_calib, bbox_mc,left_tail = False,
#                                            sensitive_atts_dict = None, sensitive_atts_idx = None,
#                                            return_khat = False, conditional = True, allow_empty = False):
#         '''

#         '''
#         n_test = X_test.shape[0]
#         labels = np.array(list(set(Y_calib)))

#         C_full = []
#         k_hat_full = []
#         k_hat_union_full = []

#         k_hat_y = np.full((n_test, len(labels)), set({}))
#         C_set_y = np.full((n_test, len(labels)), set({}))
#         C_set_add = np.full((n_test, len(labels)), set({})) # marginal or label conditional prediction set
#         if self.multiple_ttest_delta is not None:
#             C_set_eq_add = []

#         scores_calib = nonconf_scores_mc(X_calib, Y_calib, bbox_mc, alpha = self.alpha, random_state = self.random_state)

#         for idx, y in enumerate(labels):

#             scores_test_y = nonconf_scores_mc(X_test, np.repeat(y, n_test), bbox_mc, alpha = self.alpha, random_state = self.random_state)
#             idx_y = [np.where(Y_calib == y) if conditional else np.arange(len(Y_calib))][0]
#             X_calib_y = X_calib[idx_y]
#             Y_calib_y = Y_calib[idx_y]

#             # print("The place holder label is {}".format(y))

#             for i in np.arange(len(X_test)):

#                 E = []
#                 X_aug, Y_aug = self.augment_data(X_calib_y, Y_calib_y, X_test[i], y)
#                 scores_aug = np.append(scores_calib[idx_y], scores_test_y[i])
#                 n_union = X_aug.shape[0]

#                 for j in range(n_union):
#                     not_j = [i for i in range(n_union) if i != j]

#                     C_temp = predict_set_wrapper(scores_aug[not_j], X_aug[j], bbox_mc, allow_empty, self.alpha, self.random_state)
#                     E_temp = Y_aug[j] not in C_temp[0]
#                     E.append(E_temp)

#                 if sensitive_atts_dict is None:
#                     assert sensitive_atts_idx is not None, "Need to input the attribute index"
#                     sensitive_att_dict = {}
#                     for att_idx in sensitive_atts_idx:
#                         sensitive_att_dict[att_idx] = X_aug[:, att_idx]

#                 if self.multiple_ttest_delta is not None:
#                     k_hat = self.select_two_groups(self.error_func_groupwise, sensitive_att_dict, E)
#                     # k_hat = self.select_multiple_groups(self.error_func_groupwise, sensitive_att_dict, E)
#                 else:
#                     k_hat = self.select_the_worst_group(self.error_func_groupwise, sensitive_att_dict, E)

#                 # print("For the {}th sample, the selected group for label {} is {}".format(i, y, k_hat))
#                 k_hat_y[i, idx] = k_hat

#                 if k_hat == set({}):
#                     calib_scores_selected = scores_aug[:-1]
#                 else:
#                     mask = np.prod(X_calib_y[:, list(k_hat)] == X_test[i][list(k_hat)], axis = 1)
#                     calib_scores_selected = np.array(scores_aug[:-1])[mask == 1]

#                 # Construct C(X_{n+i}, y)
#                 C_y = predict_set_wrapper(calib_scores_selected, X_test[i], bbox_mc, allow_empty, self.alpha, self.random_state)
#                 if y in C_y[0]:
#                   C_set_y[i, idx] = {y}

#                 # Construct \hat{C}^{lc}(X_{n+i}) or C^{m}(X_{n+i})
#                 C_add = predict_set_wrapper(scores_aug[:-1], X_test[i], bbox_mc, allow_empty, self.alpha, self.random_state)
#                 if y in C_add[0]:
#                   C_set_add[i, idx] = {y}

#         for i in np.arange(n_test):

#             # Compute \hat{k}(X_{n+i})
#             k_hat_intersect = list(set.intersection(*k_hat_y[i,:]))
#             k_hat_full.append(k_hat_intersect)

#             if self.multiple_ttest_delta:
#                 C_set_eq_add_i = set({})
#                 k_hat_union = list(set.union(*k_hat_y[i,:]))

#                 for var in k_hat_union:
#                     selected_index = np.where(X_calib[:, var] == X_test[i][var])

#                     for y in labels:
#                         idx_y = [np.where(Y_calib == y) if conditional else np.arange(len(Y_calib))][0]
#                         idx_eqadd = np.intersect1d(selected_index, idx_y)
#                         calib_scores_selected = np.array(scores_calib)[idx_eqadd]
#                         C_eqadd = predict_set_wrapper(calib_scores_selected, X_test[i], bbox_mc, allow_empty, self.alpha, self.random_state)
#                         if y in C_eqadd[0]:
#                             C_set_eq_add_i.add(y)

#                 C_set_ = list(set.union(*C_set_add[i,:]) | set.union(*C_set_y[i,:]) | C_set_eq_add_i)
#                 C_full.append(list(C_set_))

#             else:
#                 # Construct \cup_{y}\hat{C}(X_{n+i}, y) \cup (\hat{C}^{lc}(X_{n+i}) or \hat{C}^{m}(X_{n+i}))
#                 C_set_ = list(set.union(*C_set_add[i,:]) | set.union(*C_set_y[i,:]))
#                 C_full.append(list(C_set_))


#         return [C_full, k_hat_full] if return_khat else C_full