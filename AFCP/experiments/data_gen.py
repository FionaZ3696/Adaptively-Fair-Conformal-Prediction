from itertools import islice, cycle 
import numpy as np
import pandas as pd 

# Data generating model with inside-group variance
class data_model:
  def __init__(self, p, delta1, delta0, group_perc = [0.5,0.5], K = 2, seed = 2023):
    self.K = K # number of class
    self.p = p # dimension of X
    self.delta1 = delta1 # prop of impossible to classify samples for group 1
    self.delta0 = delta0
    self.group_perc = group_perc # percentage of group 0 and group 1 [group 0, group 1]
    np.random.seed(seed)

  def sample_X(self, n):
    X = np.random.uniform(0, 1, (n, self.p))
    group_alloc = np.array([np.random.multinomial(1, self.group_perc) for i in range(X.shape[0])], dtype = float)
    group_label = np.arange(len(self.group_perc))
    groups = np.array([np.dot(group_alloc[i],group_label) for i in range(X.shape[0])], dtype = int)
    X[:,-1] = groups
    # encode noise attributes (non-sensitive)
    X[:,-2] = list(islice(cycle([1,2,3,4,5]), n))
    X[:,-3] = pd.cut(X[:,-3], bins=[0, 0.25, 0.5, 0.75, float('Inf')], labels=[6,7,8,9])
    return X

  def compute_prob(self, X):
    P = np.zeros((X.shape[0], self.K))
    K_half = max(self.K//2, 2)
    for i in range(X.shape[0]):
      if X[i, -1] == 1:
        if (X[i, 0] < self.delta1) and (X[i, 1] < 0.5):
          P[i, 0:K_half] = 1/K_half
        elif (X[i, 0] < self.delta1) and (X[i, 1] >= 0.5):
          P[i, K_half:] = 1/K_half
        else:
          idx = np.round(self.K*X[i,1]-0.5).astype(int)
          P[i,idx] = 1
      elif X[i, -1] == 0:
        if (X[i, 0] < self.delta0) and (X[i, 1] < 0.5):
          P[i, 0:K_half] = 1/K_half
        elif (X[i, 0] < self.delta0) and (X[i, 1] >= 0.5):
          P[i, K_half:] = 1/K_half
        else:
          idx = np.round(self.K*X[i,1]-0.5).astype(int)
          P[i,idx] = 1
    return P

  def sample_Y(self, X, return_prob = False):
      prob_y = self.compute_prob(X)
      g = np.array([np.random.multinomial(1,prob_y[i]) for i in range(X.shape[0])], dtype = float)
      classes_id = np.arange(self.K)
      y = np.array([np.dot(g[i],classes_id) for i in range(X.shape[0])], dtype = int)
      if return_prob:
        return prob_y, y.astype(int)
      return y.astype(int)

class Oracle:
    def __init__(self, model):
        self.model = model

    def fit(self,X,y):
        return self

    def predict(self, X):
        return self.model.sample_Y(X)

    def predict_proba(self, X):
        if(len(X.shape)==1):
            X = X.reshape((1,X.shape[0]))
        prob = self.model.compute_prob(X)
        prob = np.clip(prob, 1e-6, 1.0)
        prob = prob / prob.sum(axis=1)[:,None]
        return prob