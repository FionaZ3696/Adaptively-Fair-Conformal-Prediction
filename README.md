# AFCP (Adaptively Fair Conformal Prediction)
This paper introduces a conformal inference method to evaluate uncertainty in classification by generating prediction sets with valid coverage conditional on adaptively chosen features. These features are carefully selected to reflect potential model limitations or biases. This can be useful to find a practical compromise between efficiency---by providing informative predictions---and algorithmic fairness---by ensuring equalized coverage for the most sensitive groups. We demonstrate the validity and effectiveness of this method on simulated and real data sets.

Accompanying paper: *Conformal Classification with Equalized Coverage for Adaptively Selected Groups*.


## Contents

 - `SelectiveCI_fairness` Python package implementing our methods and several alternative benchmarks.
    - `SelectiveCI_fairness/blackbox.py` Codes to train and evaluate blackbox models.
    - `SelectiveCI_fairness/evals.py` Codes to evaluate the performance of conformal prediction sets.
    - `SelectiveCI_fairness/methods.py` Codes to build conformal prediction sets using: 1) the Marginal method, 2) the Exhaustive method, 3) the Partial method, and 4) AFCP. 
    - `SelectiveCI_fairness/networks.py`Deep networks.
 - `third_party/` Third-party Python packages.
 - `experiments/` Codes to replicate the experiments with synthetic and real data discussed in the accompanying paper.
    - `experiments/mc_sim.py` Codes to reproduce the numerical results using synthetic data for multi-class classification tasks.
    - `experiments/od_sim.py` Codes to reproduce the numerical results using synthetic data for outlier detection tasks.
    - `experiments/mc_real.py` Codes to reproduce the numerical results using the Nursery data for multi-class classification tasks.
    - `experiments/od_real.py` Codes to reproduce the numerical results using Adult Income data for outlier detection tasks.
    - `experiments/data_gen.py` Codes to generate the synthetic data used in the accompanying paper.  


    
## Prerequisites

Prerequisites for the AFCP package:
 - numpy
 - scipy
 - sklearn
 - torch
 - random
 - pathlib
 - tqdm
 - math
 - pandas
 - matplotlib
 - torchmetrics
 - statsmodels

Additional prerequisites to run the numerical experiments:
 - shutil
 - tempfile
 - pickle
 - sys
 - os
