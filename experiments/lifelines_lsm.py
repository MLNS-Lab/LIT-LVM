import os
import sys
from CoxPH_Regression_LSM_SRTR import CoxPH_LSM

import os
import lifelines
import joblib
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation
import lifelines
import joblib
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import argparse

from datetime import datetime
import torch
import numpy as np
import warnings
#from utils import GridSearch
import pandas as pd
import gc
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import copy
from torch.utils.data import TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from CoxPH_Regression_LSM_SRTR import CoxPH_LSM
import os


# Check if TPU is available
if "TPU_NAME" in os.environ:
    device = torch.device("xla")  # XLA is PyTorch's TPU device
    print("Using TPU")
else:
    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

def filter_columns(df):
    '''This function take in dataframes that are one-hot encoded and return the columns where the frequency of 1's are more than 100'''
    cols_to_keep = df.columns[df.apply(lambda col: col.sum()) > 100]
    return df[cols_to_keep]


#reading the data
print('reading the data')
data_name = ['data_basic', 'data_mm', 'data_y', 'hla_a_encoded', 'hla_b_encoded', 'hla_dr_encoded', 'hla_a_pairs', 'hla_b_pairs', 'hla_dr_pairs', 'mask_a_pairs_matrix', 'mask_b_pairs_matrix', 'mask_dr_pairs_matrix', 'mask_a_pairs_flattened', 'mask_b_pairs_flattened', 'mask_dr_pairs_flattened']
for data in data_name:
    try:
      vars()[f'{data}'] = pd.read_csv(f'{data}.csv')
    except:
      vars()[f'{data}'] = pd.read_csv(f'{data}')

try:
  data_basic = data_basic.drop(['Unnamed: 0'], axis=1)
except:
  pass
try:
  data_mm = data_mm.drop(['Unnamed: 0'], axis=1)
except:
  pass
try:
  hla_a_encoded = hla_a_encoded.drop(['Unnamed: 0'], axis=1)
except:
  pass
try:
  hla_b_encoded = hla_b_encoded.drop(['Unnamed: 0'], axis=1)
except:
  pass
try:
  hla_dr_encoded = hla_dr_encoded.drop(['Unnamed: 0'], axis=1)
except:
  pass
try:
  hla_a_pairs = hla_a_pairs.drop(['Unnamed: 0'], axis=1)
except:
  pass
try:
  hla_b_pairs = hla_b_pairs.drop(['Unnamed: 0'], axis=1)
except:
  pass
try:
  hla_dr_pairs = hla_dr_pairs.drop(['Unnamed: 0'], axis=1)
except:
  pass
try:
  data_y = data_y.drop(['Unnamed: 0'], axis=1)
except:
  pass

data_basic= data_basic.drop(['REC_COLD_ISCH_TM','REC_DISCHRG_CREAT', 'REC_CREAT', 'DON_CREAT','REC_FIRST_WEEK_DIAL_N',
       'REC_FIRST_WEEK_DIAL_Y'],axis=1)

train_test_index = joblib.load('train_test_index.pkl')



if False:
    hla_a_pairs = filter_columns(hla_a_pairs)
    hla_b_pairs = filter_columns(hla_b_pairs)
    hla_dr_pairs = filter_columns(hla_dr_pairs)
    hla_a_encoded = filter_columns(hla_a_encoded)
    hla_b_encoded = filter_columns(hla_b_encoded)
    hla_dr_encoded = filter_columns(hla_dr_encoded)



X = pd.concat([data_basic, hla_a_encoded, hla_b_encoded, hla_dr_encoded, hla_a_pairs, hla_b_pairs, hla_dr_pairs], axis=1)
ls1 = X.columns

summary = pd.read_csv(f'{summary_path}/summary_{0}_out.csv')

# Ensure that data_y is in the correct format
data_y['status'] = data_y['status'].astype(bool)
data_y['survival_censoring_days'] = data_y['survival_censoring_days'].astype(float)
ls2 = list(summary['covariate'])
ls1 = X.columns
feat_mask = [elem in ls2 for elem in ls1]
# Combine X and data_y to ensure alignment
X = torch.tensor(X.values)
X = X[:,feat_mask]

survival_censoring_time = torch.tensor(data_y['survival_censoring_days'].values, dtype=torch.float32)
events = torch.tensor(data_y['status'].values, dtype=torch.bool)



os.chdir(summary_path)
c_index_ls = []
for i in range(10):
    print(f'i:{i}')
    if i == 1:
       continue
    summary = pd.read_csv(f'summary_{i}_out.csv')
    train_index = train_test_index[f'{i}']['train_indices']
    test_index = train_test_index[f'{i}']['test_indices']

    X_train = copy.deepcopy(X[train_index])
    X_test = copy.deepcopy(X[test_index])
    events_train = events[train_index]
    events_test = events[test_index]
    sruvival_censoring_time_train = survival_censoring_time[train_index]
    survival_censoring_time_test = survival_censoring_time[test_index]
    scaler1 = StandardScaler()
    scaler1.fit(X_train)
    X_train = scaler1.transform(X_train)
    X_test = scaler1.transform(X_test)
    X_train = torch.tensor(X_train,dtype=torch.float32)
    X_test = torch.tensor(X_test,dtype=torch.float32)
    coef_values = summary['coef'].values
    estimated_w = torch.tensor(coef_values, dtype=torch.float32)
    cph_1 = CoxPH_LSM(d=1, estimated_w = estimated_w, interaction=False,  ls_penalty=False)
    cph_1 = cph_1.to(device)
    predict_1= cph_1.predict(X_train=X_test,  hla_a_pairs_train=None, hla_b_pairs_train=None, hla_dr_pairs_train=None,
              estimated_w = estimated_w, estimated_Va=None, estimated_Vb=None, estimated_Vdr=None)
    
    predict_1=predict_1.detach().cpu().numpy()
    c_index_1= concordance_index(survival_censoring_time_test, -predict_1, events_test)
    c_index_ls.append(round(c_index_1,3))
    print(c_index_ls)
print( round(np.mean(c_index_ls),3) )
