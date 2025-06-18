import os
import sys

import lifelines
import joblib
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import argparse
from sksurv.metrics import integrated_brier_score, brier_score,cumulative_dynamic_auc
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
from lifelines.utils import concordance_index
from CoxPH_Regression_LSM_SRTR import CoxPH_LSM
from utils import GridSearch
import os
from sklearn.decomposition import PCA


# Suppress all warnings
warnings.filterwarnings("ignore")
#manual_seed = 43421654
manual_seed = 45
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)

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

def pca_reconstruct(estimated_V,p0,p1):
    estimated_V = estimated_V.view(p0,p1)
    estimated_V_T = torch.t(estimated_V)

    estimated_V_np = estimated_V.detach().cpu().numpy()
    estimated_V_T_np = estimated_V_T.detach().cpu().numpy()

    pca = PCA(n_components=3)
    z_0 = pca.fit_transform(estimated_V_np)
    z_0_torch = torch.tensor(z_0)
    z_1 = pca.fit_transform(estimated_V_T_np)
    z_1_torch = torch.tensor(z_1)
    
    reconstructed = z_0_torch @ z_1_torch.T
    reconstructed = reconstructed.view(-1)
    reconstructed = reconstructed.to(device) 
    return reconstructed


parser = argparse.ArgumentParser(description='Process some data.')
parser.add_argument('--model', default=None, help='The model I am running')
parser.add_argument('--p30a', default=False, help='weather to recustructu using p30a or not')



args = parser.parse_args()
model_name =args.model
p30a = args.p30a

if model_name == None:
    raise ValueError('you should choose the model in the argument')

if True:
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

    data_basic= data_basic.drop(['REC_COLD_ISCH_TM','REC_DISCHRG_CREAT', 'REC_CREAT', 'DON_CREAT','REC_FIRST_WEEK_DIAL_N',
          'REC_FIRST_WEEK_DIAL_Y'],axis=1)


    data_basic = torch.tensor(data_basic.values, dtype=torch.float32)

    data_mm = torch.tensor(data_mm.values, dtype=torch.float32)

    events = torch.tensor(data_y['status'].values, dtype=torch.bool)
    #events = events.to(device)

    survival_censoring_time = torch.tensor(data_y['survival_censoring_days'].values, dtype=torch.float32)
    #survival_censoring_time = survival_censoring_time.to(device)

    hla_a_encoded = torch.tensor(hla_a_encoded.values, dtype=torch.float32)
    #hla_a_encoded = hla_a_encoded.to(device)
    hla_b_encoded = torch.tensor(hla_b_encoded.values, dtype=torch.float32)
    #hla_b_encoded = hla_b_encoded.to(device)
    hla_dr_encoded = torch.tensor(hla_dr_encoded.values, dtype=torch.float32)
    #hla_dr_encoded = hla_dr_encoded.to(device)

    hla_a_pairs = torch.tensor(hla_a_pairs.values, dtype=torch.float32)
    hla_b_pairs = torch.tensor(hla_b_pairs.values, dtype=torch.float32)
    hla_dr_pairs = torch.tensor(hla_dr_pairs.values, dtype=torch.float32)

    hla_a_pairs_mask = torch.tensor(mask_a_pairs_matrix.values, dtype=torch.float32)
    hla_b_pairs_mask = torch.tensor(mask_b_pairs_matrix.values, dtype=torch.float32)
    hla_dr_pairs_mask = torch.tensor(mask_dr_pairs_matrix.values, dtype=torch.float32)


    mask_a_pairs_flattened = torch.tensor(mask_a_pairs_flattened.values, dtype=torch.float32)
    mask_b_pairs_flattened = torch.tensor(mask_b_pairs_flattened.values, dtype=torch.float32)
    mask_dr_pairs_flattened = torch.tensor(mask_dr_pairs_flattened.values, dtype=torch.float32)







dataset = TensorDataset(data_basic, data_mm,hla_a_encoded, hla_b_encoded, hla_dr_encoded, hla_a_pairs, hla_b_pairs, hla_dr_pairs, survival_censoring_time, events)



d = 2
interaction=True
num_epochs = 5000

folds = 1
num_experiments = 5
es_threshold =10
tol = 0.0001
learning_rate = 0.05


if model_name in ['litlvmV1', 'litlvmV2']:
    ls_penalty_type = 'lowRank'
    ls_penalty = True
    param_grid_1 = {
      'alpha': [1],
      'kappa': [1],
      'gamma': [1000000],
      'd':[2],
      'ls_penalty_type' : ['lowRank']
      }

      
    
elif model_name == 'elasticnet':
    ls_penalty_type = 'None'
    ls_penalty = False

    param_grid_1 = {
    'alpha': [1],
    'kappa': [1],
    'gamma': [0],
    'd':[2],
    'ls_penalty_type' : ['lowRank']
}
verbose = False

print(f'ls_penalty: {ls_penalty}, ls penalty type: {ls_penalty_type}')

total_size = len(dataset)
train_size = int(0.25 * total_size)
val_size = int(0.25 * total_size)
test_size = total_size - train_size - val_size
#####################################################
today_date = datetime.now()
date_text = today_date.strftime("%b %d")

folder_name = f"/home/mxn447/codes/neurips2024/experiments/results/cph/{date_text}"
os.makedirs(folder_name, exist_ok=True)
file_path = os.path.join(folder_name, f"{'SRTR'}_model:{model_name}_d:{param_grid_1['d']}, pca:{p30a},  lr:{learning_rate}_threshold:{es_threshold}_tol:{tol}_folds:{folds}.txt")
results_text = open(file_path, "w")

results_1 = [] 
results_ibs = []
results_auc = []


for i in range(num_experiments):
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    data_basic_train, data_mm_train, hla_a_encoded_train, hla_b_encoded_train, hla_dr_encoded_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train, survival_censoring_time_train, events_train = train_dataset[:]
    data_basic_val, data_mm_val, hla_a_encoded_val, hla_b_encoded_val, hla_dr_encoded_val,hla_a_pairs_val, hla_b_pairs_val, hla_dr_pairs_val, survival_censoring_time_val, events_val = val_dataset[:]
    data_basic_test, data_mm_test, hla_a_encoded_test, hla_b_encoded_test, hla_dr_encoded_test,hla_a_pairs_test, hla_b_pairs_test, hla_dr_pairs_test, survival_censoring_time_test, events_test = test_dataset[:]

    data_basic_temp = torch.cat((data_basic_train, data_basic_val),dim=0)
    data_mm_temp = torch.cat((data_mm_train, data_mm_val), dim=0)

    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    scaler1.fit(data_basic_temp)
    data_basic_train = scaler1.transform(data_basic_train)
    data_basic_train = torch.tensor(data_basic_train, dtype=torch.float32)
    data_basic_val = scaler1.transform(data_basic_val)
    data_basic_val = torch.tensor(data_basic_val, dtype=torch.float32)
    data_basic_test = scaler1.transform(data_basic_test)
    data_basic_test = torch.tensor(data_basic_test, dtype=torch.float32)


    scaler2.fit(data_mm_temp)
    data_mm_train = scaler2.transform(data_mm_train)
    data_mm_train = torch.tensor(data_mm_train, dtype=torch.float32)
    data_mm_val = scaler2.transform(data_mm_val)
    data_mm_val = torch.tensor(data_mm_val, dtype=torch.float32)
    data_mm_test = scaler2.transform(data_mm_test)
    data_mm_test = torch.tensor(data_mm_test, dtype=torch.float32)

    del scaler1, scaler2, data_basic_temp
    
    X_train = torch.cat((data_basic_train, data_mm_train, hla_a_encoded_train,hla_b_encoded_train, hla_dr_encoded_train),dim=1)
    X_val = torch.cat((data_basic_val, data_mm_val, hla_a_encoded_val,hla_b_encoded_val, hla_dr_encoded_val),dim=1)
    X_test = torch.cat((data_basic_test, data_mm_test, hla_a_encoded_test,hla_b_encoded_test, hla_dr_encoded_test),dim=1)

    #X_train = data_basic
    p = X_train.shape[1]
    estimated_w = nn.Parameter(torch.randn(p, requires_grad=True))#print(f'shape of X_train: {X_train.shape}, shape of estimated_w: {estimated_w.shape}')

    estimated_Va = nn.Parameter(torch.randn(hla_a_pairs_mask.shape[0],hla_a_pairs_mask.shape[1] , requires_grad=True).view(-1))


    cpuDevice = torch.device('cpu')


    #we first perform the initializations on cpu then transfer them to gpu
    mask_a_pairs_flattened = mask_a_pairs_flattened.to(cpuDevice)
    mask_b_pairs_flattened = mask_b_pairs_flattened.to(cpuDevice)
    mask_dr_pairs_flattened = mask_dr_pairs_flattened.to(cpuDevice)


    estimated_Va = estimated_Va * mask_a_pairs_flattened
    estimated_Va = estimated_Va.to(device)
    #use nn.Parameter to make it a leaf
    estimated_Va = nn.Parameter(estimated_Va)

    estimated_Vb = nn.Parameter(torch.randn(hla_b_pairs_mask.shape[0],hla_b_pairs_mask.shape[1] , requires_grad=True).view(-1))
    estimated_Vb = estimated_Vb * mask_b_pairs_flattened
    estimated_Vb = estimated_Vb.to(device)
    estimated_Vb = nn.Parameter(estimated_Vb)

    estimated_Vdr = nn.Parameter(torch.randn(hla_dr_pairs_mask.shape[0],hla_dr_pairs_mask.shape[1] , requires_grad=True).view(-1))

    estimated_Vdr = estimated_Vdr * mask_dr_pairs_flattened
    estimated_Vdr = estimated_Vdr.to(device)
    estimated_Vdr = nn.Parameter(estimated_Vdr)

    Va_shape = hla_a_pairs_mask.shape
    Vb_shape = hla_b_pairs_mask.shape
    Vdr_shape = hla_dr_pairs_mask.shape

    hla_a_pairs_mask = hla_a_pairs_mask.to(device)
    hla_b_pairs_mask = hla_b_pairs_mask.to(device)
    hla_dr_pairs_mask = hla_dr_pairs_mask.to(device)

    mask_a_pairs_flattened = mask_a_pairs_flattened.to(device)
    mask_b_pairs_flattened = mask_b_pairs_flattened.to(device)
    mask_dr_pairs_flattened = mask_dr_pairs_flattened.to(device)



    cph_1 = CoxPH_LSM(d=d, estimated_w = estimated_w, estimated_Va=estimated_Va, estimated_Vb=estimated_Vb, estimated_Vdr=estimated_Vdr,
                  Va_shape=Va_shape, Vb_shape=Vb_shape, Vdr_shape=Vdr_shape,
            interaction=interaction,  ls_penalty=ls_penalty,ls_penalty_type = ls_penalty_type)

    # this sends the model parameters to GPU
    cph_1 = cph_1.to(device)

    X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train, events_train, survival_censoring_time_train = X_train.to(cph_1.device), hla_a_pairs_train.to(cph_1.device), hla_b_pairs_train.to(cph_1.device), hla_dr_pairs_train.to(cph_1.device), events_train.to(cph_1.device), survival_censoring_time_train.to(cph_1.device)
    X_val, hla_a_pairs_val, hla_b_pairs_val, hla_dr_pairs_val, events_val, survival_censoring_time_val = X_val.to(cph_1.device), hla_a_pairs_val.to(cph_1.device), hla_b_pairs_val.to(cph_1.device), hla_dr_pairs_val.to(cph_1.device), events_val.to(cph_1.device), survival_censoring_time_val.to(cph_1.device)
    X_test, hla_a_pairs_test, hla_b_pairs_test, hla_dr_pairs_test, events_test, survival_censoring_time_test = X_test.to(cph_1.device),  hla_a_pairs_test.to(cph_1.device), hla_b_pairs_test.to(cph_1.device), hla_dr_pairs_test.to(cph_1.device), events_test.to(cph_1.device), survival_censoring_time_test.to(cph_1.device)


    gs = GridSearch(num_folds=folds, epochs=num_epochs, learning_rate=learning_rate, ES_threshold=es_threshold)

    X_combined = torch.cat((X_train, X_val), dim=0)


    hla_a_pairs_combined = torch.cat((hla_a_pairs_train, hla_a_pairs_val), dim=0)
    hla_b_pairs_combined = torch.cat((hla_b_pairs_train, hla_b_pairs_val), dim=0)
    hla_dr_pairs_combined = torch.cat((hla_dr_pairs_train, hla_dr_pairs_val), dim=0)
    # Concatenate events_train, events_val
    events_combined = torch.cat((events_train, events_val), dim=0)

    # concatenate survival censoring time
    survival_censoring_time_combined = torch.cat((survival_censoring_time_train, survival_censoring_time_val), dim=0)

    #performing grid search when we considere gamma(regularization for low-rank structure)
    if folds>1:
      cph_1._reset_parameters()
      print('finding the best hyperparameters for low rank regression')
      print('cph low rank ls penalty type:', cph_1.ls_penalty_type)
      best_estimator_1, best_hyperparameters_1 = gs.custom_grid_search_cph(model=cph_1, param_grid=param_grid_1,
              X= X_combined,  hla_a_pairs=hla_a_pairs_combined, hla_b_pairs=hla_b_pairs_combined, hla_dr_pairs=hla_dr_pairs_combined,
                hla_a_pairs_mask=hla_a_pairs_mask, hla_b_pairs_mask=hla_b_pairs_mask, hla_dr_pairs_mask=hla_dr_pairs_mask,
                                      events=events_combined, survival_censoring_time=survival_censoring_time_combined)
      
      print(f'best HP for experiment {i}: {best_hyperparameters_1} ')


      ''' I should reset the parameters here again and train both models using the best hyperparameters'''
      best_estimator_1._reset_parameters()
      

      estimated_w_1, estimated_Va_1, estimated_Vb_1, estimated_Vdr_1 = best_estimator_1.fit(X_train=X_combined,
                      hla_a_pairs_train=hla_a_pairs_combined, hla_b_pairs_train=hla_b_pairs_combined, hla_dr_pairs_train=hla_dr_pairs_combined,
                      survival_censoring_time_train=survival_censoring_time_combined, events_train=events_combined,
                      X_val=X_val , hla_a_pairs_val=hla_a_pairs_val, hla_b_pairs_val=hla_b_pairs_val, hla_dr_pairs_val=hla_dr_pairs_val,
                      survival_censoring_time_val=survival_censoring_time_val, events_val=events_val,
                      hla_a_pairs_mask=hla_a_pairs_mask, hla_b_pairs_mask=hla_b_pairs_mask, hla_dr_pairs_mask=hla_dr_pairs_mask,
                      learning_rate=learning_rate, num_epochs=num_epochs, ES_threshold = 20)


      survival_censoring_time_test = survival_censoring_time_test.detach().cpu().numpy()
      events_test= events_test.detach().cpu().numpy()

      print('making prediction using low rank w and V')
      print('shape of X_test:', X_test.shape)
      
      if p30a:
        estimated_Va_1 = pca_reconstruct(estimated_Va_1, hla_a_pairs_mask.shape[0],hla_a_pairs_mask.shape[1])
        estimated_Vb_1 = pca_reconstruct(estimated_Vb_1, hla_b_pairs_mask.shape[0],hla_b_pairs_mask.shape[1])
        estimated_Vdr_1= pca_reconstruct(estimated_Vdr_1, hla_dr_pairs_mask.shape[0],hla_dr_pairs_mask.shape[1])
      
      
      
      predict_1= best_estimator_1.predict(X_test,  hla_a_pairs_test, hla_b_pairs_test, hla_dr_pairs_test,
                estimated_w_1, estimated_Va_1, estimated_Vb_1, estimated_Vdr_1)
    elif folds==1:
      print('running experiment without cross validation')
      cph_1.alpha, cph_1.kappa, cph_1.gamma, cph_1.d, cph_1.ls_penalty_type = param_grid_1['alpha'][0], param_grid_1['kappa'][0], param_grid_1['gamma'][0], param_grid_1['d'][0], param_grid_1['ls_penalty_type'][0]
      estimated_w_1, estimated_Va_1, estimated_Vb_1, estimated_Vdr_1 = cph_1.fit(X_train=X_combined,
                hla_a_pairs_train=hla_a_pairs_combined, hla_b_pairs_train=hla_b_pairs_combined, hla_dr_pairs_train=hla_dr_pairs_combined,
                survival_censoring_time_train=survival_censoring_time_combined, events_train=events_combined,
                X_val=X_val , hla_a_pairs_val=hla_a_pairs_val, hla_b_pairs_val=hla_b_pairs_val, hla_dr_pairs_val=hla_dr_pairs_val,
                survival_censoring_time_val=survival_censoring_time_val, events_val=events_val,
                hla_a_pairs_mask=hla_a_pairs_mask, hla_b_pairs_mask=hla_b_pairs_mask, hla_dr_pairs_mask=hla_dr_pairs_mask,
                learning_rate=learning_rate, num_epochs=num_epochs, ES_threshold = 20)
      
      survival_censoring_time_test = survival_censoring_time_test.detach().cpu().numpy()
      events_test= events_test.detach().cpu().numpy()
      
      if p30a:
        estimated_Va_1 = pca_reconstruct(estimated_Va_1, hla_a_pairs_mask.shape[0],hla_a_pairs_mask.shape[1])
        estimated_Vb_1 = pca_reconstruct(estimated_Vb_1, hla_b_pairs_mask.shape[0],hla_b_pairs_mask.shape[1])
        estimated_Vdr_1= pca_reconstruct(estimated_Vdr_1, hla_dr_pairs_mask.shape[0],hla_dr_pairs_mask.shape[1])
      predict_1= cph_1.predict(X_test,  hla_a_pairs_test, hla_b_pairs_test, hla_dr_pairs_test,
                estimated_w_1, estimated_Va_1, estimated_Vb_1, estimated_Vdr_1)
      
    predict_1=predict_1.detach().cpu().numpy()
    c_index_1= concordance_index(survival_censoring_time_test, -predict_1, events_test)
    results_1.append(c_index_1)


    times, surv_matrix = cph_1.predict_survival_curve_cox_lsm(
            cph_1,
            X_combined, hla_a_pairs_combined, hla_b_pairs_combined, hla_dr_pairs_combined,survival_censoring_time_combined, events_combined,  # for baseline hazard
            X_test, hla_a_pairs_test, hla_b_pairs_test, hla_dr_pairs_test,  # target subject's features
            t_min=0.0,
            t_max=None,
            num_points=30
        )


    events_train_cpu = events_combined.cpu().numpy()                # shape [N_train]
    times_train_cpu = survival_censoring_time_combined.cpu().numpy()# shape [N_train]


    survival_train = np.zeros(
        len(times_train_cpu),
        dtype=[('event','?',), ('time','<f8',)]
    )
    for i in range(len(times_train_cpu)):
        survival_train[i] = (bool(events_train_cpu[i]), float(times_train_cpu[i]))

    # 2) Bring test data to CPU and build a structured array
    events_test_cpu = events_test               # shape [N_test]
    times_test_cpu = survival_censoring_time_test# shape [N_test]

    survival_test = np.zeros(
        len(times_test_cpu),
        dtype=[('event','?',), ('time','<f8',)]
    )
    for i in range(len(times_test_cpu)):
        survival_test[i] = (bool(events_test_cpu[i]), float(times_test_cpu[i]))

    # 3) Convert your predicted survival estimates to NumPy, as sksurv expects
    #    estimate[i, j] = probability that subject i is event-free at times[j]
    times_cpu = times.cpu().numpy()            # shape [num_points]
    estimate  = surv_matrix.cpu().numpy()      # shape [N_test, num_points]
    ibs_value = integrated_brier_score(
    survival_train,   # training set for censoring distribution
    survival_test,    # test set times
    estimate,         # shape [N_test, num_points]
    times_cpu         # shape [num_points]
    )
    
    times, bs = brier_score(survival_train, survival_test, estimate, times_cpu)
    auc, mean_auc = cumulative_dynamic_auc(survival_train, survival_test, estimate, times_cpu)


    results_ibs.append(ibs_value)
    results_auc.append(mean_auc)
    print(results_ibs)
    print(f'bs:{bs}', file=results_text)
    print(f'auc:{auc}', file=results_text)
    print('deleting all the variables before the next loop')
    del estimated_w_1, estimated_Va_1, estimated_Vdr_1
    del train_dataset, val_dataset, test_dataset
    del data_basic_train, hla_a_encoded_train, hla_b_encoded_train, hla_dr_encoded_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train, survival_censoring_time_train, events_train
    del data_basic_val, hla_a_encoded_val, hla_b_encoded_val, hla_dr_encoded_val,hla_a_pairs_val, hla_b_pairs_val, hla_dr_pairs_val, survival_censoring_time_val, events_val
    del data_basic_test, hla_a_encoded_test, hla_b_encoded_test, hla_dr_encoded_test,hla_a_pairs_test, hla_b_pairs_test, hla_dr_pairs_test, survival_censoring_time_test, events_test
    del X_train, X_val, X_test
    del estimated_w, estimated_Va, estimated_Vb, estimated_Vdr
    del Va_shape, Vb_shape, Vdr_shape
    del gs
    del X_combined, hla_a_pairs_combined,hla_b_pairs_combined,hla_dr_pairs_combined
    del events_combined, survival_censoring_time_combined
    gc.collect()

standard_error = np.std(results_1, ddof=1) / np.sqrt(len(results_1))
mean = np.mean(results_1)
print(f"{results_1}, mean: {mean}, se: {standard_error}", file=results_text)

standard_error = np.std(results_ibs, ddof=1) / np.sqrt(len(results_ibs))
mean = np.mean(results_ibs)
print(f"{results_ibs}, mean: {mean}, se: {standard_error}", file=results_text)


standard_error = np.std(results_auc, ddof=1) / np.sqrt(len(results_auc))
mean = np.mean(results_auc)
print(f"{results_auc}, mean: {mean}, se: {standard_error}", file=results_text)

print(f'times:{times}', file=results_text)

print(param_grid_1, file=results_text)
results_text.close()



