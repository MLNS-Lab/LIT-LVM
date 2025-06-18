import os
import sys

from itertools import product
from datetime import datetime
import os
import torch
import numpy as np
import argparse
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from utils import GridSearch
import pandas as pd
import copy
import gc
import datetime
from scipy.io.arff import loadarff 
import torch
import warnings
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import copy
from torch.utils.data import TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestRegressor

from dataset import data_loader

# Suppress all warnings
warnings.filterwarnings("ignore")
#manual_seed = 434214
manual_seed = 454214
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)


import torch
import os


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



parser = argparse.ArgumentParser(description='Process some data.')
parser.add_argument('--dataset', default='tecator.arff', help='Name of the dataset (default: pol)')
parser.add_argument('--model', default=None, help='The model I am running')

args = parser.parse_args()
dataset_name = args.dataset
model_name =args.model

if model_name == None:
    raise ValueError('you should choose the model in the argument')
X,y = data_loader(dataset_name=dataset_name, task='regression')



def compute_interactions_in_place(X_tensor,vdim):
    indices = torch.arange(vdim)
    combinations = torch.cartesian_prod(indices, indices)
    num_samples = X_tensor.size(0)
    X_interaction = torch.empty(num_samples, len(combinations))

    for i, (idx1, idx2) in enumerate(combinations):
        X_interaction[:, i] = X_tensor[:, idx1] * X_tensor[:, idx2]

    return X_interaction



def weight_generation(p, mask_flattened):
        estimated_w = nn.Parameter(torch.randn(p+1, requires_grad=True)) #print(f'shape of X_train: {X_train.shape}, shape of estimated_w: {estimated_w.shape}')
        estimated_V = nn.Parameter(torch.randn(p,p , requires_grad=True).view(-1))
        estimated_V = estimated_V * mask_flattened
        estimated_V = nn.Parameter(estimated_V)
        return estimated_w, estimated_V

def standard_error(ls):
    return round(np.std(ls, ddof=1) / np.sqrt(len(ls)),3 )

def run_experiment(d,es_threshold, learning_rate, param_grid_rf, param_grid_1, param_grid_2, num_experiments, results_text):
    results_rf = [] # result of direct random forest

    results_R2_1 = [] # low rank model with interaction
    results_R2_2 = [] # elasticnet with interaction
    results_R2_3 = [] # elasticnet wo interaction

    results_MSE_1 = [] # low rank model with interaction
    results_MSE_2 = [] # elasticnet with interaction
    results_MSE_3 = [] # elasticnet wo interaction

    if model_name in ['litlvmV1', 'litlvmV2']:
        if model_name == 'litlvmV1':
            from linreg_LIT_LVM_V1 import LinearRegression_LIT_LVM
            print('litlvV1 chosen')
        elif model_name == 'litlvmV2':
            from linreg_LIT_LVM_V2 import LinearRegression_LIT_LVM
            print('litlvmV2 chosen')
        else:
            raise ValueError('the model should be either litlvmV1 or litlvmV2')

        model_1_train = True
        model_2_train = False
        model_3_train = False
        model_rf_train = False
    elif model_name == 'elasticnet':
        from linreg_LIT_LVM_V1 import LinearRegression_LIT_LVM

        print('the model is elasticnet')
        model_1_train = False
        model_2_train = True
        model_3_train = True
        model_rf_train = False
    else:
        raise ValueError('Choose the right model name')
    
    for i in range(num_experiments):
    

        print(f'experiment {i}')
        # First split the data into training and remaining data
        train_idx, test_idx = train_test_split(range(len(X)), test_size=(val_size + test_size), random_state=42+i, shuffle=True)
        # Then split the remaining data into validation and test data
        val_idx, test_idx = train_test_split(test_idx, test_size=(test_size / (val_size + test_size)), random_state=42+i, shuffle=True)
        X_values = X.values
        X_values = X_values.astype(np.float32)
        X_rf = X_values
        X_tensor = torch.tensor(X_values, dtype=torch.float32)
        y_tensor = y.astype(np.float32)
        y_rf = y
        y_tensor = torch.tensor(y_tensor.values, dtype=torch.float32)
        y_tensor = torch.squeeze(y_tensor)

        vdim= X_tensor.shape[1]
        indices = torch.arange(vdim)
        combinations = torch.cartesian_prod(indices, indices)
        #X_interaction = X_tensor[:, combinations[:, 0]] * X_tensor[:, combinations[:, 1]]



        X_train, X_val, X_test = X_tensor[train_idx], X_tensor[val_idx], X_tensor[test_idx]
        y_train, y_val, y_test = y_tensor[train_idx], y_tensor[val_idx], y_tensor[test_idx]

        X_interaction_train = compute_interactions_in_place(X_train, vdim)
        X_interaction_val = X_val[:, combinations[:, 0]] * X_val[:, combinations[:, 1]]
        X_interaction_test = X_test[:, combinations[:, 0]] * X_test[:, combinations[:, 1]]


        #dividing cpu matrices into train test splits
        X_train_rf, X_val_rf, X_test_rf = X_rf[train_idx], X_rf[val_idx], X_rf[test_idx]
        y_train_rf, y_val_rf, y_test_rf = y_rf[train_idx], y_rf[val_idx], y_rf[test_idx]
        #X_interaction_train, X_interaction_val, X_interaction_test = X_interaction[train_idx], X_interaction[val_idx], X_interaction[test_idx]

        del X_tensor, y_tensor

        cpuDevice = torch.device('cpu')
        mask = torch.triu(torch.ones(vdim, vdim), diagonal=1)
        mask_flattened = mask.view(-1)
        mask_flattened = mask_flattened.to(cpuDevice)


        p = X_train.shape[1]




        X_train, X_interaction_train, y_train = X_train.to(device), X_interaction_train.to(device), y_train.to(device)
        X_test, X_interaction_test, y_test = X_test.to(device), X_interaction_test.to(device), y_test.to(device)
        X_val, X_interaction_val, y_val = X_val.to(device), X_interaction_val.to(device), y_val.to(device) 

        gs = GridSearch(num_folds=folds, epochs=num_epochs,ES_threshold=es_threshold, batch_size = len(X_train), learning_rate=learning_rate)


        scaler1 = StandardScaler()
        scaler2 = StandardScaler()


        scaler1.fit(X_train.detach().cpu().numpy())
        scaler2.fit(X_interaction_train.detach().cpu().numpy())

        X_train = scaler1.transform(X_train.detach().cpu().numpy())
        X_train = torch.tensor(X_train,dtype=torch.float32).to(device)
        X_val = scaler1.transform(X_val.detach().cpu().numpy())
        X_val = torch.tensor(X_val,dtype=torch.float32).to(device)
        X_test = scaler1.transform(X_test.detach().cpu().numpy())
        X_test = torch.tensor(X_test,dtype=torch.float32).to(device)

        X_interaction_train = scaler2.transform(X_interaction_train.detach().cpu().numpy())
        X_interaction_train = torch.tensor(X_interaction_train,dtype=torch.float32).to(device)
        X_interaction_val = scaler2.transform(X_interaction_val.detach().cpu().numpy())
        X_interaction_val = torch.tensor(X_interaction_val,dtype=torch.float32).to(device)
        X_interaction_test = scaler2.transform(X_interaction_test.detach().cpu().numpy())
        X_interaction_test = torch.tensor(X_interaction_test,dtype=torch.float32).to(device)


        if model_rf_train:
            print('training random forest')   
            rf = RandomForestRegressor(random_state=42+i)
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=2, scoring='neg_mean_squared_error')
            grid_search.fit(X_train_rf, y_train_rf)
            best_rf = grid_search.best_estimator_
            #best_params = grid_search.best_params_
            y_test_rf = best_rf.predict(X_test_rf)[:, 1]
            r2_score_rf = r2_score(y_test_rf, y_test)
            results_rf.append(r2_score_rf)

            y_train_rf = best_rf.predict(X_train_rf)[:, 1]
            y_val_rf = best_rf.predict_proba(X_val_rf)[:, 1]


            y_train_rf = torch.tensor(y_train_rf,dtype=torch.float32).to(device)
            y_val_rf = torch.tensor(y_val_rf,dtype=torch.float32).to(device)
            y_test_rf = torch.tensor(y_test_rf,dtype=torch.float32).to(device)

            del grid_search, best_rf
            
        if model_1_train:
            print('model 1 hp tuning')
            
            estimated_w, estimated_V = weight_generation(p, mask_flattened)

            model_1 = LinearRegression_LIT_LVM(d=d, estimated_w=estimated_w, estimated_V=estimated_V, interaction = True, regularization=True,
                                        ls_penalty=True, ls_penalty_type = ls_penalty_type)
            model_1 = model_1.to(device)

            best_estimator_1, best_hyperparameters_1 = gs.custom_grid_search_mse(model=model_1, param_grid=param_grid_1, X= X_train,
                                                                        X_interaction=X_interaction_train , y=y_train)

            print('model 1 final training')
            #best_estimator_1._reset_parameters()
            estimated_w_1, estimated_V_1 = best_estimator_1.fit(X_train = X_train, X_train_interaction = X_interaction_train,
                                y_train = y_train,
                                    learning_rate=learning_rate, num_epochs=num_epochs, ES_threshold=es_threshold)

            y_test = y_test.detach().cpu().numpy()

            prediction_1 = best_estimator_1.predict(X_test, X_interaction_test, w=estimated_w_1, V=estimated_V_1).detach().cpu().numpy()
            R2_1 = r2_score(y_test, prediction_1)
            mse_1 = mean_squared_error(y_test, prediction_1)
            results_R2_1.append(R2_1)
            results_MSE_1.append(np.sqrt(mse_1))
            print(results_R2_1, round(np.mean(results_R2_1),3))


            print(f'best hp for {i}th experiment for model 1: {best_hyperparameters_1}', file=results_text)
            print(f'best hp for {i}th experiment for model 1: {best_hyperparameters_1}')

            del best_estimator_1, best_hyperparameters_1, model_1, estimated_w_1, estimated_V_1, estimated_w, estimated_V, prediction_1
        
        #############################################################
        #############################################################


        if model_2_train:
            print('model 2 hp tuning')
            
            estimated_w, estimated_V = weight_generation(p, mask_flattened)

            model_2 = LinearRegression_LIT_LVM(d=d, estimated_w=estimated_w, estimated_V=estimated_V, interaction = True, regularization=True,
                                        ls_penalty=False, ls_penalty_type = ls_penalty_type)
            model_2 = model_2.to(device)

            best_estimator_2, best_hyperparameters_2 = gs.custom_grid_search_mse(model=model_2, param_grid=param_grid_2, X= X_train,
                                                                        X_interaction=X_interaction_train , y=y_train)

            print('model 2 final training')
            #best_estimator_2._reset_parameters()
            estimated_w_2, estimated_V_2 = best_estimator_2.fit(X_train = X_train, X_train_interaction = X_interaction_train,
                                y_train = y_train,
                                    learning_rate=learning_rate, num_epochs=num_epochs, ES_threshold=es_threshold)

            try:
                y_test = y_test.detach().cpu().numpy()
            except:
                pass

            prediction_2 = best_estimator_2.predict(X_test, X_interaction_test, w=estimated_w_2, V=estimated_V_2).detach().cpu().numpy()
            R2_2 = r2_score(y_test, prediction_2)
            mse_2 = mean_squared_error(y_test, prediction_2)
            results_R2_2.append(R2_2)
            results_MSE_2.append(mse_2)

            print(results_R2_2, round(np.mean(results_R2_2),3))


            print(f'best hp for {i}th experiment for model 2: {best_hyperparameters_2}', file=results_text)
            print(f'best hp for {i}th experiment for model 2: {best_hyperparameters_2}')

            del best_estimator_2, best_hyperparameters_2, model_2, estimated_w_2, estimated_V_2, estimated_w, estimated_V, prediction_2, R2_2
        
        #############################################################
        #############################################################

        print('__________________________________________________')
        gc.collect()

        if model_3_train:
            print('model 3 hp tuning')
            
            estimated_w, estimated_V = weight_generation(p, mask_flattened)

            model_3 = LinearRegression_LIT_LVM(d=d, estimated_w=estimated_w, estimated_V=estimated_V, interaction = False, regularization=True,
                                        ls_penalty=False, ls_penalty_type = ls_penalty_type)
            model_3 = model_3.to(device)

            best_estimator_3, best_hyperparameters_3 = gs.custom_grid_search_mse(model=model_3, param_grid=param_grid_2, X= X_train,
                                                                        X_interaction=X_interaction_train , y=y_train)

            print('model 3 final training')
            #best_estimator_3._reset_parameters()
            estimated_w_3, estimated_V_3 = best_estimator_3.fit(X_train = X_train, X_train_interaction = X_interaction_train,
                                y_train = y_train,
                                    learning_rate=learning_rate, num_epochs=num_epochs, ES_threshold=es_threshold)

            try:
                y_test = y_test.detach().cpu().numpy()
            except:
                pass

            prediction_3 = best_estimator_3.predict(X_test, X_interaction_test, w=estimated_w_3, V=estimated_V_3).detach().cpu().numpy()
            R2_3 = r2_score(y_test, prediction_3)
            mse_3 = mean_squared_error(y_test, prediction_3)
            results_R2_3.append(R2_3)
            results_MSE_3.append(mse_3)

            print(results_R2_3, round(np.mean(results_R2_3),3))


            print(f'best hp for {i}th experiment for model 3: {best_hyperparameters_3}', file=results_text)
            print(f'best hp for {i}th experiment for model 3: {best_hyperparameters_3}')

            del best_estimator_3, best_hyperparameters_3, model_3, estimated_w_3, estimated_V_3, estimated_w, estimated_V, prediction_3, R2_3
        
        #############################################################
        #############################################################
        print('__________________________________________________')
        gc.collect()


    print('___________________________________________', file=results_text)
    if model_1_train:
        ls = [round(x,3) for x in results_R2_1]
        print(f'results R2 1:{ls}, mean={round(np.mean(ls),3)}, se: {standard_error(ls)}', file=results_text)
    if model_2_train:
        ls = [round(x,3) for x in results_R2_2]
        mean =round(np.mean(ls),3)
        print(f'results R2 2:{ls}, mean:{mean}, se: {standard_error(ls)}', file=results_text)
    if model_3_train:
        ls = [round(x,3) for x in results_R2_3]
        mean =round(np.mean(ls),3)
        print(f'results R2 3:{ls}, mean:{mean}, se:{standard_error(ls)}', file=results_text)
    if model_rf_train:
        ls = [round(x,3) for x in results_rf]
        mean =round(np.mean(ls),3)
        se = standard_error(ls)
        print(f'results rf:{ls}, mean:{mean}, se:{se}', file=results_text)


    if model_1_train:
        ls = [round(x,3) for x in results_MSE_1]
        print(f'results MSE 1:{ls}, mean={round(np.mean(ls),3)}, se: {standard_error(ls)}', file=results_text)
    if model_2_train:
        ls = [round(x,3) for x in results_MSE_2]
        mean =round(np.mean(ls),3)
        print(f'results MSE 2:{ls}, mean:{mean}, se: {standard_error(ls)}', file=results_text)
    if model_3_train:
        ls = [round(x,3) for x in results_MSE_3]
        mean =round(np.mean(ls),3)
        print(f'results MSE 3:{ls}, mean:{mean}, se:{standard_error(ls)}', file=results_text)
    if model_rf_train:
        ls = [round(x,3) for x in results_rf]
        mean =round(np.mean(ls),3)
        se = standard_error(ls)
        print(f'results rf:{ls}, mean:{mean}, se:{se}', file=results_text)



# param_grid_1 is being used for lowrank and 2 for just elasticnet

param_grid_1 = {
    'alpha': [0.01,0.1,1],
    'kappa': [0.01,0.1, 1],
    'gamma': [10000000],
    'd': [2],
    'ls_penalty_type' : ['lowRank']
}

param_grid_2 = {
    'alpha': [0.01,0.1,1,10],
    'kappa': [0,0.01,0.1,1],
    'gamma': [0],
}


param_grid_rf = {'n_estimators': [5,10,50,100],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [2,5,10]}


###################################################
interaction=True
num_epochs = 5000
d = 2
folds = 2
num_experiments = 5
es_threshold =10
tol = 0.0001
learning_rate = 0.05
if model_name == 'elasticnet':
    ls_penalty_type = 'None'
    ls_penalty = False
elif model_name == 'litlvmV1' or 'litlvmV2':
    ls_penalty_type = 'lowRank'
    ls_penalty = True
verbose = False

total_size = len(X)
train_size = int(0.5 * total_size)
val_size = int(0.01 * total_size)
test_size = total_size - train_size - val_size
run = True
save = False
#####################################################



today_date = datetime.datetime.now()
date_text = today_date.strftime("%b %d")


if True:

    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, f"{dataset_name}_model:{model_name}_lr:{learning_rate}_threshold:{es_threshold}_tol:{tol}_folds:{folds}.txt")
    results_text = open(file_path, "w")
    print(param_grid_1, file=results_text)
    print('running the experiments!')
    run_experiment(d=d, es_threshold = es_threshold, learning_rate = learning_rate, param_grid_rf = param_grid_rf, param_grid_1 = param_grid_1, param_grid_2=param_grid_2, num_experiments = num_experiments, results_text= results_text)
    results_text.close()
