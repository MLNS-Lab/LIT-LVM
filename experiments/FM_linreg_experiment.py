import os
import sys

from linreg_FM import LinearRegression_FM
from itertools import product
from datetime import datetime
import os
import torch
import numpy as np
import argparse
from sklearn.metrics import mean_squared_error, r2_score
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# Suppress all warnings
warnings.filterwarnings("ignore")
#manual_seed = 434214
manual_seed = 454214
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
from dataset import data_loader


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




def run_experiment(d, es_threshold, learning_rate, param_grid_1, num_experiments, results_text):

    results_mse_1 = [] # low rank model with interaction
    results_r2_1 = [] # low rank model with interaction
    model_1_train = True

    for i in range(num_experiments):




        print(f'experiment {i}')
        train_idx, test_idx = train_test_split(range(len(X)), test_size=(val_size + test_size), random_state=42+i, shuffle=True)
        val_idx, test_idx = train_test_split(test_idx, test_size=(test_size / (val_size + test_size)), random_state=42+i, shuffle=True)
        X_values = X.values
        X_values = X_values.astype(np.float32)
        X_tensor = torch.tensor(X_values, dtype=torch.float32)
        y_tensor = y.astype(np.float32)
        y_tensor = torch.tensor(y_tensor.values, dtype=torch.float32)
        y_tensor = torch.squeeze(y_tensor)

        vdim= X_tensor.shape[1]
        indices = torch.arange(vdim)
        combinations = torch.cartesian_prod(indices, indices)


        print('train_index')
        print(train_idx[0:5])

        X_train, X_val, X_test = X_tensor[train_idx], X_tensor[val_idx], X_tensor[test_idx]
        y_train, y_val, y_test = y_tensor[train_idx], y_tensor[val_idx], y_tensor[test_idx]


        X_interaction_train = compute_interactions_in_place(X_train, vdim)
        X_interaction_val = X_val[:, combinations[:, 0]] * X_val[:, combinations[:, 1]]
        X_interaction_test = X_test[:, combinations[:, 0]] * X_test[:, combinations[:, 1]]
        #

    

        del  X_tensor, y_tensor

        cpuDevice = torch.device('cpu')
        mask = torch.triu(torch.ones(vdim, vdim), diagonal=1)
        mask_flattened = mask.view(-1)
        mask_flattened = mask_flattened.to(cpuDevice)


        p = X_train.shape[1]

        
        # moving the tensors into the gpu for the LR model
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


        if model_1_train:

            model_1 = LinearRegression_FM(p=p , d=d)
            model_1 = model_1.to(device)

            best_estimator_1, best_hyperparameters_1 = gs.custom_grid_search_mse(model=model_1, param_grid=param_grid_1, X= X_train,
                                                                        X_interaction=X_interaction_train , y=y_train)

            print('model 1 final training')
            best_estimator_1._reset_parameters()
            estimated_w_1, estimated_V_1 = best_estimator_1.fit(X_train = X_train, X_train_interaction = X_interaction_train,
                                y_train = y_train, X_val=X_val, X_val_interaction=X_interaction_val, y_val=y_val,
                                    learning_rate=learning_rate, num_epochs=num_epochs, ES_threshold=es_threshold)

            try:
                y_test = y_test.detach().cpu().numpy()
            except:
                pass


            prediction_1 = best_estimator_1.predict(X_test, X_interaction_test, w=estimated_w_1, V=estimated_V_1).detach().cpu().numpy()
            mse = mean_squared_error(y_test, prediction_1)
            r2 = r2_score(y_test, prediction_1)

            results_mse_1.append(mse)
            results_r2_1.append(r2)

            print(results_r2_1, round(np.mean(results_r2_1),3))
            print(np.sqrt(mse), round(np.mean(np.sqrt(results_mse_1)), 3))
            
            


            print(f'best hp for {i}th experiment for model 1: {best_hyperparameters_1}', file=results_text)
            print(f'best hp for {i}th experiment for model 1: {best_hyperparameters_1}')

            del best_estimator_1, best_hyperparameters_1, model_1, estimated_w_1, estimated_V_1, prediction_1, mse, r2
        

    if model_1_train:
        print(f'results_r2_1: {[round(x, 3) for x in results_r2_1]}, mean= {round(np.mean(results_r2_1), 3)}', file=results_text)
        print(f'results_rmse_1: {[round(x, 3) for x in np.sqrt(results_mse_1)]}, mean= {round(np.mean(np.sqrt(results_mse_1)), 3)}', file=results_text)

        



d = 20
param_grid_1 = {
    'alpha': [0.01,0.1,1,10],
    'kappa': [0.01,0.1,1,10],
    'd': [d],
}




num_epochs = 5000

folds = 2
num_experiments = 5
es_threshold =20
learning_rate = 0.05



total_size = len(X)
train_size = int(0.5 * total_size)
val_size = int(0.01 * total_size)
test_size = total_size - train_size - val_size



today_date = datetime.datetime.now()
date_text = today_date.strftime("%b %d")

os.makedirs(folder_name, exist_ok=True)
file_path = os.path.join(folder_name, f"{dataset_name}_model:{model_name}__d:{d}_lr:{learning_rate}_es:{es_threshold}_folds:{folds}.txt")
results_text = open(file_path, "w")
print('running the experiments!')
run_experiment(d=d, es_threshold = es_threshold, learning_rate = learning_rate, param_grid_1 = param_grid_1, num_experiments = num_experiments, results_text= results_text)
results_text.close()
