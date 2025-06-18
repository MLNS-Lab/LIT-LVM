

from itertools import product
from datetime import datetime
import os
import torch
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from dataset import data_loader
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

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
parser.add_argument('--d', default=2, help='d')
parser.add_argument('--note', default=None, help='a custome note being made when staritng the training for future underestanding of the experiment setting')
parser.add_argument('--lr', default=0.05, help='learning rate')



args = parser.parse_args()
dataset_name = args.dataset
model_name =args.model
d = int(args.d)
lr = float(args.lr)
note = args.note

if model_name == None:
    raise ValueError('you should choose the model in the argument')
X,y = data_loader(dataset_name=dataset_name, task='classification')
        

def standard_error(ls):
    return round(np.std(ls, ddof=1) / np.sqrt(len(ls)),3 )


def weight_generation(p, mask_flattened):
        estimated_w = nn.Parameter(torch.randn(p+1, requires_grad=True)) #print(f'shape of X_train: {X_train.shape}, shape of estimated_w: {estimated_w.shape}')
        estimated_V = nn.Parameter(torch.randn(p,p , requires_grad=True).view(-1))
        estimated_V = estimated_V * mask_flattened
        estimated_V = nn.Parameter(estimated_V)
        return estimated_w, estimated_V


def pca_reconstruct(estimated_V_4,p):
    estimated_V_4 = estimated_V_4.view(p,p)

    estimated_V_4_np = estimated_V_4.detach().cpu().numpy()

    pca = PCA(n_components=2)
    z = pca.fit_transform(estimated_V_4_np)
    z_torch = torch.tensor(z)
    reconstructed = z_torch @ z_torch.T
    reconstructed = reconstructed.view(-1)
    reconstructed = reconstructed.to(device) 
    return reconstructed

def run_experiment(d, es_threshold, learning_rate, param_grid_rf, param_grid_1, param_grid_2, num_experiments, verbose, results_text):
    results_rf = [] # result of direct random forest

    results_1 = [] # low rank model with interaction
    results_2 = [] # elasticnet with interaction
    results_3 = [] # elasticnet wo interaction
    results_4 = [] # en with interaction with PCA
    results_xgb = []
    results_tabnet = []
    if model_name in ['litlvmV1', 'litlvmV2', 'litlvmFFv1', 'litlvmFFv2' ]:
        if model_name == 'litlvmV1':
            from logreg_LIT_LVM_V1 import LogisticRegression_LIT_LVM
            print('litlvmV1 chosen')
        elif model_name == 'litlvmV2':
            from logreg_LIT_LVM_V2 import LogisticRegression_LIT_LVM
            print('litlvmV2 chosen')
        elif model_name == 'litlvmFFv2':
            from logreg_LIT_LVM_FF_V2 import LogisticRegression_LIT_LVM
        elif model_name == 'litlvmFFv1':
            from logreg_LIT_LVM_FF_V1 import LogisticRegression_LIT_LVM
        else:
            raise ValueError('the model should be either litlvmV1 or litlvmV2')
        model_1_train = False     #litlvm
        model_2_train = False    # EN with interaction
        model_3_train = False    # EN no interaction
        model_4_train = False
        model_rf_train = True
        model_xgb_train = False
        model_tabnet_train = False
        
    elif model_name == 'elasticnet':
        print('the model is elasticnet')
        from logreg_LIT_LVM_V1 import LogisticRegression_LIT_LVM
        model_1_train = False  # litlvm
        model_2_train = False   # EN with interaction
        model_3_train = False   # EN no interaction
        model_4_train = True   # EN with interaction PCA
        model_rf_train = True # RF

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
        X_interaction = X_tensor[:, combinations[:, 0]] * X_tensor[:, combinations[:, 1]]


        print('train_index')
        print(train_idx[0:5])

        #dividing gpu tensors into train test split
        X_train, X_val, X_test = X_tensor[train_idx], X_tensor[val_idx], X_tensor[test_idx]
        y_train, y_val, y_test = y_tensor[train_idx], y_tensor[val_idx], y_tensor[test_idx]
        X_interaction_train, X_interaction_val, X_interaction_test = X_interaction[train_idx], X_interaction[val_idx], X_interaction[test_idx]

        #dividing cpu matrices into train test splits
        X_train_rf, X_val_rf, X_test_rf = X_rf[train_idx], X_rf[val_idx], X_rf[test_idx]
        y_train_rf, y_val_rf, y_test_rf = y_rf[train_idx], y_rf[val_idx], y_rf[test_idx]




    

        del X_interaction, X_tensor, y_tensor

        cpuDevice = torch.device('cpu')
        mask = torch.triu(torch.ones(vdim, vdim), diagonal=1)
        mask_flattened = mask.view(-1)
        mask_flattened = mask_flattened.to(cpuDevice)


        p = X_train.shape[1]



        


        X_train, X_interaction_train, y_train = X_train.to(device), X_interaction_train.to(device), y_train.to(device)
        X_test, X_interaction_test, y_test = X_test.to(device), X_interaction_test.to(device), y_test.to(device)
        X_val, X_interaction_val, y_val = X_val.to(device), X_interaction_val.to(device), y_val.to(device)



        gs = GridSearch(num_folds=folds, epochs=num_epochs,ES_threshold=es_threshold, batch_size = len(X_train), learning_rate=learning_rate)



        # standardizing the inputs for the LR model. 
        # I do not standardize the inputs for RF
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
            rf = RandomForestClassifier(random_state=42+i)
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=2, scoring='roc_auc')
            grid_search.fit(X_train_rf, y_train_rf)
            best_rf = grid_search.best_estimator_
            #best_params = grid_search.best_params_
            y_prob_test = best_rf.predict_proba(X_test_rf)[:, 1]
            auc_score_rf = roc_auc_score(y_test_rf, y_prob_test)
            results_rf.append(auc_score_rf)
            print(f'[RF] AUC: {auc_score_rf}')
            y_prob_train = best_rf.predict_proba(X_train_rf)[:, 1]
            y_prob_val = best_rf.predict_proba(X_val_rf)[:, 1]
            y_train_rf = (y_prob_train >= 0.5).astype(int)
            y_val_rf = (y_prob_val >= 0.5).astype(int)
            y_test_rf = (y_prob_test >= 0.5).astype(int)

            

            # y_train_rf = torch.tensor(y_train_rf,dtype=torch.float32).to(device)
            # y_val_rf = torch.tensor(y_val_rf,dtype=torch.float32).to(device)
            # y_test_rf = torch.tensor(y_test_rf,dtype=torch.float32).to(device)

            del grid_search, best_rf
        
        if model_xgb_train:
            print('training XGBoost')   
            xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42+i)
            xgb_grid = GridSearchCV(estimator=xgb, param_grid=param_grid_rf, cv=2, scoring='roc_auc')
            xgb_grid.fit(X_train_rf, y_train_rf)
            best_xgb = xgb_grid.best_estimator_
            y_prob_test_xgb = best_xgb.predict_proba(X_test_rf)[:, 1]
            auc_score_xgb = roc_auc_score(y_test_rf, y_prob_test_xgb)
            results_xgb.append(auc_score_xgb)
            print(f'[XGBoost] AUC: {auc_score_xgb}')
            del best_xgb, xgb_grid
        if model_1_train:
            print('model 1 hp tuning')
            
            estimated_w, estimated_V = weight_generation(p, mask_flattened)

            model_1 = LogisticRegression_LIT_LVM(d, estimated_w, estimated_V,V_dim=vdim, interaction = True, sparsity = False, regularization=True,
                                        ls_penalty=True, ls_penalty_type = ls_penalty_type,alpha = 0.5, kappa = 0.5, gamma=0.5, verbose=False)
            model_1 = model_1.to(device)

            best_estimator_1, best_hyperparameters_1 = gs.custom_grid_search_logreg(model=model_1, param_grid=param_grid_1, X= X_train,
                                                                        X_interaction=X_interaction_train , y=y_train, mask_bool = mask_flattened.bool())

            print('model 1 final training')
            best_estimator_1._reset_parameters()
            estimated_w_1, estimated_V_1 = best_estimator_1.fit(X_train = X_train, X_train_interaction = X_interaction_train,
                                y_train = y_train, X_val=X_val, X_val_interaction=X_interaction_val, y_val=y_val,
                                    learning_rate=learning_rate, num_epochs=num_epochs, ES_threshold=es_threshold,batch_size=5000)

            y_test = y_test.detach().cpu().numpy()

            prediction_1 = best_estimator_1.predict_proba(X_test, X_interaction_test, w=estimated_w_1, V=estimated_V_1,
                                                                                mask_bool= mask_flattened.bool()).detach().cpu().numpy()
            auc_1 = roc_auc_score(y_test, prediction_1)
            results_1.append(auc_1)
            print(results_1, round(np.mean(results_1),3))


            print(f'best hp for {i}th experiment for model 1: {best_hyperparameters_1}', file=results_text)
            print(f'best hp for {i}th experiment for model 1: {best_hyperparameters_1}')

            del best_estimator_1, best_hyperparameters_1, model_1, estimated_w_1, estimated_V_1, estimated_w, estimated_V, prediction_1, auc_1
        
        #############################################################
        #############################################################
        
        if model_2_train:
            estimated_w, estimated_V = weight_generation(p, mask_flattened)



            #interaction True, ls_penalty = False
            model_2 = LogisticRegression_LIT_LVM(d, estimated_w, estimated_V,V_dim=vdim, interaction = True, regularization=True, ls_penalty=False)
            model_2 = model_2.to(device)

            print('model 2 hp tuning')
            best_estimator_2, best_hyperparameters_2 = gs.custom_grid_search_logreg(model=model_2, param_grid=param_grid_2, X= X_train,
                                                                        X_interaction=X_interaction_train , y=y_train, mask_bool = mask_flattened.bool())
            print(f'best hp for {i}th experiment for model 2: {best_hyperparameters_2}', file=results_text)
            print(f'best hp for {i}th experiment for model 2: {best_hyperparameters_2}')
            print(f'--------------------------------------------------------------------', file=results_text)
            print('model 2 final training')
            best_estimator_2._reset_parameters()
            estimated_w_2, estimated_V_2 = best_estimator_2.fit(X_train = X_train, X_train_interaction = X_interaction_train,
                                y_train = y_train, X_val=X_val, X_val_interaction=X_interaction_val, y_val=y_val,
                                    learning_rate=learning_rate, num_epochs=num_epochs, ES_threshold=es_threshold,batch_size=5000)

            # y_test was detached before for lowrank prediction
            try:
                y_test = y_test.detach().cpu().numpy()
            except:
                pass

            prediction_2 = best_estimator_2.predict_proba(X_test, X_interaction_test, w=estimated_w_2, V=estimated_V_2,
                                                                            mask_bool = mask_flattened.bool()).detach().cpu().numpy()
            auc_2 = roc_auc_score(y_test, prediction_2)
            results_2.append(auc_2)
            del best_estimator_2, best_hyperparameters_2, model_2, estimated_w_2, estimated_V_2, estimated_w, estimated_V, prediction_2, auc_2

        if model_4_train:
            estimated_w, estimated_V = weight_generation(p, mask_flattened)



            #interaction True, ls_penalty = False
            model_4 = LogisticRegression_LIT_LVM(d, estimated_w, estimated_V,V_dim=vdim, interaction = True, regularization=True, ls_penalty=False)
            model_4 = model_4.to(device)

            print('model 4 hp tuning')
            best_estimator_4, best_hyperparameters_4 = gs.custom_grid_search_logreg(model=model_4, param_grid=param_grid_4, X= X_train,
                                                                        X_interaction=X_interaction_train , y=y_train, mask_bool = mask_flattened.bool())
            print(f'best hp for {i}th experiment for model 4: {best_hyperparameters_4}', file=results_text)
            print(f'best hp for {i}th experiment for model 4: {best_hyperparameters_4}')
            print(f'--------------------------------------------------------------------', file=results_text)
            print('model 4 final training')
            best_estimator_4._reset_parameters()
            estimated_w_4, estimated_V_4 = best_estimator_4.fit(X_train = X_train, X_train_interaction = X_interaction_train,
                                y_train = y_train, X_val=X_val, X_val_interaction=X_interaction_val, y_val=y_val,
                                    learning_rate=learning_rate, num_epochs=num_epochs, ES_threshold=es_threshold,batch_size=5000)

            # y_test was detached before for lowrank prediction
            try:
                y_test = y_test.detach().cpu().numpy()
            except:
                pass
            
            estimated_V_4 = pca_reconstruct(estimated_V_4,p)
            

            prediction_4 = best_estimator_4.predict_proba(X_test, X_interaction_test, w=estimated_w_4, V=estimated_V_4,
                                                                            mask_bool = mask_flattened.bool()).detach().cpu().numpy()
            auc_4 = roc_auc_score(y_test, prediction_4)
            results_4.append(auc_4)
            del best_estimator_4, best_hyperparameters_4, model_4, estimated_w_4, estimated_V_4, estimated_w, estimated_V, prediction_4, auc_4
        ##########################################################
        ##########################################################
        if model_3_train:
            estimated_w, estimated_V = weight_generation(p, mask_flattened)

            model_3 = LogisticRegression_LIT_LVM(d, estimated_w, estimated_V, V_dim=vdim, interaction=False, regularization=True, ls_penalty=False)
            model_3 = model_3.to(device)

            print('model 3 hp tuning')
            #here param grid should be param_grid_2
            best_estimator_3, best_hyperparameters_3 = gs.custom_grid_search_logreg(model=model_3, param_grid=param_grid_2, X= X_train, 
                                                        X_interaction= X_interaction_train, y=y_train, mask_bool=mask_flattened.bool())
            
            print(f'best hp for {i}th experiment for model 3: {best_hyperparameters_3}', file=results_text)
            print(f'best hp for {i}th experiment for model 3: {best_hyperparameters_3}')
            print(f'--------------------------------------------------------------------', file=results_text)
            print('model 3 final training')

            best_estimator_3._reset_parameters()
            estimated_w_3, estimated_V_3 = best_estimator_3.fit(X_train = X_train, X_train_interaction = X_interaction_train,
                                y_train = y_train, X_val=X_val, X_val_interaction=X_interaction_val, y_val=y_val,
                                    learning_rate=learning_rate, num_epochs=num_epochs, ES_threshold=es_threshold,batch_size=5000)
        

            
            try:
                y_test = y_test.detach().cpu().numpy()
            except:
                pass

            prediction_3 = best_estimator_3.predict_proba(X_test, X_interaction_test, w=estimated_w_3, V=estimated_V_3,
                                                                                mask_bool = mask_flattened.bool()).detach().cpu().numpy()
            auc_3 = roc_auc_score(y_test, prediction_3)
            results_3.append(auc_3)
            del best_estimator_3, best_hyperparameters_3, model_3, estimated_w_3, estimated_V_3,estimated_w, estimated_V, prediction_3, auc_3
            
        if model_tabnet_train:
            print("training TabNet")
            # Standardize RF inputs
            scaler_tabnet = StandardScaler()
            X_train_std = scaler_tabnet.fit_transform(X_train_rf)
            X_val_std = scaler_tabnet.transform(X_val_rf)
            X_test_std = scaler_tabnet.transform(X_test_rf)

            tabnet = TabNetClassifier(
                n_d=16, n_a=16, n_steps=4, gamma=1.3,
                n_independent=2, n_shared=2, seed=9000 + i,
                verbose=0, device_name=device.type)
            
            tabnet.fit(X_train_std, y_train_rf,
                    eval_set=[(X_val_std, y_val_rf)],
                    eval_metric=['auc'], max_epochs=200, patience=20,
                    batch_size=512, virtual_batch_size=128)
            
            y_prob_test_tabnet = tabnet.predict_proba(X_test_std)[:, 1]
            auc_score_tabnet = roc_auc_score(y_test_rf, y_prob_test_tabnet)
            results_tabnet.append(auc_score_tabnet)
            print(f"[TabNet] AUC: {auc_score_tabnet}")
            del tabnet

        ###############################################################################
        ###############################################################################


        print('__________________________________________________')
        gc.collect()
        print('___________________________________________', file=results_text)
        if model_1_train:
            ls = [round(x,3) for x in results_1]
            print(f'results 1:{ls}, mean={round(np.mean(ls),3)}, se: {standard_error(ls)}', file=results_text)
        if model_2_train:
            ls = [round(x,3) for x in results_2]
            mean =round(np.mean(ls),3)
            print(f'results 2:{ls}, mean:{mean}, se: {standard_error(ls)}', file=results_text)
        if model_4_train:
            ls = [round(x,3) for x in results_4]
            mean =round(np.mean(ls),3)
            print(f'results 4:{ls}, mean:{mean}, se: {standard_error(ls)}', file=results_text)
        if model_3_train:
            ls = [round(x,3) for x in results_3]
            mean =round(np.mean(ls),3)
            print(f'results 3:{ls}, mean:{mean}, se:{standard_error(ls)}', file=results_text)
        if model_rf_train:
            ls = [round(x,3) for x in results_rf]
            mean =round(np.mean(ls),3)
            se = standard_error(ls)
            print(f'results rf:{ls}, mean:{mean}, se:{se}', file=results_text)
        
        if model_xgb_train:
            ls = [round(x,3) for x in results_xgb]
            mean = round(np.mean(ls), 3)
            se = standard_error(ls)
            print(f'results xgboost: {ls}, mean: {mean}, se: {se}', file=results_text)

        if model_tabnet_train:
            ls = [round(x,3) for x in results_tabnet]
            mean = round(np.mean(ls), 3)
            se = standard_error(ls)
            print(f'results tabnet: {ls}, mean: {mean}, se: {se}', file=results_text)



# param_grid_1 is being used for lowrank and 2 for just elasticnet

param_grid_1 = {
    'alpha': [0.01, 0.1, 1],
    'kappa': [0.01, 0.1, 1],
    'gamma': [10000000],
    'd': [d],
    'ls_penalty_type' : ['lowRank']
}


param_grid_2 = {
    'alpha': [0.01,0.1,1],
    'kappa': [0.01,0.1,1],
    'gamma': [0.01, 0.1, 1]
}



param_grid_4 = {
    'alpha': [0.01,0.1,1],
    'kappa': [0.01,0.1,1],
    'gamma': [0],
}

param_grid_rf = {'n_estimators': [5,10,50,100,200],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [2,5,10,15]}


#RUNNING CONFIGURATION
interaction=True
num_epochs = 5000
folds = 2
num_experiments = 5
es_threshold =10
tol = 0.0001
learning_rate = lr
if model_name == 'elasticnet':
    ls_penalty_type = 'None'
    ls_penalty = False
elif model_name == 'litlvmV1' or 'litlvmV2' or 'litlvmFFv1' or 'litlvmFFv2':
    ls_penalty_type = 'lowRank'
    ls_penalty = True
verbose = False

total_size = len(X)
train_size = int(0.5 * total_size)
val_size = int(0.01 * total_size)
test_size = total_size - train_size - val_size
#####################################################



today_date = datetime.datetime.now()
date_text = today_date.strftime("%b %d")

os.makedirs(folder_name, exist_ok=True)
file_path = os.path.join(folder_name, f"{dataset_name}_model:{model_name}_lr:{learning_rate}_d:{d}_tol:{tol}_folds:{folds}.txt")
results_text = open(file_path, "w")
if note != None:
    print(note, file=results_text)
print(param_grid_1, file=results_text)
print('running the experiments!')
run_experiment(d=d, es_threshold = es_threshold, learning_rate = learning_rate, param_grid_rf = param_grid_rf, param_grid_1 = param_grid_1, param_grid_2=param_grid_2, num_experiments = num_experiments, verbose = verbose, results_text= results_text)
results_text.close()
