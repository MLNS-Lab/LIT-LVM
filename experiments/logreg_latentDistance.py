

from logreg_LSM import LogisticRegression
from itertools import product
from sklearn.metrics import roc_auc_score
from datetime import datetime
import os
import torch
import numpy as np
import warnings
from utils import GridSearch
import pandas as pd
import copy
import gc
import datetime
from datetime import datetime
# Suppress all warnings
warnings.filterwarnings("ignore")
#manual_seed = 434214
manual_seed = 454214
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)



#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Check if TPU is available
if "TPU_NAME" in os.environ:
    device = torch.device("xla")  # XLA is PyTorch's TPU device

else:
    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

param_grid_lowRank = {
    'alpha': [0.01,0.1,1],
    'kappa': [0,0.01],
    'gamma': [0.01,0.1,1],
}

param_grid_l1l2 = {
    'alpha': [0.01,0.1,1],
    'kappa': [0,0.01],
    'gamma': [0]
}


def run_experiments(num_samples, p,d, num_experiments, penalty = 'lowRank', sparsity = False, V_noise = 0.000001, learning_rate=0.00001):
    '''this takes p and d and then run the experiment

    '''
    res_2 = []
    res_5 = []
    res_10 = []
    res_l1l2 = []
    for data_split in range(num_experiments):
        print(f'data split: {data_split}')
        print(f'p: {p}, d: {d}')
        reg_lowRank = LogisticRegression(p=p ,d=d,interaction = True, sparsity = sparsity, sparsity_strengh= 0.0000000001, regularization=True, ls_penalty=True, V_noise=V_noise, lsm_penalty_type= penalty)
        # this sends the model parameters to GPU
        reg_lowRank = reg_lowRank.to(device)

        # generating training, validation, and test data sent then ensuring they are on GPU (reg_lowRank.device)
        X_train, X_train_interaction, y_train = reg_lowRank.simulate(num_samples=num_samples)
        X_train, X_train_interaction, y_train = X_train.to(reg_lowRank.device), X_train_interaction.to(reg_lowRank.device), y_train.to(reg_lowRank.device)

        X_test, X_test_interaction,  y_test = reg_lowRank.simulate(num_samples)
        X_test, X_test_interaction,  y_test = X_test.to(reg_lowRank.device), X_test_interaction.to(reg_lowRank.device),  y_test.to(reg_lowRank.device)

        X_val, X_val_interaction, y_val = reg_lowRank.simulate(num_samples)
        X_val, X_val_interaction, y_val = X_val.to(reg_lowRank.device), X_val_interaction.to(reg_lowRank.device), y_val.to(reg_lowRank.device)

        gs = GridSearch(num_folds=2, epochs=5000,ES_threshold=20)

        # Concatenate X_train and X_val along the first dimension (assuming they have the same shape except for the first dimension)
        X_combined = torch.cat((X_train, X_val), dim=0)

        # Concatenate X_train_interaction and X_val_interaction along the first dimension
        X_interaction_combined = torch.cat((X_train_interaction, X_val_interaction), dim=0)

        # Concatenate y_train and y_val along the first dimension
        y_combined = torch.cat((y_train, y_val), dim=0)

        
        logreg_l1l2 =  LogisticRegression(p=p ,d=1,interaction = True, sparsity =sparsity, regularization=True, ls_penalty=False,V_noise=V_noise, lsm_penalty_type=penalty)
        logreg_l1l2 = logreg_l1l2.to(device)
        model_1 = LogisticRegression(p=p ,d=2,interaction = True, sparsity = sparsity, regularization=True, ls_penalty=True,V_noise=V_noise, lsm_penalty_type=penalty)
        model_1 = model_1.to(device)
        model_2 = LogisticRegression(p=p ,d=5,interaction = True, sparsity = sparsity, regularization=True, ls_penalty=True,V_noise=V_noise, lsm_penalty_type=penalty)
        model_2 = model_2.to(device)
        model_3 = LogisticRegression(p=p ,d=10,interaction = True, sparsity = sparsity, regularization=True, ls_penalty=True,V_noise=V_noise, lsm_penalty_type=penalty)
        model_3 = model_3.to(device)

        

        models = [model_1, model_2, model_3, logreg_l1l2]


        for model in models:
            print(f'model: {model.d}')
            #performing grid search when we considere gamma(regularization for low-rank structure)
            model._reset_parameters()
            mask= model.mask

            if model.d == 1:
                # d=1 is only specified for l1l2 model not the low rank model
                best_estimator, best_hyperparameters = gs.custom_grid_search_logreg(model=model, param_grid=param_grid_l1l2, X= X_combined,
                                                                        X_interaction=X_interaction_combined , y=y_combined,mask_bool = (mask.view(-1)).bool())
            else:
                best_estimator, best_hyperparameters = gs.custom_grid_search_logreg(model=model, param_grid=param_grid_lowRank, X= X_combined,
                                                                        X_interaction=X_interaction_combined , y=y_combined,mask_bool = (mask.view(-1)).bool())

            ''' I should reset the parameters here again and train both models using the best hyperparameters'''
            best_estimator._reset_parameters()
            #print(f'best_estimatorO_lowRank: alpha={best_estimator_lowRank.alpha}, gamma={best_estimator_lowRank.gamma}')
            #print(f'best_estimatorO_lowRank estimated_w: {best_estimator_lowRank.estimated_w}')
            print('lowRank final training')
            estimated_w, estimated_V = best_estimator.fit(X_train = X_combined, X_train_interaction = X_interaction_combined,
                                    y_train = y_combined, X_val=X_val, X_val_interaction=X_val_interaction, y_val=y_val,
                                        learning_rate=learning_rate, num_epochs=5000, ES_threshold=20)
            
            
            try:
              y_test = y_test.detach().cpu().numpy()
            except:
              pass



            #predict using estimated_w, estimated_V using LowRank regression
            # using LowRank Regression
            predict_estimated_w_estimated_V = best_estimator.predict_proba(X_test, X_test_interaction, w=estimated_w, V=estimated_V,mask_bool=(mask.view(-1)).bool()).detach().cpu().numpy()
            auc_estimated_w_estimated_V = roc_auc_score(y_test, predict_estimated_w_estimated_V)
            print('low rank predicion:', auc_estimated_w_estimated_V)

            if model.d == 2:
                res_2.append(auc_estimated_w_estimated_V)
            elif model.d == 5:
                res_5.append(auc_estimated_w_estimated_V)
            elif model.d == 10:
                res_10.append(auc_estimated_w_estimated_V)
            elif model.d == 1:
                res_l1l2.append(auc_estimated_w_estimated_V)

            del best_estimator, best_hyperparameters, model
            gc.collect()
            print('____________________________________________________________________')

    return res_2, res_5, res_10, res_l1l2


print('running the code')
penalty = 'latent_distance'
sparsity = True
V_noise = 0.00000001

today_date = datetime.now()
date_text = today_date.strftime("%B %d, %Y %H:%M")
print(date_text)
ps=[10,20,30]
ds = [2,5,10]   # these are true values of d
number_experiment=10
learning_rate=0.05
print(f'number of experiments: {number_experiment}, penalty: {penalty}, sparsity:{sparsity}')

for p in ps:
    for d in ds:
        print(f'p: {p}, true d: {d}')  
        res_2, res_5, res_10, res_l1l2 = run_experiments(num_samples=1000, p=p,d=d, num_experiments=number_experiment, sparsity=sparsity, penalty=penalty, V_noise=V_noise, learning_rate=learning_rate)

        results.write(f'p: {p}, d:{d}\n')
        results.write('------------------------------------------------------------------------\n')
        results.write(f'{str(res_2)},\n')
        results.write(f'{str(res_5)},\n')
        results.write(f'{str(res_10)},\n')
        results.write(f'{str(res_l1l2)},\n')
        results.write('_________________________________________________________________________\n')


results.close()


