import os
import sys

from Linear_regression_LSM import Regression
from itertools import product
from sklearn.metrics import roc_auc_score
from datetime import datetime
import os
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from utils import GridSearch
import pandas as pd
import copy
import gc
import datetime
# Suppress all warnings
warnings.filterwarnings("ignore")
manual_seed = 434214
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


print('device:', device)

def svd(estimated_V,p,d):
    estimated_V = estimated_V.view(p,p)
    # Perform SVD
    U, S, V = torch.svd(estimated_V)
    U_d = U[:, :d]
    S_d = torch.diag(S[:d])
    V_d = V[:, :d]
    return (U_d @ S_d @ V_d.t()).view(p*p)

# training and testing
# alpha for w, kappa for V, and gamma for reduced rank penalty
num_samples = 1000

p=70
V_variance = 0.1
d = 2

param_grid_lowRank = {
    'alpha': [0,0.01,0.1,1],
    'kappa': [0],
    'gamma': [0.01,0.1,1]
}





res_2 = []
res_5 = []
res_10 = []
for data_split in range(10):
    print(f'data split {data_split}')
    reg_lowRank = Regression(p=p ,d=d,interaction = True, sparsity = False, regularization=True, ls_penalty=True)

    #print('reg_row rank estimated w device', reg_lowRank.true_w.device)
    #print('reg_row rank estimated V device', reg_lowRank.estimated_V.device)
    #print('reg_row rank true V device', reg_lowRank.true_V.device)
    #print('reg_lowRank sparsity:', reg_lowRank.sparsity)


    # this sends the model parameters to GPU
    reg_lowRank = reg_lowRank.to(device)


    # generating training, validation, and test data sent then ensuring they are on GPU (reg_lowRank.device)
    X_train, X_train_interaction, y_train = reg_lowRank.simulate(num_samples=num_samples)
    X_train, X_train_interaction, y_train = X_train.to(reg_lowRank.device), X_train_interaction.to(reg_lowRank.device), y_train.to(reg_lowRank.device)

    X_test, X_test_interaction,  y_test = reg_lowRank.simulate(num_samples)
    X_test, X_test_interaction,  y_test = X_test.to(reg_lowRank.device), X_test_interaction.to(reg_lowRank.device),  y_test.to(reg_lowRank.device)

    X_val, X_val_interaction, y_val = reg_lowRank.simulate(num_samples)
    X_val, X_val_interaction, y_val = X_val.to(reg_lowRank.device), X_val_interaction.to(reg_lowRank.device), y_val.to(reg_lowRank.device)

    gs = GridSearch(num_folds=2, epochs=5000, eval_metric='mse',ES_threshold=20,learning_rate = 0.1)

    # Concatenate X_train and X_val along the first dimension (assuming they have the same shape except for the first dimension)
    X_combined = torch.cat((X_train, X_val), dim=0)

    # Concatenate X_train_interaction and X_val_interaction along the first dimension
    X_interaction_combined = torch.cat((X_train_interaction, X_val_interaction), dim=0)

    # Concatenate y_train and y_val along the first dimension
    y_combined = torch.cat((y_train, y_val), dim=0)


    print('finding the best hyper parameters for L2 regression')

    model_1 = Regression(p=p ,d=2,interaction = True, sparsity = False, regularization=True, ls_penalty=True)
    model_1 = model_1.to(device)
    model_2 = Regression(p=p ,d=5,interaction = True, sparsity = False, regularization=True, ls_penalty=True)
    model_2 = model_2.to(device)
    model_3 = Regression(p=p ,d=10,interaction = True, sparsity = False, regularization=True, ls_penalty=True)
    model_3 = model_3.to(device)

    models = [model_1, model_2, model_3]
    #performing grid search when we considere gamma(regularization for low-rank structure)

    for model in models:
        print('model_d:', model.d)
        model._reset_parameters()
        print('finding the best hyperparameters for low rank regression')
        best_estimator, best_hyperparameters = gs.custom_grid_search_mse(model=model, param_grid=param_grid_lowRank, X= X_combined,
                                                                X_interaction=X_interaction_combined , y=y_combined)


        ''' I should reset the parameters here again and train both models using the best hyperparameters'''
        best_estimator._reset_parameters()
        #print(f'best_estimatorO_lowRank: alpha={best_estimator_lowRank.alpha}, gamma={best_estimator_lowRank.gamma}')
        #print(f'best_estimatorO_lowRank estimated_w: {best_estimator_lowRank.estimated_w}')
        print('lowRank final training')
        estimated_w, estimated_V = best_estimator.fit(X_train = X_combined, X_train_interaction = X_interaction_combined,
                                y_train = y_combined, X_val=X_val, X_val_interaction=X_val_interaction, y_val=y_val,
                                    learning_rate=0.05, num_epochs=5000, ES_threshold=20)




        try:
          y_test = y_test.detach().cpu().numpy()
        except:
          pass

        ####################################################################################################
####    #####################################################################################################



        #predict using estimated_w, estimated_V using LowRank regression
        # using LowRank Regression
        predict_estimated_w_estimated_V = best_estimator.predict(X_test, X_test_interaction, w=estimated_w, V=estimated_V).detach().cpu().numpy()
        mse_estimated_w_estimated_V = mean_squared_error(y_test, predict_estimated_w_estimated_V)
        print(f'mse_{model.d} {mse_estimated_w_estimated_V}')

        if model.d == 2:
            res_2.append(mse_estimated_w_estimated_V)
        elif model.d == 5:
            res_5.append(mse_estimated_w_estimated_V)
        elif model.d == 10:
            res_10.append(mse_estimated_w_estimated_V)
        gc.collect()


        print('____________________________________________________________________')


print(res_2)
print(res_5)
print(res_10)

print(np.mean(res_2))
print(np.mean(res_5))
print(np.mean(res_10))
