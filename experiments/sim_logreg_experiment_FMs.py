import os
os.chdir('/home/mxn447/codes/neurips2024')
import sys
sys.path.append('/home/mxn447/codes/neurips2024')


from sim_logreg import LogisticRegression_Simulator
from logreg_FM import LogisticRegression_FM
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
import torch.nn as nn
# Suppress all warnings
warnings.filterwarnings("ignore")
manual_seed = 434214
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
import argparse

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


num_samples = 1000

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--p', type=int, default=60)
parser.add_argument('--d', type=int, default=2)
parser.add_argument('--model', type=str, default='litlvm')
parser.add_argument('--V_noise', type=str, default='0.01')
parser.add_argument('--sparsity', type=int, default=0)
parser.add_argument('--noise', type=int, default=0)
parser.add_argument('--interaction', type=int, default=0)
parser.add_argument('--sigma', type=float, default=0.0001)



args = parser.parse_args()


p=int(args.p)
d=int(args.d)
model = str(args.model)
V_noise = float(args.V_noise)
sparsity = int(args.sparsity)
noise = int(args.noise)

if sparsity == 0:
    sparsity = False
else:
    sparsity = True
    
if noise == 0:
    noise = False
else:
    noise = True
    
interaction = int(args.interaction)
if interaction == 0:
    interaction = False
else:
    interaction = True
sigma = float(args.sigma)

    
sigma = float(args.sigma)
print(f'p:{p}, d:{d}, model:{model}')
if model != 'fm':
    raise ValueError("model is not FM")



# this param_grid adds low_rank penalty and then choose the best l2 regularization coefficient
if sparsity:
    param_grid = {
        'alpha': [0.01,0.1,1],
        'kappa': [0.01, 0.1, 1,10],
    }
else:
    param_grid = {
        'alpha': [0.01,0.1,1,10],
        'kappa': [0],

    }
learning_rate=0.05

today_date = datetime.datetime.now()
date_text = today_date.strftime("%b %d")
folder_name = f"/home/mxn447/codes/neurips2024/experiments/results/simulations/logistic_regression/{date_text}"
os.makedirs(folder_name, exist_ok=True)
file_path = os.path.join(folder_name, f"{model.upper()}_p:{p}_d:{d}_lr:{learning_rate}_Noise:{noise}_V_noise:{V_noise}_interaction:{interaction}_Sparsity:{sparsity}_sigma:{sigma}.txt")
results_text = open(file_path, "w")
print(param_grid, file=results_text)

res_2 = []
res_2_w_mse = []
res_2_V_mse = []
res_5 = []
res_5_w_mse = []
res_5_V_mse = []
res_10 = []
res_10_w_mse = []
res_10_V_mse = []

for data_split in range(5):
    print(f'data split: {data_split}, p: {p} original_d: {d} V_noise: {V_noise} Sparsity: {sparsity} Noise: {noise}')
    SIMULATOR = LogisticRegression_Simulator(p=p ,d=d,interaction = True, sparsity = sparsity, sigma=sigma, V_noise=V_noise, lsm_penalty_type='lowRank')

    # this sends the model parameters to GPU
    SIMULATOR = SIMULATOR.to(device)


    # generating training, validation, and test data sent then ensuring they are on GPU (reg_lowRank.device)
    X_train, X_train_interaction, y_train = SIMULATOR.simulate(num_samples=num_samples)
    X_train, X_train_interaction, y_train = X_train.to(SIMULATOR.device), X_train_interaction.to(SIMULATOR.device), y_train.to(SIMULATOR.device)

    X_test, X_test_interaction,  y_test = SIMULATOR.simulate(num_samples)
    X_test, X_test_interaction,  y_test = X_test.to(SIMULATOR.device), X_test_interaction.to(SIMULATOR.device),  y_test.to(SIMULATOR.device)

    X_val, X_val_interaction, y_val = SIMULATOR.simulate(num_samples)
    X_val, X_val_interaction, y_val = X_val.to(SIMULATOR.device), X_val_interaction.to(SIMULATOR.device), y_val.to(SIMULATOR.device)

    gs = GridSearch(num_folds=2, epochs=5000,ES_threshold=20)
    folds = 2
    num_epochs = 5000
    es_threshold = 20
    learning_rate = 0.05
    gs = GridSearch(num_folds=folds, epochs=num_epochs,ES_threshold=es_threshold, learning_rate=learning_rate)

    # Concatenate X_train and X_val along the first dimension (assuming they have the same shape except for the first dimension)
    X_combined = torch.cat((X_train, X_val), dim=0)

    # Concatenate X_train_interaction and X_val_interaction along the first dimension
    X_interaction_combined = torch.cat((X_train_interaction, X_val_interaction), dim=0)

    # Concatenate y_train and y_val along the first dimension
    y_combined = torch.cat((y_train, y_val), dim=0)


    model_1 = LogisticRegression_FM(p=p ,d=2)
    model_1 = model_1.to(device)
    model_2 = LogisticRegression_FM(p=p ,d=5)
    model_2 = model_2.to(device)
    model_3 = LogisticRegression_FM(p=p ,d=10)
    model_3 = model_3.to(device)

    models = [model_1, model_2, model_3]


    for model in models:
        print(f'model: {model.d}')
        model._reset_parameters()
        mask= model.mask
        best_estimator, best_hyperparameters = gs.custom_grid_search_logreg_FM(model=model, param_grid=param_grid, X= X_combined,
                                                                    X_interaction=X_interaction_combined , y=y_combined,mask_bool = (mask.view(-1)).bool())




        ''' I should reset the parameters here again and train both models using the best hyperparameters'''
        best_estimator._reset_parameters()

        print('FM final training')
        estimated_w, estimated_V = best_estimator.fit(X_train = X_combined, X_train_interaction = X_interaction_combined,
                                y_train = y_combined, X_val=X_val, X_val_interaction=X_val_interaction, y_val=y_val,
                                    learning_rate=learning_rate, num_epochs=num_epochs, ES_threshold=es_threshold)


        try:
            y_test = y_test.detach().cpu().numpy()
        except:
            pass


        prediction = best_estimator.predict_proba(X_test, X_test_interaction, w=estimated_w, V=estimated_V,mask_bool=(mask.view(-1)).bool()).detach().cpu().numpy()
        auc = roc_auc_score(y_test, prediction)
        print('FM predicion:', auc)
        if interaction:
            w_mse = torch.mean((SIMULATOR.true_w - estimated_w) ** 2)
            V_mse = torch.mean((SIMULATOR.true_V - estimated_V) ** 2)

        if model.d == 2:
            res_2.append(auc)
            res_2_w_mse.append(w_mse.detach().cpu().item())
            res_2_V_mse.append(V_mse.detach().cpu().item())
        elif model.d == 5:
            res_5.append(auc)
            res_5_w_mse.append(w_mse.detach().cpu().item())
            res_5_V_mse.append(V_mse.detach().cpu().item())
        elif model.d == 10:
            res_10.append(auc)
            res_10_w_mse.append(w_mse.detach().cpu().item())
            res_10_V_mse.append(V_mse.detach().cpu().item())

        gc.collect()

    print('____________________________________________________________________')
    print('____________________________________________________________________')




print(f'result_2: {res_2}, mean: {round(np.mean(res_2),3)}, SE: {round(np.std(res_2, ddof=1) / np.sqrt(len(res_2)), 3)}', file=results_text)
print(f'result_2_w_MSE: {res_2_w_mse}, mean: {round(np.mean(res_2_w_mse), 3)}, SE: {round(np.std(res_2_w_mse, ddof=1) / np.sqrt(len(res_2_w_mse)), 3)}', file=results_text)
print(f'result_2_V_MSE: {res_2_V_mse}, mean: {round(np.mean(res_2_V_mse), 3)}, SE: {round(np.std(res_2_V_mse, ddof=1) / np.sqrt(len(res_2_V_mse)), 3)}', file=results_text)
print('----------------------------------', file=results_text)
print(f'result_5: {res_5}, mean: {round(np.mean(res_5),3)}, SE: {round(np.std(res_5, ddof=1) / np.sqrt(len(res_5)), 3)}', file=results_text)
print(f'result_5_w_MSE: {res_5_w_mse}, mean: {round(np.mean(res_5_w_mse), 3)}, SE: {round(np.std(res_5_w_mse, ddof=1) / np.sqrt(len(res_5_w_mse)), 3)}', file=results_text)
print(f'result_5_V_MSE: {res_5_V_mse}, mean: {round(np.mean(res_5_V_mse), 3)}, SE: {round(np.std(res_5_V_mse, ddof=1) / np.sqrt(len(res_5_V_mse)), 3)}', file=results_text)
print('----------------------------------',file=results_text)

print(f'result_10: {res_10}, mean: {round(np.mean(res_10),3)}, SE: {round(np.std(res_10, ddof=1) / np.sqrt(len(res_10)), 3)}', file=results_text)
print(f'result_10_w_MSE: {res_10_w_mse}, mean: {round(np.mean(res_10_w_mse), 3)}, SE: {round(np.std(res_10_w_mse, ddof=1) / np.sqrt(len(res_10_w_mse)), 3)}', file=results_text)
print(f'result_10_V_MSE: {res_10_V_mse}, mean: {round(np.mean(res_10_V_mse), 3)}, SE: {round(np.std(res_10_V_mse, ddof=1) / np.sqrt(len(res_10_V_mse)), 3)}', file=results_text)


results_text.close()
