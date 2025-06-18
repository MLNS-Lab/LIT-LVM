


from sim_linreg import LinearRegression_Simulator
from linreg_FM import LinearRegression_FM
from itertools import product
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
import argparse
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


num_samples = 1000


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--p', type=int, default=60)
parser.add_argument('--d', type=int, default=2)
parser.add_argument('--model', type=str, default='litlvm')
parser.add_argument('--V_noise', type=float, default=0.1)

args = parser.parse_args()


p=int(args.p)
d=int(args.d)
model = str(args.model)

print(f'p:{p}, d:{d}, model:{model}')
if model != 'fm':
    raise ValueError("model is not FM")

# this param_grid adds low_rank penalty and then choose the best l2 regularization coefficient
param_grid = {
    'alpha': [0.1,1,10,100],
    'kappa': [0, 0.005, 0.01],
}

learning_rate=0.05
V_noise=float(args.V_noise)
OP = True # old inplementation OP is ZdZd.T
sparsity = False
today_date = datetime.datetime.now()
date_text = today_date.strftime("%b %d")
os.makedirs(folder_name, exist_ok=True)
file_path = os.path.join(folder_name, f"{model}_p:{p}_d:{d}_V_noise:{V_noise}_OP:{OP}_Sparsity:{sparsity}.txt")
results_text = open(file_path, "w")
print(param_grid, file=results_text)

res_mse_2 = []
res_mse_5 = []
res_mse_10 = []


for data_split in range(5):
    print(f'data split: {data_split}, p: {p} original_d: {d} V_noise:{V_noise}')
    SIMULATOR = LinearRegression_Simulator(p=p ,d=d,interaction = True, sparsity = sparsity, regularization=True, ls_penalty=True,V_noise=V_noise)

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
    gs = GridSearch(num_folds=folds, epochs=num_epochs,ES_threshold=es_threshold, learning_rate=0.1)

    # Concatenate X_train and X_val along the first dimension (assuming they have the same shape except for the first dimension)
    X_combined = torch.cat((X_train, X_val), dim=0)

    # Concatenate X_train_interaction and X_val_interaction along the first dimension
    X_interaction_combined = torch.cat((X_train_interaction, X_val_interaction), dim=0)

    # Concatenate y_train and y_val along the first dimension
    y_combined = torch.cat((y_train, y_val), dim=0)



    model_1 = LinearRegression_FM(p=p ,d=2)
    model_1 = model_1.to(device)
    model_2 = LinearRegression_FM(p=p ,d=5)
    model_2 = model_2.to(device)
    model_3 = LinearRegression_FM(p=p ,d=10)
    model_3 = model_3.to(device)

    models = [model_1, model_2, model_3]


    for model in models:
        print(f'model: {model.d}')
        #performing grid search when we considere gamma(regularization for low-rank structure)
        model._reset_parameters()
        mask= model.mask
        best_estimator, best_hyperparameters = gs.custom_grid_search_mse(model=model, param_grid=param_grid, X= X_combined,
                                                                    X_interaction=X_interaction_combined , y=y_combined)



        ''' I should reset the parameters here again and train both models using the best hyperparameters'''
        best_estimator._reset_parameters()
        #print(f'best_estimatorO_lowRank: alpha={best_estimator_lowRank.alpha}, gamma={best_estimator_lowRank.gamma}')
        #print(f'best_estimatorO_lowRank estimated_w: {best_estimator_lowRank.estimated_w}')
        print('litlvm final training')
        estimated_w, estimated_V = best_estimator.fit(X_train = X_combined, X_train_interaction = X_interaction_combined,
                                y_train = y_combined, X_val=X_val, X_val_interaction=X_val_interaction, y_val=y_val,
                                    learning_rate=learning_rate, num_epochs=num_epochs, ES_threshold=es_threshold)


        try:
          y_test = y_test.detach().cpu().numpy()
        except:
          pass


        prediction = best_estimator.predict(X_test, X_test_interaction, w=estimated_w, V=estimated_V).detach().cpu().numpy()
 
        mse = mean_squared_error(y_test, prediction)
        print('FM MSE:', mse)

        if model.d == 2:
            res_mse_2.append(mse)
        elif model.d == 5:
            res_mse_5.append(mse)
        elif model.d == 10:
            res_mse_10.append(mse)


        gc.collect()

    print('____________________________________________________________________')
    print('____________________________________________________________________')


print(f'result_2: {res_mse_2}, mean: {round(np.mean(res_mse_2),3)}', file=results_text)
print(f'result_5: {res_mse_5}, mean: {round(np.mean(res_mse_5),3)}', file=results_text)
print(f'result_10: {res_mse_10}, mean: {round(np.mean(res_mse_10),3)}', file=results_text)


results_text.close()