import os
os.chdir('/home/mxn447/codes/neurips2024')
import sys
sys.path.append('/home/mxn447/codes/neurips2024')


import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

from datetime import datetime
import torch
import numpy as np
import warnings
import pandas as pd
import gc
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
from torch.utils.data import TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
from CoxPH_Regression_LSM_SRTR import CoxPH_LSM
import os
import matplotlib.pyplot as plt
import seaborn as sns



# Suppress all warnings
warnings.filterwarnings("ignore")
#manual_seed = 43421654
manual_seed = 45
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)

# Check if TPU is available
if "TPU_NAME" in os.environ:
    device = torch.device("xla")  # XLA is PyTorch's TPU device
else:
    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")






import torch
import matplotlib.pyplot as plt


def scatter_plot(weight, distance, xlabel="Weights", ylabel="Distances", save_path=None):
    # Creating the scatter plot with smaller dot size for better visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(weight, distance, c='blue', alpha=0.5, s=5)  # Adjust 's' for smaller dot size
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Save the plot to a file if save_path is provided
    if save_path:
        plt.savefig(save_path, format='pdf')
        print(f"Plot saved as {save_path}")
    
    # Show the plot
    plt.show()





















def create_empty_df(columns_example):

    donors_ordered = []
    recipients_ordered = []

    for col in columns_example:
        don_part, rec_part = col.split('_REC_')
        if don_part not in donors_ordered:
            donors_ordered.append(don_part)
        if rec_part not in recipients_ordered:
            recipients_ordered.append(rec_part)

# Create an empty DataFrame with ordered donors as rows and ordered recipients as columns
    df_empty_ordered = pd.DataFrame(index=donors_ordered, columns=recipients_ordered)

    return df_empty_ordered

if True:
    os.chdir('/home/mxn447/Datasets/SRTR')
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

    a_pairs_matrix = create_empty_df(list(mask_a_pairs_flattened.columns))
    b_pairs_matrix = create_empty_df(list(mask_b_pairs_flattened.columns))
    dr_pairs_matrix = create_empty_df(list(mask_dr_pairs_flattened.columns))

    a_pair_columns = list(mask_a_pairs_flattened.columns)
    b_pair_columns = list(mask_b_pairs_flattened.columns)
    dr_pair_columns = list(mask_dr_pairs_flattened.columns)


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


parser = argparse.ArgumentParser(description='Process some data.')
parser.add_argument('--gamma', default=0.01, help='gamma value')
parser.add_argument('--ep1', default=5000, help='epoch1')
parser.add_argument('--alpha', default=0.01, help='alpha')



args = parser.parse_args()
gm = float(args.gamma)
alpha = float(args.alpha)

d = 2
interaction=True
num_epochs_2 = 5000
num_epochs_1 = int(args.ep1)

folds = 2
num_experiments = 10
es_threshold =10
lr_2 = 0.05
lr_1 = 0.05


gamma_list = [gm]





total_size = len(dataset)
train_size = int(0.9 * total_size)
val_size = int(0.05* total_size)
test_size = total_size - train_size - val_size
#####################################################
today_date = datetime.now()
date_text = today_date.strftime("%b %d")

today_date = datetime.now()
date_text = today_date.strftime("%b %d")

folder_name = f"/home/mxn447/codes/neurips2024/experiments/results/plots/cph/{date_text}"
os.makedirs(folder_name, exist_ok=True)

file_path = f'{folder_name}/MM_weights_{gm}.txt'
results_text = open(file_path, "w")


for i in range(num_experiments):

    if i == 1:
        break
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



    cph_2 = CoxPH_LSM(d=2, estimated_w = estimated_w, estimated_Va=estimated_Va, estimated_Vb=estimated_Vb, estimated_Vdr=estimated_Vdr,
                  Va_shape=Va_shape, Vb_shape=Vb_shape, Vdr_shape=Vdr_shape,
            interaction=interaction,  ls_penalty=False, alpha = alpha, kappa=0)

    # this sends the model parameters to GPU
    cph_2 = cph_2.to(device)

    X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train, events_train, survival_censoring_time_train = X_train.to(cph_2.device), hla_a_pairs_train.to(cph_2.device), hla_b_pairs_train.to(cph_2.device), hla_dr_pairs_train.to(cph_2.device), events_train.to(cph_2.device), survival_censoring_time_train.to(cph_2.device)
    X_val, hla_a_pairs_val, hla_b_pairs_val, hla_dr_pairs_val, events_val, survival_censoring_time_val = X_val.to(cph_2.device), hla_a_pairs_val.to(cph_2.device), hla_b_pairs_val.to(cph_2.device), hla_dr_pairs_val.to(cph_2.device), events_val.to(cph_2.device), survival_censoring_time_val.to(cph_2.device)
    X_test, hla_a_pairs_test, hla_b_pairs_test, hla_dr_pairs_test, events_test, survival_censoring_time_test = X_test.to(cph_2.device),  hla_a_pairs_test.to(cph_2.device), hla_b_pairs_test.to(cph_2.device), hla_dr_pairs_test.to(cph_2.device), events_test.to(cph_2.device), survival_censoring_time_test.to(cph_2.device)

    # Concatenate X_train and X_val along the first dimension (assuming they have the same shape except for the first dimension)
    X_combined = torch.cat((X_train, X_val), dim=0)


    hla_a_pairs_combined = torch.cat((hla_a_pairs_train, hla_a_pairs_val), dim=0)
    hla_b_pairs_combined = torch.cat((hla_b_pairs_train, hla_b_pairs_val), dim=0)
    hla_dr_pairs_combined = torch.cat((hla_dr_pairs_train, hla_dr_pairs_val), dim=0)
    # Concatenate events_train, events_val
    events_combined = torch.cat((events_train, events_val), dim=0)

    # concatenate survival censoring time
    survival_censoring_time_combined = torch.cat((survival_censoring_time_train, survival_censoring_time_val), dim=0)

    print('fitting elasticnet model started')
    estimated_w_2, estimated_Va_2, estimated_Vb_2, estimated_Vdr_2 = cph_2.fit(X_train=X_combined,
                    hla_a_pairs_train=hla_a_pairs_combined, hla_b_pairs_train=hla_b_pairs_combined, hla_dr_pairs_train=hla_dr_pairs_combined,
                    survival_censoring_time_train=survival_censoring_time_combined, events_train=events_combined,
                    X_val=X_val , hla_a_pairs_val=hla_a_pairs_val, hla_b_pairs_val=hla_b_pairs_val, hla_dr_pairs_val=hla_dr_pairs_val,
                    survival_censoring_time_val=survival_censoring_time_val, events_val=events_val,
                    hla_a_pairs_mask=hla_a_pairs_mask, hla_b_pairs_mask=hla_b_pairs_mask, hla_dr_pairs_mask=hla_dr_pairs_mask,
                    learning_rate=lr_2, num_epochs=num_epochs_2, ES_threshold = 20)


    estimated_w_2_copy = copy.deepcopy(estimated_w_2)
    estimated_Va_2_copy = copy.deepcopy(estimated_Va_2)
    estimated_Vb_2_copy = copy.deepcopy(estimated_Vb_2)
    estimated_Vdr_2_copy = copy.deepcopy(estimated_Vdr_2)
    print('fitting elasticnet model finished')

    # survival_censoring_time_test = survival_censoring_time_test.detach().cpu().numpy()
    # events_test= events_test.detach().cpu().numpy()

    Zd_a_2 = cph_2.estimated_Za_don
    Zd_b_2 = cph_2.estimated_Zb_don
    Zd_dr_2 = cph_2.estimated_Zdr_don

    Zr_a_2 = cph_2.estimated_Za_rec
    Zr_b_2 = cph_2.estimated_Zb_rec
    Zr_dr_2 = cph_2.estimated_Zdr_rec


    Va_2 = estimated_Va_2
    Va_2 = Va_2.view(1,-1)
    Va_2 = Va_2[mask_a_pairs_flattened.bool()] 
    Vb_2 = estimated_Vb_2
    Vb_2 = Vb_2.view(1,-1)
    Vb_2 = Vb_2[mask_b_pairs_flattened.bool()] 
    Vdr_2 = estimated_Vdr_2
    Vdr_2 = Vdr_2.view(1,-1)
    Vdr_2 = Vdr_2[mask_dr_pairs_flattened.bool()] 

    Va_2 = (Va_2.detach().cpu().numpy()).tolist()
    Vb_2 = (Vb_2.detach().cpu().numpy()).tolist()
    Vdr_2 = (Vdr_2.detach().cpu().numpy()).tolist()

    

    for gamma in gamma_list:

        print(f'training LD with gamma: {gamma} started ')
        cph_1 = CoxPH_LSM(d=2, estimated_w = estimated_w_2_copy, estimated_Va=estimated_Va_2_copy, estimated_Vb=estimated_Vb_2_copy, estimated_Vdr=estimated_Vdr_2_copy,
              Va_shape=Va_shape, Vb_shape=Vb_shape, Vdr_shape=Vdr_shape,
        interaction=interaction,  ls_penalty=True, ls_penalty_type='latent_distance', alpha = alpha, kappa=0, gamma = gamma)

        # this sends the model parameters to GPU
        cph_1 = cph_1.to(device)
            
        estimated_w_1, estimated_Va_1, estimated_Vb_1, estimated_Vdr_1 = cph_1.fit(X_train=X_combined,
                        hla_a_pairs_train=hla_a_pairs_combined, hla_b_pairs_train=hla_b_pairs_combined, hla_dr_pairs_train=hla_dr_pairs_combined,
                        survival_censoring_time_train=survival_censoring_time_combined, events_train=events_combined,
                        X_val=X_val , hla_a_pairs_val=hla_a_pairs_val, hla_b_pairs_val=hla_b_pairs_val, hla_dr_pairs_val=hla_dr_pairs_val,
                        survival_censoring_time_val=survival_censoring_time_val, events_val=events_val,
                        hla_a_pairs_mask=hla_a_pairs_mask, hla_b_pairs_mask=hla_b_pairs_mask, hla_dr_pairs_mask=hla_dr_pairs_mask,
                        learning_rate=lr_1, num_epochs=num_epochs_1, ES_threshold = 20)
    
        estimated_w_2_copy = copy.deepcopy(estimated_w_2)
        estimated_Va_2_copy = copy.deepcopy(estimated_Va_2)
        estimated_Vb_2_copy = copy.deepcopy(estimated_Vb_2)
        estimated_Vdr_2_copy = copy.deepcopy(estimated_Vdr_2)


        Zd_a_1 =cph_1.estimated_Za_don
        Zd_b_1 =cph_1.estimated_Zb_don
        Zd_dr_1 =cph_1.estimated_Zdr_don

        Zr_a_1 =cph_1.estimated_Za_rec
        Zr_b_1 =cph_1.estimated_Zb_rec
        Zr_dr_1 =cph_1.estimated_Zdr_rec


          # these are the weight calculated based on the latent space representaiton
        if True:
          Va_Z = cph_1._compatibility(cph_1.estimated_alpha_0_a,Zd_a_1, Zr_a_1)
          Va_Z = Va_Z.view(1,-1)
          Va_Z = Va_Z[mask_a_pairs_flattened.bool()]
          Va_Z = (Va_Z.detach().cpu().numpy()).tolist()

          Vb_Z = cph_1._compatibility(cph_1.estimated_alpha_0_b, Zd_b_1, Zr_b_1)
          Vb_Z = Vb_Z.view(1,-1)
          Vb_Z = Vb_Z[mask_b_pairs_flattened.bool()] 
          Vb_Z = (Vb_Z.detach().cpu().numpy()).tolist()


          Vdr_Z = cph_1._compatibility(cph_1.estimated_alpha_0_dr,Zd_dr_1, Zr_dr_1)
          Vdr_Z = Vdr_Z.view(1,-1)
          Vdr_Z = Vdr_Z[mask_dr_pairs_flattened.bool()] 
          Vdr_Z = (Vdr_Z.detach().cpu().numpy()).tolist()
        if True:
          pass

        # extracting the names for donors and recipients. Index includes donors categories and column includes recipient categorie
        a_don_name = list(a_pairs_matrix.index)
        a_rec_name = list(a_pairs_matrix.columns)
        b_don_name = list(b_pairs_matrix.index)
        b_rec_name = list(b_pairs_matrix.columns)
        dr_don_name = list(dr_pairs_matrix.index)
        dr_rec_name = list(dr_pairs_matrix.columns)



        # extracting the estimated weights of  interaction terms after optimization for A, B, DR
        Va_1 = estimated_Va_1
        Va_1 = Va_1.view(1,-1)
        Va_1 = Va_1[mask_a_pairs_flattened.bool()] 
        Vb_1 = estimated_Vb_1
        Vb_1 = Vb_1.view(1,-1)
        Vb_1 = Vb_1[mask_b_pairs_flattened.bool()] 
        Vdr_1 = estimated_Vdr_1
        Vdr_1 = Vdr_1.view(1,-1)
        Vdr_1 = Vdr_1[mask_dr_pairs_flattened.bool()] 

        Va_1 = (Va_1.detach().cpu().numpy()).tolist()
        Vb_1 = (Vb_1.detach().cpu().numpy()).tolist()
        Vdr_1 = (Vdr_1.detach().cpu().numpy()).tolist()




        filename = f'{folder_name}/Scatter_A_1_split:{i}_gamma:{gamma}.pdf'
        scatter_plot(Va_1, Va_Z, xlabel="Weights", ylabel="Distances", save_path=filename)
        filename = f'{folder_name}/Scatter_B_1_split:{i}_gamma:{gamma}.pdf'
        scatter_plot(Vb_1, Vb_Z, xlabel="Weights", ylabel="Distances", save_path=filename)
        filename = f'{folder_name}/Scatter_DR_1_split:{i}_gamma:{gamma}.pdf'
        scatter_plot(Vdr_1, Vdr_Z, xlabel="Weights", ylabel="Distances", save_path=filename)

        
        filename = f'{folder_name}/Scatter_A_2_split:{i}_gamma:{gamma}.pdf'
        scatter_plot(Va_2, Va_Z, xlabel="Weights", ylabel="Distances", save_path=filename)
        filename = f'{folder_name}/Scatter_B_2_split:{i}_gamma:{gamma}.pdf'
        scatter_plot(Vb_2, Vb_Z, xlabel="Weights", ylabel="Distances", save_path=filename)
        filename = f'{folder_name}/Scatter_DR_2_split:{i}_gamma:{gamma}.pdf'
        scatter_plot(Vdr_2, Vdr_Z, xlabel="Weights", ylabel="Distances", save_path=filename)


results_text.close()






