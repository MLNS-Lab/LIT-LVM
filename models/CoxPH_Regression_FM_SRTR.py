import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import copy
import os
import pandas as pd

manual_seed = 21123334
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)



class CoxPH_FM(nn.Module):

    def __init__(self, d=2, estimated_w = None, 
                Va_shape=None, Vb_shape=None, Vdr_shape=None, alpha=0.5, kappa=0.5):
        super(CoxPH_FM, self).__init__()
        

        # Check if TPU is available
        if "TPU_NAME" in os.environ:
            self.device = torch.device("xla")  # XLA is PyTorch's TPU device
    
        else:
            # Check if GPU is available
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                
            else:
                self.device = torch.device("cpu")

        self.d = d

        

        #p represents the original dimensionality of the data
        self.estimated_w = estimated_w
        self.initial_w = self.estimated_w.clone()


        #these are used for restarting the parameters of the model
        self.initial_w = copy.deepcopy(self.estimated_w)


        self.estimated_Za_don = nn.Parameter(torch.randn(Va_shape[0],self.d, requires_grad=True))
        self.estimated_Za_rec = nn.Parameter(torch.randn(Va_shape[1],self.d, requires_grad=True))
        self.initial_Za_don = copy.deepcopy(self.estimated_Za_don)
        self.initial_Za_rec = copy.deepcopy(self.estimated_Za_rec)



        self.estimated_Zb_don = nn.Parameter(torch.randn(Vb_shape[0],self.d, requires_grad=True))
        self.estimated_Zb_rec = nn.Parameter(torch.randn(Vb_shape[1],self.d, requires_grad=True))
        self.initial_Zb_don = copy.deepcopy(self.estimated_Zb_don)
        self.initial_Zb_rec = copy.deepcopy(self.estimated_Zb_rec)


        self.estimated_Zdr_don = nn.Parameter(torch.randn(Vdr_shape[0],self.d, requires_grad=True))
        self.estimated_Zdr_rec = nn.Parameter(torch.randn(Vdr_shape[1],self.d, requires_grad=True))

        self.initial_Zdr_don = copy.deepcopy(self.estimated_Zdr_don)
        self.initial_Zdr_rec = copy.deepcopy(self.estimated_Zdr_rec)

        #ls_penaly = True --> adding reduced randked regression to the penalty


        self.alpha = alpha

        self.kappa = kappa
        self.train_losses = []

        self.val_losses = []


    
    def _reset_parameters(self):
        # Reset estimated_w to its initial value

        self.estimated_w.data = self.initial_w.data.clone()
            

        # Reset U to its initial value
        self.estimated_Za_don.data = self.initial_Za_don.data.clone()
        self.estimated_Za_rec.data = self.initial_Za_rec.data.clone()

        self.estimated_Zb_don.data = self.initial_Zb_don.data.clone()
        self.estimated_Zb_rec.data = self.initial_Zb_rec.data.clone()

        self.estimated_Zdr_don.data = self.initial_Zdr_don.data.clone()
        self.estimated_Zdr_rec.data = self.initial_Zdr_rec.data.clone()


    def predict(self, X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            estimated_w, estimated_Va, estimated_Vb, estimated_Vdr):
        '''This Calculates exp(W'X)'''
        #h = W'X
        hazard = self._hazard(X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            estimated_w, estimated_Va, estimated_Vb, estimated_Vdr)
        return torch.exp(hazard) #exp(h)
        
    def _hazard(self, X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            estimated_w, estimated_Va, estimated_Vb, estimated_Vdr):
        
        X_concat = torch.cat((X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train), dim=1)
        #some of the elements of estimated_Va is zero (if it's not getting optimized) and I remove those elemenst so that the shape of matrix and the length of the weight for matrix multiplication matches
        squeezed_estimated_Va = torch.squeeze(estimated_Va)  # squeeze to align the size of estimated_Va and self.hla_a_pairs_mask_flattened_bool
        squeezed_estimated_Vb = torch.squeeze(estimated_Vb)
        squeezed_estimated_Vdr = torch.squeeze(estimated_Vdr)
        wV_concat = torch.cat((estimated_w, squeezed_estimated_Va[self.hla_a_pairs_mask_flattened_bool], squeezed_estimated_Vb[self.hla_b_pairs_mask_flattened_bool], squeezed_estimated_Vdr[self.hla_dr_pairs_mask_flattened_bool]))
        h = torch.matmul(X_concat, wV_concat)
        return h

    def _l2_penalty(self):
        estimated_Va, estimated_Vb, estimated_Vdr = self._V_calc()
        return torch.sum(self.estimated_w**2) + torch.sum((estimated_Va[self.hla_a_pairs_mask_flattened_bool])**2) + torch.sum((estimated_Vb[self.hla_b_pairs_mask_flattened_bool])**2) + torch.sum((estimated_Vdr[self.hla_dr_pairs_mask_flattened_bool])**2)

        
    def _l1_penalty(self):
        estimated_Va, estimated_Vb, estimated_Vdr = self._V_calc()
        return torch.sum(torch.abs(self.estimated_w)) + torch.sum(torch.abs((estimated_Va[self.hla_a_pairs_mask_flattened_bool]))) + torch.sum(torch.abs((estimated_Vb[self.hla_b_pairs_mask_flattened_bool]))) + torch.sum(torch.abs((estimated_Vdr[self.hla_dr_pairs_mask_flattened_bool])))

    def _V_calc(self):
        estimated_Va =  torch.matmul(self.estimated_Za_don, torch.t(self.estimated_Za_rec)).view(self.estimated_Za_don.shape[0]*self.estimated_Za_rec.shape[0])
        estimated_Vb = torch.matmul(self.estimated_Zb_don, torch.t(self.estimated_Zb_rec)).view(self.estimated_Zb_don.shape[0]*self.estimated_Zb_rec.shape[0])
        estimated_Vdr =  torch.matmul(self.estimated_Zdr_don, torch.t(self.estimated_Zdr_rec)).view(self.estimated_Zdr_don.shape[0]*self.estimated_Zdr_rec.shape[0])
        return estimated_Va, estimated_Vb, estimated_Vdr 



    def _loss(self,X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            survival_censoring_time_train, events_train):
            return self._negative_partial_log_liklihood(X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            survival_censoring_time_train, events_train) + self.alpha * self._l2_penalty()


    def _negative_partial_log_liklihood(self, X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            survival_censoring_time_train, events_train):
        estimated_Va, estimated_Vb, estimated_Vdr = self._V_calc()
        #X.w + X_interaction.v --> (n,p).(p,1) + (n,p^2).(p^2,1) ---> (n,1)   
        h = self._hazard(X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            self.estimated_w, estimated_Va, estimated_Vb, estimated_Vdr)
        neg_log_partial_likelihood = 0.0
        # we sort all the tensors based on the order of survival or censoring time
        _, indices = torch.sort(survival_censoring_time_train)
        # sorting events based on the order of survival or censoring time
        events_sorted = events_train[indices]
        # sorting h based on the order of survival or censoring time
        h_sorted = h[indices]
        #print('h sorted', h_sorted)
        h_sorted_reversed = torch.flip(h_sorted, dims=[0])
        logcumsumexp = torch.logcumsumexp(h_sorted_reversed,dim=0)
        logcumsumexp_reversed = torch.flip(logcumsumexp,dims=[0])
        risk_set_sizes = torch.sum(events_sorted == 1, dim=0)
        neg_log_partial_likelihood = -1 * torch.sum(  (h_sorted - logcumsumexp_reversed)[events_sorted] )/risk_set_sizes
        return neg_log_partial_likelihood
    


    def fit(self, X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            survival_censoring_time_train, events_train,
            X_val, hla_a_pairs_val, hla_b_pairs_val, hla_dr_pairs_val, 
            survival_censoring_time_val, events_val, 
            hla_a_pairs_mask, hla_b_pairs_mask, hla_dr_pairs_mask,
            learning_rate=0.1, num_epochs=5000, ES_threshold = 20):
        '''X_train: data_basic + MM + hla types'''

        # flattening the mask matrices for easier computation 
        #print('flattening the mask matrices')
        self.hla_a_pairs_mask_flattened = hla_a_pairs_mask.reshape(-1)
        self.hla_a_pairs_mask_flattened_bool = self.hla_a_pairs_mask_flattened.bool()
        #print(f'shape of hla_a_pairs_mask: {hla_a_pairs_mask.shape}, shape of flattened: {self.hla_a_pairs_mask_flattened.shape}')
        self.hla_b_pairs_mask_flattened = hla_b_pairs_mask.reshape(-1)
        self.hla_b_pairs_mask_flattened_bool = self.hla_b_pairs_mask_flattened.bool()
        #print(f'shape of hla_b_pairs_mask: {hla_b_pairs_mask.shape}, shape of flattened: {self.hla_b_pairs_mask_flattened.shape}')
        self.hla_dr_pairs_mask_flattened = hla_dr_pairs_mask.reshape(-1)
        self.hla_dr_pairs_mask_flattened_bool = self.hla_dr_pairs_mask_flattened.bool()
        #print(f'shape of hla_dr_pairs_mask: {hla_dr_pairs_mask.shape}, shape of flattened: {self.hla_dr_pairs_mask_flattened.shape}')

        optimizer = optim.Adam([self.estimated_w, self.estimated_Za_don, self.estimated_Za_rec,self.estimated_Zb_don,self.estimated_Zb_rec, self.estimated_Zdr_don,self.estimated_Zdr_rec], lr=learning_rate)

        
        tolerance = 1e-4  # Define the tolerance for convergence
        patience_threshold = 10  # Number of epochs to wait after convergence
        patience_counter = 0
        best_loss = float('inf')  # Initialize best_loss with infinity
        self.train_losses = []  # Ensure train_losses is initialized

        for epoch in range(num_epochs):
            #self.like.append(-1*self._negative_log_likelihood(X,y))
        
            #self._data_collection() #only when the interaction is considered!
            optimizer.zero_grad()
            train_loss = self._loss(X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            survival_censoring_time_train, events_train)
            self.train_losses.append(train_loss.item() + self._l1_penalty())  # Convert tensor to scalar
            train_loss.backward()
            optimizer.step()


            if self.kappa != 0:
                with torch.no_grad():
                    # Apply proximal operator to weights (excluding bias at index 0)
                    self.estimated_w[1:].copy_(
                        torch.sign(self.estimated_w[1:]) * torch.clamp(
                            torch.abs(self.estimated_w[1:]) - learning_rate * self.kappa, min=0.0
                        )
                    )
            # Early stopping based on training loss convergence
            if epoch > 0:
                loss_change = abs(self.train_losses[-2] - self.train_losses[-1])
                if loss_change < tolerance:
                    patience_counter += 1
                else:
                    patience_counter = 0  # Reset if loss change is significant
            else:
                patience_counter = 0  # Initialize patience_counter

            # Early stopping condition
            if patience_counter >= patience_threshold:
                #print(f"Training converged at epoch {epoch + 1}")
                break  # Exit the training loop 
        estimated_Va, estimated_Vb, estimated_Vdr = self._V_calc()

        return self.estimated_w, estimated_Va, estimated_Vb, estimated_Vdr
