import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import copy
import os
import torch.nn.init as init
from torch.utils.data import TensorDataset, DataLoader
from utils import GridSearch




manual_seed = 43421654
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)




class LinearRegression_LIT_LVM(nn.Module):
    def __init__(self,d, estimated_w, estimated_V, interaction = True,  regularization=True,
                    ls_penalty=True, ls_penalty_type = 'lowRank',
                    alpha = 0.5, kappa = 0.5, gamma=0.5, verbose=True):
        super(LinearRegression_LIT_LVM, self).__init__()

        

        # Check if TPU is available
        if "TPU_NAME" in os.environ:
            self.device = torch.device("xla")  # XLA is PyTorch's TPU device
    
        else:
            # Check if GPU is available
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                
            else:
                self.device = torch.device("cpu")
                

        #p represents the original dimensionality of the data
        self.estimated_w = estimated_w
        self.initial_w = copy.deepcopy(self.estimated_w)
        
        self.estimated_V = estimated_V
        self.initial_V = copy.deepcopy(self.estimated_V)

        self.verbose = verbose
        self.ls_penalty_type = ls_penalty_type
        self.d = d

        self.interaction  = interaction

        self.ls_penalty = ls_penalty
        self.regularization = regularization

        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa        



        self.estimated_alpha_0 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.initial_alpha_0 = copy.deepcopy(self.estimated_alpha_0)

        self.train_losses = []
        self.val_losses = []


    def _reset_parameters(self):
        # Reset estimated_w to its initial value
        
        self.estimated_w.data = self.initial_w.data.clone()

        # Reset estimated_V to its initial value
        if self.interaction:
            self.estimated_V.data = self.initial_V.data.clone()

        # self.estimated_Zd.data = self.initial_Zd.data.clone()
        # # Reset U to its initial value
        # if self.ls_penalty_type == 'latent_distance':
        #     self.estimated_alpha_0.data = self.initial_alpha_0.data.clone()
        #     #self.estimated_alpha_1.data = self.initial_alpha_1.data.clone()
        #     self.estimated_Zd.data = self.initial_Zd.data.clone()

    def _l2_penalty(self):
        if self.interaction:
            return torch.sum(self.estimated_w ** 2) + torch.sum((self.estimated_V[self.mask_bool.view(-1)])**2)
        else:
            return torch.sum(self.estimated_w**2)
        
    def _l1_penalty(self):
        if self.interaction:
            return torch.sum(torch.abs(self.estimated_w)) + torch.sum(torch.abs(self.estimated_V[self.mask_bool.view(-1)]))
        else:
            return torch.sum(torch.abs(self.estimated_w))

    
    def _lsm_penalty(self):
        # this is the difference between estimated interaction weights and UU.T
        if self.ls_penalty_type == 'latent_distance':
            if not self.interaction:
                raise ValueError('reduced ranked pernalty is only used when the interaction terms are incorporated')
            Zd_expanded = self.estimated_Zd.unsqueeze(1) # (p,1,d)
            Z2_expanded = self.estimated_Zd.unsqueeze(0) # (1,p,d)  

            compatibility = self.estimated_alpha_0 - torch.norm(Zd_expanded - Z2_expanded, dim=2,p=2)**2
            compatibility = compatibility.view(-1)

            diff = self.estimated_V[self.mask_bool.view(-1)] - compatibility[self.mask_bool.view(-1)]
        
            penalty = torch.sum(diff**2)
            
        if self.ls_penalty_type == 'lowRank':
            if not self.interaction:
                raise ValueError('reduced ranked pernalty is only used when the interaction terms are incorporated')
            else:
                #Za: (pd, d) . (pr,d).T ---> (pd, pr) ----> flatten ----> diff = Zda_Zra_T_flattened - self.estimated_Va
                C_flattened = torch.matmul(self.estimated_Zd, torch.t(self.estimated_Zd)).view(self.estimated_Zd.shape[0]*self.estimated_Zd.shape[0])
                diff = self.estimated_V[self.mask_bool.view(-1)] - C_flattened[self.mask_bool.view(-1)]
                penalty = torch.sum(diff**2)
        return penalty


    def _mse(self, X, X_interaction, y):
        X_interaction_masked = X_interaction[:, self.mask.view(-1).bool()]
        prediction = self.predict(X, X_interaction_masked, self.estimated_w, self.estimated_V[self.mask.view(-1).bool()])
        squared_error = (prediction - y)**2
        mse = 0.5 * torch.sum(squared_error)
        return mse
    

    def predict(self, X, X_interaction, w, V):

        if self.interaction:
            
            X_concat = torch.cat((X, X_interaction), dim=1)
            wV_concat = torch.cat((w[1:], V))
            prediction = torch.matmul(X_concat, wV_concat)+w[0]


        if not self.interaction:
            prediction = torch.matmul(X, w[1:]) + w[0]
            

        return prediction



    def _loss(self,X,X_interction, y):
        if not self.regularization and not self.interaction and not self.ls_penalty:
            return self._mse(X,X_interction,y)
        
        elif not self.regularization and not self.interaction and self.ls_penalty:
            raise ValueError('reduced rank regression works only when there is interaction')
        
        elif not self.regularization and self.interaction and not self.ls_penalty:
            return self._mse(X,X_interction,y)
        
        elif not self.regularization and self.interaction and self.ls_penalty:
            return self._mse(X,X_interction,y) + self.gamma * self._lsm_penalty()
        
        elif  self.regularization and not self.interaction and not self.ls_penalty:
            return self._mse(X,X_interction,y) + self.alpha * self._l2_penalty() 
        
        elif self.regularization and not self.interaction and self.ls_penalty:
            raise ValueError('reduced rank regression works only when there is interaction')
        
        elif self.regularization and self.interaction and not self.ls_penalty:
            return self._mse(X,X_interction,y) + self.alpha * self._l2_penalty()
        
        elif self.regularization and self.interaction and self.ls_penalty:
            #print(f'mse: {self._mse(X,X_interction, y)},l2_penalty: {self.alpha * self._l2_penalty()},l1_penalty:  {self.kappa * self._l1_penalty()}, lsm_penalty: {self.gamma * self._lsm_penalty()}  ')
            return self._mse(X,X_interction, y) + self.alpha * self._l2_penalty() + self.gamma * self._lsm_penalty() 
        


    def fit(self, X_train, X_train_interaction, y_train,X_val = None, X_val_interaction=None, y_val=None, learning_rate=0.001, num_epochs=500, ES_threshold = 20):

        #self.to(self.device) # moving model parameters to cuda
        self.p = X_train.shape[1]
        self.mask = torch.triu(torch.ones(self.p, self.p), diagonal=1).to(self.device)
        self.mask_bool = torch.triu(torch.ones(self.p, self.p), diagonal=1).bool().to(self.device)

        self.mask = self.mask.to(self.device)
        self.mask_bool = self.mask_bool.to(self.device)
        self.estimated_Zd = nn.Parameter(torch.randn(self.p,self.d, requires_grad=True))
        self.estimated_Zd = torch.nn.Parameter(self.estimated_Zd.to(self.device))
        self.initial_Zd = copy.deepcopy(self.estimated_Zd)

        if  not self.interaction and not self.ls_penalty: 
            optimizer = optim.Adam([self.estimated_w], lr=learning_rate)
        elif self.interaction and not self.ls_penalty:
            optimizer = optim.Adam([self.estimated_w, self.estimated_V], lr=learning_rate)
        elif self.interaction and self.ls_penalty: 
            optimizer = optim.Adam([self.estimated_w, self.estimated_V, self.estimated_Zd], lr=learning_rate)
        else:
            raise ValueError('Choose a correct combination of self.interaction and self.ls_penalty')



        # Move training data to GPU
        X_train, X_train_interaction, y_train = X_train.to(self.device), X_train_interaction.to(self.device), y_train.to(self.device)
        tolerance = 1e-4  # Define the tolerance for convergence
        patience_threshold = 10  # Number of epochs to wait after convergence
        patience_counter = 0
        best_loss = float('inf')  # Initialize best_loss with infinity
        self.train_losses = []  
        patience_counter = 0
        for epoch in range(num_epochs):
            #self.like.append(-1*self._negative_log_likelihood(X,y))
        
            optimizer.zero_grad()
            train_loss = self._loss(X_train, X_train_interaction, y_train)
            self.train_losses.append(train_loss.item()+ self._l1_penalty())  # Convert tensor to scalar
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
                    # Bias term self.estimated_w[0] is not modified

                    if self.interaction:
                        # Apply proximal operator to estimated_V (if applicable)
                        self.estimated_V.copy_(
                            torch.sign(self.estimated_V) * torch.clamp(
                                torch.abs(self.estimated_V) - learning_rate * self.kappa, min=0.0
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
                
        return self.estimated_w, self.estimated_V
    