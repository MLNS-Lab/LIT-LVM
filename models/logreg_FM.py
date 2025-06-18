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
import torch.nn.init as init
from torch.utils.data import TensorDataset, DataLoader
from utils import GridSearch
from sklearn.preprocessing import StandardScaler




manual_seed = 43421654
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)




class LogisticRegression_FM(nn.Module):
    def __init__(self,d=2, p = None,
                    alpha = 0.5, kappa = 0.5):

        super(LogisticRegression_FM, self).__init__()

        

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

        
        self.d = d
        # whether to perform calculations with intercept or not

        self.p = p
        
        self.estimated_w  = nn.Parameter(torch.randn(self.p+1, requires_grad=True))
        self.initial_w = copy.deepcopy(self.estimated_w)

        self.mask = torch.triu(torch.ones(self.p, self.p), diagonal=1)
        self.mask_bool = torch.triu(torch.ones(self.p, self.p), diagonal=1).bool()



        self.alpha = alpha
        self.kappa = kappa
        
        self.estimated_Zd = nn.Parameter(torch.randn(self.p,d, requires_grad=True))
        self.initial_Zd = copy.deepcopy(self.estimated_Zd)
        



        # this is being used to capture the losses in each epoch
        self.train_losses = []
        self.val_losses = []



        self.criterion = nn.BCELoss()



    def _V_calc(self):
        #return torch.matmul(self.estimated_Zd, torch.t(self.estimated_Zr)).view(self.estimated_Zd.shape[0]*self.estimated_Zr.shape[0])
        return torch.matmul(self.estimated_Zd, torch.t(self.estimated_Zd)).view(self.estimated_Zd.shape[0]*self.estimated_Zd.shape[0])



    def _reset_parameters(self):
        self.estimated_w.data = self.initial_w.data.clone()
        self.estimated_Zd.data = self.initial_Zd.data.clone()

    def _l2_penalty(self):
        estimated_V = self._V_calc()

        return torch.sum(self.estimated_w ** 2) + torch.sum((estimated_V[self.mask_bool.view(-1)])**2)

        
    def _l1_penalty(self):
        estimated_V = self._V_calc()
        return torch.sum(torch.abs(self.estimated_w)) + torch.sum(torch.abs(estimated_V[self.mask_bool.view(-1)] ))



        

    
    def _BCEL(self, X, X_interaction, y):
        # this class predicts the probability of each instance to belong to a class!
        estimated_V = self._V_calc()
        y_pred = self.predict_proba(X, X_interaction, self.estimated_w, estimated_V, self.mask_bool.view(-1))    
        bcel = self.criterion(y_pred, y)
        return bcel




    def _loss(self,X,X_interction, y):

        return self._BCEL(X,X_interction, y) + self.alpha * self._l2_penalty()



    def fit(self, X_train, X_train_interaction, y_train,X_val = None, X_val_interaction=None, y_val=None, learning_rate=0.001, num_epochs=500, batch_size=128, ES_threshold = 20):
        #self.to(self.device) # moving model parameters to cuda
        self.mask = self.mask.to(self.device)
        self.mask_bool = self.mask_bool.to(self.device)


        # Move training data to GPU
        X_train, X_train_interaction, y_train = X_train.to(self.device), X_train_interaction.to(self.device), y_train.to(self.device)


        #optimizer = optim.AdamW([self.estimated_w, self.estimated_Zd, self.estimated_Zr], lr=learning_rate)
        optimizer = optim.AdamW([self.estimated_w, self.estimated_Zd], lr=learning_rate)

        tolerance = 1e-4  # Define the tolerance for convergence
        patience_threshold = 10  # Number of epochs to wait after convergence
        patience_counter = 0
        best_loss = float('inf')  # Initialize best_loss with infinity
        self.train_losses = []  # Ensure train_losses is initialized

        for epoch in range(num_epochs):
            #self.like.append(-1*self._negative_log_likelihood(X,y))

            #self._data_collection() #only when the interaction is considered!
            optimizer.zero_grad()
            train_loss = self._loss(X_train, X_train_interaction, y_train)
            self.train_losses.append(train_loss.item()+ self._l1_penalty())  # Convert tensor to scalar
            train_loss.backward()
            optimizer.step()

            if self.kappa != 0:
                with torch.no_grad():
                    # Apply proximal operator directly to the weights (excluding the bias)
                    self.estimated_w[1:].copy_(
                        torch.sign(self.estimated_w[1:]) * torch.clamp(
                            torch.abs(self.estimated_w[1:]) - learning_rate * self.kappa, min=0.0
                        )
                    )

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

        

    # if (epoch + 1) % 1000 == 0:
    #     print(f"Epoch {epoch + 1}")
        estimated_V = self._V_calc()
        return self.estimated_w, estimated_V
    
    
    def predict_proba(self, X, X_interaction, w, V, mask_bool):
        # Mask V with mask_bool
        V_masked = V[mask_bool]  # Assuming V is structured as (features, interactions), and mask_bool applies to interaction dimension

        # Mask X_interaction with mask_bool
        X_interaction_masked = X_interaction[:, mask_bool]

        # Concatenating the weights and the feature matrices to perform the matrix multiplication
        X_concat = torch.cat((X, X_interaction_masked), dim=1)
        # Concatenating masked V with w (excluding the bias w[0]) for the linear combination
        wV_concat = torch.cat((w[1:], V_masked))

        # Computing the linear combination
        linear_comb = torch.matmul(X_concat, wV_concat) + w[0]


        predicted_probability = torch.sigmoid(linear_comb)
        return predicted_probability

    def predict_binary(self, X, X_interaction , w, V):

        prediction_prob = self.predict_proba(X, X_interaction, w, V)
        prediction_binary = (prediction_prob >= 0.5).float()


        return prediction_binary
    
