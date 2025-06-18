import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import r2_score
import copy
import os




manual_seed = 43421654
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)



class LinearRegression_FM(nn.Module):
    def __init__(self, p=None,d=None, alpha = 0.5, kappa = 0.5):

        super(LinearRegression_FM, self).__init__()

        

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
        self.p = p

        self.d = d


        # defining mask to remove the duplicate columns from the calculations
        # when diagonal = 1, it means that it sets the diagonal to 0 so that the quadratic terms x1x1, x2x2, are removed!
        self.mask = torch.triu(torch.ones(p, p), diagonal=1)
        self.mask_bool = torch.triu(torch.ones(self.p, self.p), diagonal=1).bool()



        # alpha is regularization coefficient for vector of weights (w: individual features)
        self.alpha = alpha

        #kappa: L1 regularization coefficient
        self.kappa = kappa

        self.estimated_w  = nn.Parameter(torch.randn(self.p+1, requires_grad=True))
        self.initial_w = copy.deepcopy(self.estimated_w)


        self.mask = torch.triu(torch.ones(self.p, self.p), diagonal=1)
        self.mask_bool = torch.triu(torch.ones(self.p, self.p), diagonal=1).bool()

        self.alpha = alpha
        self.kappa = kappa

        self.estimated_Zd = nn.Parameter(torch.randn(p,d, requires_grad=True))
        self.initial_Zd = copy.deepcopy(self.estimated_Zd)


        # this is being used to capture the losses in each epoch
        self.train_losses = []
        self.val_losses = []

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


    def _mse(self, X, X_interaction, y):
        estimated_V = self._V_calc()
        X_interaction_masked = X_interaction[:, self.mask.view(-1).bool()]
        prediction = self.predict(X, X_interaction_masked, self.estimated_w, estimated_V[self.mask.view(-1).bool()])
        squared_error = (prediction - y)**2
        mse = 0.5 * torch.sum(squared_error)
        return mse


    def _loss(self,X,X_interction, y):
        return self._mse(X,X_interction,y) + self.alpha * self._l2_penalty()
        

    def fit(self, X_train, X_train_interaction, y_train,X_val = None, X_val_interaction=None, y_val=None, learning_rate=0.001, num_epochs=500, ES_threshold = 20):

        #self.to(self.device) # moving model parameters to cuda
        self.mask = self.mask.to(self.device)


        optimizer = optim.AdamW([self.estimated_w,  self.estimated_Zd], lr=learning_rate)

        # Move training data to GPU
        X_train, X_train_interaction, y_train = X_train.to(self.device), X_train_interaction.to(self.device), y_train.to(self.device)


        tolerance = 1e-4  # Define the tolerance for convergence
        patience_threshold = 10  # Number of epochs to wait after convergence
        patience_counter = 0
        best_loss = float('inf')  # Initialize best_loss with infinity
        self.train_losses = []  # Ensure train_losses is initialized
        for epoch in range(num_epochs):
        
            optimizer.zero_grad()
            train_loss = self._loss(X_train, X_train_interaction, y_train)
            self.train_losses.append(train_loss+ self._l1_penalty())
            train_loss.backward()
            optimizer.step()

            # Apply proximal operator for L1 penalty excluding the bias term
            if self.kappa != 0:
                with torch.no_grad():
                    # Apply proximal operator directly to the weights (excluding the bias)
                    self.estimated_w[1:].copy_(
                        torch.sign(self.estimated_w[1:]) * torch.clamp(
                            torch.abs(self.estimated_w[1:]) - learning_rate * self.kappa, min=0.0
                        )
                    )                                                                                                                     
            # Explicitly release GPU memory
            
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
  
            estimated_V = self._V_calc()
        return self.estimated_w, estimated_V

    def predict(self, X, X_interaction, w, V):
        
        X_concat = torch.cat((X, X_interaction), dim=1)

        wV_concat = torch.cat((w[1:], V))
        prediction = torch.matmul(X_concat, wV_concat)+w[0]
        return prediction
    