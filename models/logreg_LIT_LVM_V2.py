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




class LogisticRegression_LIT_LVM(nn.Module):
    def __init__(self,d, estimated_w, estimated_V, V_dim = None, interaction = True, sparsity = False, regularization=True,
                   ls_penalty=True, ls_penalty_type = 'latent_distance',
                    alpha = 0.5, kappa = 0.5, gamma=0.5, verbose=False):
        '''
        p: dimension of X
        d: dimension of latent space
        interaction: to incorporate interaction of features in training or not
        regularization: whether we want regularization or not
        alpha: This is the regularization parameter for the regression coecients
        kappa: This the regularization for L1 penalty
        gamma: This is the regularization parameter for reduced ranked penalty
        ls_penalty: whether you want to add latent space penalty
        V_noise: noise added to the true weights of interaction matrix (V). This is the variance
        sparsity: if True, it makes the true weights sparse
        
        '''
        super(LogisticRegression_LIT_LVM, self).__init__()

        

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
        print(self.ls_penalty_type)
        self.d = d
        # whether to perform calculations with intercept or not

        self.p = V_dim
        
        self.mask = torch.triu(torch.ones(self.p, self.p), diagonal=1)
        self.mask_bool = torch.triu(torch.ones(self.p, self.p), diagonal=1).bool()

        # whether to incorporate interaction in the analysis or not
        self.interaction  = interaction


        #RRR = True --> adding reduced randked regression to the penalty
        self.ls_penalty = ls_penalty

        #whether we want to do regularization or not
        self.regularization = regularization


        # alpha is regularization coefficient for vector of weights (w: individual features)
        self.alpha = alpha

        self.sparsity = sparsity


        # gamma is the reduced rank regularization coefficient
        self.gamma = gamma

        #kappa: L1 regularization coefficient
        self.kappa = kappa
        
        self.estimated_Zd = nn.Parameter(torch.randn(self.p,d, requires_grad=True))
        self.initial_Zd = copy.deepcopy(self.estimated_Zd)
        
        self.estimated_Zr = nn.Parameter(torch.randn(self.p,d, requires_grad=True))
        self.initial_Zr = copy.deepcopy(self.estimated_Zr)

        if self.ls_penalty_type == 'latent_distance':
            self.estimated_alpha_0 = nn.Parameter(torch.randn(1, requires_grad=True))
            self.initial_alpha_0 = copy.deepcopy(self.estimated_alpha_0)



# Wrap it with nn.Parameter


        #self.like is a list that captures the log-liklihood of data during the training phase
        self.like = []
        #Vtrue_Vestimate means norm of difference of matrix of true weights and matrix of estimated weights at each epoch
        self.Vtrue_Vestimate = []
        #Vtrue_UUT means the norm of difference between true weight matrix and U@U.T at each epochf
        self.Vtrue_UUT = []
        #Vestimate_UUT means the norm of difference between matrix of estimated weights and matrix of U@U.T at each epoch
        self.Vestimate_UUT = []
        # this is being used to capture the losses in each epoch
        self.train_losses = []
        self.val_losses = []



        self.criterion = nn.BCELoss()



    def _reset_parameters(self):
        # Reset estimated_w to its initial value
        
        self.estimated_w.data = self.initial_w.data.clone()

        # Reset estimated_V to its initial value
        if self.interaction:
            self.estimated_V.data = self.initial_V.data.clone()

        self.estimated_Zd.data = self.initial_Zd.data.clone()
        self.estimated_Zr.data = self.initial_Zr.data.clone()
        # Reset U to its initial value
        if self.ls_penalty_type == 'latent_distance':
            self.estimated_alpha_0.data = self.initial_alpha_0.data.clone()


    def _l2_penalty(self):
        if self.interaction:
            return torch.sum(self.estimated_w ** 2) + torch.sum((self.estimated_V[self.mask_bool.view(-1)])**2)
        else:
            return torch.sum(self.estimated_w**2)
        
    def _l1_penalty(self):
        if self.interaction:
            return torch.sum(torch.abs(self.estimated_w)) + torch.sum(torch.abs( self.estimated_V[self.mask_bool.view(-1)] ))
        else:
            return torch.sum(torch.abs(self.estimated_w))

    
    def _lsm_penalty(self):
        # this is the difference between estimated interaction weights and UU.T
        if self.ls_penalty_type == 'latent_distance':
            if not self.interaction:
                raise ValueError('reduced ranked pernalty is only used when the interaction terms are incorporated')
            Zd_expanded = self.estimated_Zd.unsqueeze(1) # (p,1,d)
            Zr_expanded = self.estimated_Zr.unsqueeze(0) # (1,p,d) 

            compatibility = self.estimated_alpha_0 - torch.norm(Zd_expanded - Zr_expanded, dim=2,p=2)**2
            compatibility = compatibility.view(-1)

            diff = self.estimated_V[self.mask_bool.view(-1)] - compatibility[self.mask_bool.view(-1)]
        
            penalty = torch.sum(diff**2)
            
        if self.ls_penalty_type == 'lowRank':
            if not self.interaction:
                raise ValueError('reduced ranked pernalty is only used when the interaction terms are incorporated')
            else:
                C_flattened = torch.matmul(self.estimated_Zd, torch.t(self.estimated_Zr)).view(self.estimated_Zd.shape[0]*self.estimated_Zr.shape[0])
                diff = self.estimated_V[self.mask_bool.view(-1)] - C_flattened[self.mask_bool.view(-1)]
                penalty = torch.sum(diff**2)
        return penalty


    
    def _BCEL(self, X, X_interaction, y):
        # this class predicts the probability of each instance to belong to a class!
        y_pred = self.predict_proba(X, X_interaction, self.estimated_w, self.estimated_V, self.mask_bool.view(-1))    
        bcel = self.criterion(y_pred, y)
        return bcel




    def _loss(self,X,X_interction, y):
        if not self.regularization and not self.interaction and not self.ls_penalty:
            return self._BCEL(X,X_interction,y)
        
        elif not self.regularization and not self.interaction and self.ls_penalty:
            raise ValueError('reduced rank regression works only when there is interaction')
        
        elif not self.regularization and self.interaction and not self.ls_penalty:
            return self._BCEL(X,X_interction,y)
        
        elif not self.regularization and self.interaction and self.ls_penalty:
            return self._BCEL(X,X_interction,y) + self.gamma * self._lsm_penalty()
        
        elif  self.regularization and not self.interaction and not self.ls_penalty:
            return self._BCEL(X,X_interction,y) + self.alpha * self._l2_penalty() 
        
        elif self.regularization and not self.interaction and self.ls_penalty:
            raise ValueError('reduced rank regression works only when there is interaction')
        
        elif self.regularization and self.interaction and not self.ls_penalty:
            return self._BCEL(X,X_interction,y) + self.alpha * self._l2_penalty() 
        
        elif self.regularization and self.interaction and self.ls_penalty:
            return self._BCEL(X,X_interction, y) + self.alpha * self._l2_penalty() + self.gamma * self._lsm_penalty() 





    def fit(self, X_train, X_train_interaction, y_train,X_val = None, X_val_interaction=None, y_val=None, learning_rate=0.001, num_epochs=500, batch_size=128, ES_threshold = 20):
        #print(f'fitting the model using {self.ls_penalty_type}')
        #self.to(self.device) # moving model parameters to cuda
        self.mask = self.mask.to(self.device)
        self.mask_bool = self.mask_bool.to(self.device)


        # Move training data to GPU
        if self.interaction:
            X_train, X_train_interaction, y_train = X_train.to(self.device), X_train_interaction.to(self.device), y_train.to(self.device)
        else:
            X_train, y_train = X_train.to(self.device), y_train.to(self.device)


        if  not self.interaction and not self.ls_penalty: 
            optimizer = optim.AdamW([self.estimated_w], lr=learning_rate)
        elif self.interaction and not self.ls_penalty:
            optimizer = optim.AdamW([self.estimated_w, self.estimated_V], lr=learning_rate)
        elif self.interaction and self.ls_penalty: 
            #print('hello')
            if self.ls_penalty_type == 'lowRank':
                #print('lowRank penlaty optimization')
                optimizer = optim.AdamW([self.estimated_w, self.estimated_V, self.estimated_Zd, self.estimated_Zr], lr=learning_rate)
            elif self.ls_penalty_type == 'latent_distance':
                #print('latent space penlaty optimization')
                optimizer = optim.AdamW([self.estimated_w, self.estimated_V, self.estimated_alpha_0, self.estimated_Zd,self.estimated_Zr], lr=learning_rate)
        else:
            raise ValueError('Choose a correct combination of self.interaction and self.ls_penalty')


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
    
    
    def predict_proba(self, X, X_interaction, w, V, mask_bool):
        if self.interaction:
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

        else:
            linear_comb = torch.matmul(X, w[1:]) + w[0]

        predicted_probability = torch.sigmoid(linear_comb)
        return predicted_probability

    def predict_binary(self, X, X_interaction , w, V):

        prediction_prob = self.predict_proba(X, X_interaction, w, V)
        prediction_binary = (prediction_prob >= 0.5).float()


        return prediction_binary
    
