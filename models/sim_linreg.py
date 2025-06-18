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




manual_seed = 43421654
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)




class LinearRegression_Simulator(nn.Module):
    def __init__(self, p=None,d=None,interaction = True, sparsity = False, regularization=True, ls_penalty=True, alpha = 0.5, kappa = 0.5, gamma=0.5, V_noise=0.1):
        '''
        p: dimension of X
        d: dimension of latent space
        interaction: to incorporate interaction of features in training or not
        regularization: whether we want regularization or not
        alpha: This is the regularization parameter for the regression coefficients
        kappa: This the regularization for L1 penalty
        gamma: This is the regularization parameter for reduced ranked penalty
        ls_penalty: whether you want to add latent space penalty
        V_noise: noise added to the true weights of interaction matrix (V). This is the variance
        sparsity: if True, it makes the true weights sparse
        
        '''
        super(LinearRegression_Simulator, self).__init__()

        

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


        # whether to incorporate interaction in the analysis or not
        self.interaction  = interaction


        #RRR = True --> adding reduced randked regression to the penalty
        self.rrr = ls_penalty

        #whether we want to do regularization or not
        self.regularization = regularization


        # alpha is regularization coefficient for vector of weights (w: individual features)
        self.alpha = alpha

        self.sparsity = sparsity
        self.sparsity_strength = 0.00000000001
        # gamma is the reduced rank regularization coefficient
        self.gamma = gamma

        #kappa: L1 regularization coefficient
        self.kappa = kappa

        # this is the regularization coefficient for the weight matrix (V: interaction terms)
        #
        # self.kappa = kappa


        #this is the initialization weight vector for the model --> for each feature
    
        self.estimated_w = nn.Parameter(torch.randn(p+1, requires_grad=True))


            
        self.initial_w = copy.deepcopy(self.estimated_w)

        #this is the initialiaation weight matrix --> for each interaction term in the model
        if interaction:

            #self.estimated_V = nn.Parameter(torch.randn(p,p, requires_grad=True).view(p*p))
            self.estimated_V = nn.Parameter((torch.randn(p,p, requires_grad=True).view(p*p)) * self.mask.view(-1))

        else:
            self.estimated_V = None

        self.initial_V = copy.deepcopy(self.estimated_V)
        #self.u is meant to be use if we assume that the weight matrix has low-rank structure
        # this is the initialization for U which will be optimized
        self.estimated_Zd = nn.Parameter(torch.randn(p,d, requires_grad=True))

        self.initial_Zd = copy.deepcopy(self.estimated_Zd)


        # this is being used to capture the losses in each epoch
        self.train_losses = []

        self.val_losses = []

    # generating true weight of the model using torch.randn --> drawing the weights from noramal distribution of mean 0 and var 1
    # the reason for generating true weights and matrix is to simulate the data and compare the outcome of optimization with that

        # these are true weights for the model
        self.true_w = torch.randn(self.p+1)
        if self.sparsity:
            self.true_w = self._apply_sparsity(self.true_w,self.sparsity_strength)
        self.true_w = self.true_w.to(self.device)


        if self.interaction:
            
            # self.trueU is the true latent representation of U
            self.trueZd = torch.randn(self.p, self.d)
            self.true_V  = torch.mm(self.trueZd, self.trueZd.t()).view(p*p)

            noise = np.random.normal(0, np.sqrt(V_noise), size=len(self.true_V))
            noise = torch.tensor(noise, dtype = torch.float32)
            self.true_V = self.true_V + noise
            self.true_V = self.true_V * self.mask.view(-1)

            if self.sparsity:
                self.true_V = self._apply_sparsity(self.true_V, self.sparsity_strength)
            self.true_V = self.true_V.to(self.device)


        else:
            self.true_V = None
        
    def _apply_sparsity(self, vector, sigma):
        # Generate random noise with the same shape as the input vector
        noise = 0.001 * torch.randn_like(vector)

        # Calculate the probability of setting each coefficient to 0
        probability = torch.exp(-(vector**2) / sigma)

        # Create a mask based on the generated probability and a random noise
        mask = (noise > probability).float()

        # Apply the mask to the input vector
        sparse_vector = vector * mask


        return sparse_vector

    def simulate(self, num_samples, noise_variance=0.001):
        '''
        we need true weights to calculate the target variables. after the optimizatin, initial weights should be close to the true weights. 
        '''
        
        # specifying the mean and covariance for feature matrix which will be drawn from a multivariate gaussian
        mean = np.zeros(self.p) # mean 0
        cov = np.eye(self.p)  # identity covariance matrix
        X = np.random.multivariate_normal(mean, cov, num_samples) # drawing multivariate normal using mean vector and covariance matrix
        X_tensor = torch.tensor(X, dtype=torch.float32)   # converting numpy to tensor
        X_tensor= X_tensor.to(self.device)
        # Create indices for all combinations of columns

        # if we want to add interaction to our analysis, create an interaction matrix of X_tensor ---> X_tensor_interaction
        if self.interaction:
            indices = torch.arange(self.p)
            combinations = torch.cartesian_prod(indices, indices)

            # Extract the corresponding columns and compute element-wise products
            X_tensor_interaction = X_tensor[:, combinations[:, 0]] * X_tensor[:, combinations[:, 1]]
            X_tensor_interaction = X_tensor_interaction.to(self.device)

        # prediction is the measure in which we calculate the target variable before adding the noise
        # the second part calculates the sum of interaction terms which will be added to the linear combination of weights and feature matrix

        if self.interaction:
            
            #concatenating the weights and the feature matrices to perfomr the matrix multiplication
            X_concat = torch.cat((X_tensor, X_tensor_interaction), dim=1)
            #print('X_concat device:', X_concat.device)
            wV_concat = torch.cat((self.true_w[1:], self.true_V))
            #print('wV_concat device', wV_concat.device)
            prediction = torch.matmul(X_concat, wV_concat)+self.true_w[0]


        if not self.interaction:
            prediction = torch.matmul(X_tensor, self.true_w[1:]) + self.true_w[0]




        # creating noise (epsilon) to add to the prediction. Increasing the variance make the problem harder
        noise = np.random.normal(0, np.sqrt(noise_variance), size=num_samples)
        noise = torch.tensor(noise, dtype = torch.float32) 
        noise = noise.to(self.device)
        y = prediction + noise
        y_tensor = torch.tensor(y, dtype=torch.float32)


        if self.interaction:
            return X_tensor.to(self.device), X_tensor_interaction.to(self.device), y_tensor.to(self.device)
        
        else: 
            return X_tensor.to(self.device), None,  y_tensor.to(self.device)



    def _reset_parameters(self):
        # Reset estimated_w to its initial value
        self.estimated_w.data = self.initial_w.data.clone()

        # Reset estimated_V to its initial value
        if self.interaction:
            self.estimated_V.data = self.initial_V.data.clone()

        # Reset U to its initial value
        self.estimated_Zd.data = self.initial_Zd.data.clone()

    def _l2_penalty(self):
        if self.interaction:
            #return torch.sum(self.estimated_w ** 2) + torch.sum(self.estimated_V**2)
            #return torch.sum(self.estimated_w ** 2) + torch.sum((self.estimated_V*self.mask.view(-1))**2)
            return torch.sum(self.estimated_w ** 2) + torch.sum((self.estimated_V[self.mask_bool.view(-1)])**2)

        else:
            return torch.sum(self.estimated_w**2)
        
    def _l1_penalty(self):
        if self.interaction:
            #return torch.sum(torch.abs(self.estimated_w)) + torch.sum(torch.abs(self.estimated_V))
            #return torch.sum(torch.abs(self.estimated_w)) + torch.sum(torch.abs( self.estimated_V*self.mask.view(-1) ))
            return torch.sum(torch.abs(self.estimated_w)) + torch.sum(torch.abs(self.estimated_V[self.mask_bool.view(-1)] ))

        else:
            return torch.sum(torch.abs(self.estimated_w))


    def _lsm_penalty(self):
        # this is the difference between estimated interaction weights and UU.T
        if not self.interaction:
            raise ValueError('reduced ranked pernalty is only used when the interaction terms are incorporated')
        C_flattened = torch.matmul(self.estimated_Zd, torch.t(self.estimated_Zd)).view(self.p*self.p)
        #diff = self.estimated_V * self.mask.view(-1) - flattened_UUT * self.mask.view(-1)
        diff = self.estimated_V[self.mask_bool.view(-1)] - C_flattened[self.mask_bool.view(-1)]



        penalty = torch.sum(diff**2)
        return penalty

    def _mse(self, X, X_interaction, y):

        X_interaction_masked = X_interaction[:, self.mask.view(-1).bool()]
        #prediction = self.predict(X, X_interaction, self.estimated_w, self.estimated_V*self.mask.view(-1))
        prediction = self.predict(X, X_interaction_masked, self.estimated_w, self.estimated_V[self.mask.view(-1).bool()])
        squared_error = (prediction - y)**2
        mse = 0.5 * torch.sum(squared_error)
        return mse




    def _loss(self,X,X_interction, y):
        if not self.regularization and not self.interaction and not self.rrr:
            return self._mse(X,X_interction,y)
        
        elif not self.regularization and not self.interaction and self.rrr:
            raise ValueError('reduced rank regression works only when there is interaction')
        
        elif not self.regularization and self.interaction and not self.rrr:
            return self._mse(X,X_interction,y)
        
        elif not self.regularization and self.interaction and self.rrr:
            return self._mse(X,X_interction,y) + self.gamma * self._lsm_penalty()
        
        elif  self.regularization and not self.interaction and not self.rrr:
            return self._mse(X,X_interction,y) + self.alpha * self._l2_penalty() + self.kappa * self._l1_penalty()
        
        elif self.regularization and not self.interaction and self.rrr:
            raise ValueError('reduced rank regression works only when there is interaction')
        
        elif self.regularization and self.interaction and not self.rrr:
            #return self._mse(X,X_interction,y) + self.alpha * self._l2_penalty_w() +self.kappa * self._l2_penalty_V()
            return self._mse(X,X_interction,y) + self.alpha * self._l2_penalty() + self.kappa * self._l1_penalty()
        
        elif self.regularization and self.interaction and self.rrr:
            #return self._mse(X,X_interction, y) + self.alpha * self._l2_penalty_w() + self.kappa * self._l2_penalty_V() + self.gamma * self._lsm_penalty() 
            return self._mse(X,X_interction, y) + self.alpha * self._l2_penalty() + self.kappa * self._l1_penalty() + self.gamma * self._lsm_penalty() 



    def fit(self, X_train, X_train_interaction, y_train,X_val = None, X_val_interaction=None, y_val=None, learning_rate=0.001, num_epochs=500, ES_threshold = 20):

        #self.to(self.device) # moving model parameters to cuda
        self.mask = self.mask.to(self.device)

        if  not self.interaction and not self.rrr: 
            optimizer = optim.Adam([self.estimated_w], lr=learning_rate)
        elif self.interaction and not self.rrr:
            optimizer = optim.Adam([self.estimated_w, self.estimated_V], lr=learning_rate)
        elif self.interaction and self.rrr: 
            optimizer = optim.Adam([self.estimated_w, self.estimated_V, self.estimated_Zd], lr=learning_rate)
        else:
            raise ValueError('Choose a correct combination of self.interaction and self.rrr')



        # Move training data to GPU
        X_train, X_train_interaction, y_train = X_train.to(self.device), X_train_interaction.to(self.device), y_train.to(self.device)

        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(num_epochs):
            #self.like.append(-1*self._negative_log_likelihood(X,y))
        
            #self._data_collection() #only when the interaction is considered!
            optimizer.zero_grad()
            train_loss = self._loss(X_train, X_train_interaction, y_train)
            self.train_losses.append(train_loss)
            train_loss.backward()
            optimizer.step()


            # Clear optimizer state to release GPU memory
            optimizer.zero_grad()
            optimizer.step()

            # Explicitly release GPU memory
            torch.cuda.empty_cache()

            # adding early stopping
            if X_val != None and X_val_interaction != None and y_val != None:
                X_val, X_val_interaction, y_val = X_val.to(self.device), X_val_interaction.to(self.device), y_val.to(self.device)
                with torch.no_grad():
                    val_loss = self._loss(X_val, X_val_interaction, y_val)
                self.val_losses.append(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= ES_threshold:
                    #print(f"Early stopping at epoch {epoch} with patience {EStreshold}")
                    break  # This break is inside the if statement
            #if (epoch + 1) % 1000 == 0:
            #    print(f"Epoch {epoch + 1}")
                
        return self.estimated_w, self.estimated_V

    def predict(self, X, X_interaction, w, V):
        
        # ensuring that the data is on the gpu
        X, X_interaction, w, V = X.to(self.device), X_interaction.to(self.device), w.to(self.device), V.to(self.device)

        if self.interaction:
            
            #concatenating the weights and the feature matrices to perfomr the matrix multiplication
            X_concat = torch.cat((X, X_interaction), dim=1)
            wV_concat = torch.cat((w[1:], V))
            prediction = torch.matmul(X_concat, wV_concat)+w[0]


        if not self.interaction:
            # choosing 1st index to last index as the coefficients of the features and choosing index 0 as the intercept
            prediction = torch.matmul(X, w[1:]) + w[0]
            

        return prediction
    