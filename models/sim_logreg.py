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
from scipy.linalg import toeplitz




manual_seed = 43421654
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)




class LogisticRegression_Simulator(nn.Module):
    def __init__(self,d, p, interaction = True, sparsity = True, sigma =  0.00001, lsm_penalty_type = 'latent_distance', V_noise=0.1,
                    cov_type='iid',        # 'iid', 'ar1', or 't'
                    rho=0.5,               # AR(1) correlation
                    t_df=5):

        super(LogisticRegression_Simulator, self).__init__()

        if "TPU_NAME" in os.environ:
            self.device = torch.device("xla")  # XLA is PyTorch's TPU device
        else:
            # Check if GPU is available
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                
        self.cov_type = cov_type.lower()
        self.rho      = rho
        self.t_df     = t_df
        #p represents the original dimensionality of the data
        self.p = p

        self.d = d
        # whether to perform calculations with intercept or not
        
        self.mask = torch.triu(torch.ones(p, p), diagonal=1)
        self.mask_bool = torch.triu(torch.ones(self.p, self.p), diagonal=1).bool()

        # whether to incorporate interaction in the analysis or not
        self.interaction  = interaction

        self.sigma = sigma


        self.sparsity = sparsity

        self.lsm_penalty_type = lsm_penalty_type 
        
        self.estimated_w = nn.Parameter(torch.randn(p+1, requires_grad=True))
                    
        self.initial_w = copy.deepcopy(self.estimated_w)
        

        #this is the initialiaation weight matrix --> for each interaction term in the model
        if interaction:

            self.estimated_V = nn.Parameter((torch.randn(p,p, requires_grad=True).view(p*p)) * self.mask.view(-1))

        else:
            self.estimated_V = None


        #self.estimated_V.data = init.xavier_normal_(self.estimated_V.data)

        self.initial_V = copy.deepcopy(self.estimated_V)
        #self.u is meant to be use if we assume that the weight matrix has low-rank structure
        # this is the initialization for U which will be optimized
        
        self.estimated_Zd = nn.Parameter(torch.randn(p,d, requires_grad=True))
        self.initial_Zd = copy.deepcopy(self.estimated_Zd)

        self.estimated_alpha_0 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.initial_alpha_0 = copy.deepcopy(self.estimated_alpha_0)

        self.true_w = torch.randn(self.p+1)
        # if self.sparsity:
        #     self.true_w = self._apply_sparsity(self.true_w,self.sigma)
        self.true_w = self.true_w.to(self.device)


        # whether we have intercept or not, it's not gonna change the number of weights for the interaction weights
        if self.interaction:

            self.trueZd = torch.randn(self.p, self.d)

            if self.lsm_penalty_type == 'latent_distance':

                Z1_expanded = self.trueZd.unsqueeze(1) # (p,1,d)
                Z2_expanded = self.trueZd.unsqueeze(0) # (1,p,d)
                self.true_V = torch.norm(Z1_expanded - Z2_expanded, dim=2,p=2)**2
                self.true_V = self.true_V.view(-1)
                noise = np.random.normal(0, np.sqrt(V_noise), size=len(self.true_V))
                noise = torch.tensor(noise, dtype = torch.float32)
                self.true_V = self.true_V + noise
                self.true_V = self.true_V * self.mask.view(-1)

                if self.sparsity:
                    self.true_V = self._apply_sparsity(self.true_V[self.mask_bool.view(-1)], self.sigma)
                self.true_V = self.true_V.to(self.device)
                
            elif self.lsm_penalty_type == 'lowRank':

                self.true_V = torch.matmul(self.trueZd, torch.t(self.trueZd)).view(self.trueZd.shape[0]*self.trueZd.shape[0])
                noise = np.random.normal(0, np.sqrt(V_noise), size=len(self.true_V))
                noise = torch.tensor(noise, dtype = torch.float32)
                self.true_V = self.true_V + noise
                self.true_V = self.true_V * self.mask.view(-1)

                if self.sparsity:
                    sparse_values = self._apply_sparsity(self.true_V[self.mask_bool.view(-1)], self.sigma)
                    sparse_values = sparse_values.to(self.device)
                    self.true_V = torch.zeros(self.p, self.p, device=self.device)
                    self.true_V[self.mask_bool] = sparse_values
                    self.true_V = self.true_V.view(-1)
                self.true_V = self.true_V.to(self.device)
        
        else:
            self.true_V = None

        

        self.criterion = nn.BCELoss()


    def _apply_sparsity(self, vector, sigma):
        """
        The higher the sigma, the more spare the vector is! I've tested this and it works properly!
        """

        probability = torch.exp(-(vector**2) / sigma)
        noise = torch.rand_like(vector)
        mask = (noise > probability).float()  # Retain elements with higher probability
        sparse_vector = vector * mask
        return sparse_vector


    
    def simulate(self, num_samples):
        '''
        we need true weights to calculate the target variables. after the optimizatin, initial weights should be close to the true weights. 
        '''
        
        # ---------- build covariance Σ ----------
        if self.cov_type == 'iid':
            cov = np.eye(self.p)

        elif self.cov_type == 'ar1':
            # Toeplitz(1, ρ, ρ², …)
            cov = toeplitz(self.rho ** np.arange(self.p))

        elif self.cov_type == 't':
            cov = toeplitz(self.rho ** np.arange(self.p))
        else:
            raise ValueError("cov_type must be 'iid', 'ar1', or 't'")

        # ---------- draw feature matrix ----------
        # if self.cov_type in ('iid', 'ar1'):
        #     X = np.random.multivariate_normal(np.zeros(self.p), cov, size=num_samples)

        # elif self.cov_type == 't':
        #     # sample t_ν, then correlate via Σ½
        #     Z = np.random.standard_t(self.t_df, size=(num_samples, self.p))
        #     L = np.linalg.cholesky(cov)          # Σ = LLᵀ
        #     X = Z @ L.T                          # gives desired covariance
        # specifying the mean and covariance for feature matrix which will be drawn from a multivariate gaussian
        mean = np.zeros(self.p) # mean 0
        # cov = np.eye(self.p)  # identity covariance matrix
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
            #print(f'self.true_W device: {self.true_w.device} self.true_V device: {self.true_V.device}, self.mask_bool device: {self.mask_bool.device}')
            y = self.predict_proba(X_tensor, X_tensor_interaction, self.true_w, self.true_V, self.mask_bool.view(-1,1))
        else:
            y = self.predict_proba(X=X_tensor, X_interaction=None, w=self.true_w, V=self.true_V, mask_bool=self.mask_bool.view(-1,1))
        y = (y >= 0.5).float()

        # creating noise (epsilon) to add to the prediction. Increasing the variance make the problem harder
        #noise = np.random.normal(0, np.sqrt(noise_variance), size=num_samples)
        #noise = torch.tensor(noise, dtype = torch.float32) 
        #noise = noise.to(self.device)
        #y = prediction + noise
        #y_tensor = torch.tensor(y, dtype=torch.float32)


        if self.interaction:
            return X_tensor.to(self.device), X_tensor_interaction.to(self.device), y.to(self.device)
        
        else: 
            return X_tensor.to(self.device), None,  y.to(self.device)

    def predict_proba(self, X, X_interaction, w, V, mask_bool):
        if self.interaction:
            V_masked = V[mask_bool.view(-1)]

            X_interaction_masked = X_interaction[:, mask_bool.view(-1)]
            #concatenating the weights and the feature matrices to perfomr the matrix multiplication
            #X_concat = torch.cat((X, X_interaction), dim=1)
            X_concat = torch.cat((X, X_interaction_masked), dim=1)
            #wV_concat = torch.cat((w[1:], V))
            wV_concat = torch.cat((w[1:], V_masked))

            linear_comb = torch.matmul(X_concat, wV_concat) + w[0]

        else:
            linear_comb = torch.matmul(X, w[1:]) + w[0]

        predict_proba = torch.sigmoid(linear_comb)
        return predict_proba