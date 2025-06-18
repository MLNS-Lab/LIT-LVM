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



class CoxPH_LSM(nn.Module):

    def __init__(self, d=2, estimated_w = None, estimated_Va=None, estimated_Vb=None, estimated_Vdr=None, 
                Va_shape=None, Vb_shape=None, Vdr_shape=None, 
                interaction=True,  ls_penalty=True, ls_penalty_type = 'lowRank', alpha=0.5, kappa=0.5, gamma=0.5,c=1):
        super(CoxPH_LSM, self).__init__()
        

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


        self.ls_penalty_type = ls_penalty_type

        #p represents the original dimensionality of the data
        self.estimated_w = estimated_w
        self.initial_w = self.estimated_w.clone()
        self.estimated_Va = estimated_Va
        self.estimated_Vb = estimated_Vb
        self.estimated_Vdr = estimated_Vdr

        #these are used for restarting the parameters of the model
        self.initial_w = copy.deepcopy(self.estimated_w)
        self.initial_Va = copy.deepcopy(self.estimated_Va)
        self.initial_Vb = copy.deepcopy(self.estimated_Vb)
        self.initial_Vdr = copy.deepcopy(self.estimated_Vdr)


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

        # if ls_penalty_type == 'latent_distance':

        self.estimated_alpha_0_a = nn.Parameter(torch.randn(1, requires_grad=True))
        self.initial_alpha_0_a = copy.deepcopy(self.estimated_alpha_0_a)

        self.estimated_alpha_0_b = nn.Parameter(torch.randn(1, requires_grad=True))
        self.initial_alpha_0_b = copy.deepcopy(self.estimated_alpha_0_b)

        self.estimated_alpha_0_dr = nn.Parameter(torch.randn(1, requires_grad=True))
        self.initial_alpha_0_dr = copy.deepcopy(self.estimated_alpha_0_dr)
        

        # whether to incorporate interaction in the analysis or not
        self.interaction  = interaction


        #ls_penaly = True --> adding reduced randked regression to the penalty
        self.lsm_penalty = ls_penalty


        self.alpha = alpha

        self.kappa = kappa
        self.gamma = gamma
        self.train_losses = []

        self.val_losses = []


    
    def _reset_parameters(self):
        # Reset estimated_w to its initial value

        self.estimated_w.data = self.initial_w.data.clone()
            

        # Reset estimated_V to its initial value
        if self.interaction:
            self.estimated_Va.data = self.initial_Va.data.clone()
            self.estimated_Vb.data = self.initial_Vb.data.clone()
            self.estimated_Vdr.data = self.initial_Vdr.data.clone()

        # Reset U to its initial value
        self.estimated_Za_don.data = self.initial_Za_don.data.clone()
        self.estimated_Za_rec.data = self.initial_Za_rec.data.clone()

        self.estimated_Zb_don.data = self.initial_Zb_don.data.clone()
        self.estimated_Zb_rec.data = self.initial_Zb_rec.data.clone()

        self.estimated_Zdr_don.data = self.initial_Zdr_don.data.clone()
        self.estimated_Zdr_rec.data = self.initial_Zdr_rec.data.clone()

        # if self.ls_penalty_type == 'latent_distance':
            
        self.estimated_alpha_0_a.data = self.initial_alpha_0_a.data.clone()

        self.estimated_alpha_0_b.data = self.initial_alpha_0_b.data.clone()

        self.estimated_alpha_0_dr.data = self.initial_alpha_0_dr.data.clone()

    def predict(self, X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            estimated_w, estimated_Va, estimated_Vb, estimated_Vdr):
        '''This Calculates exp(W'X)'''
        
        #h = W'X
        hazard = self._hazard(X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            estimated_w, estimated_Va, estimated_Vb, estimated_Vdr)

        return torch.exp(hazard) #exp(h)
        
    def predict_freq(self, X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            estimated_w, estimated_Va, estimated_Vb, estimated_Vdr):
        '''This Calculates exp(W'X)'''
        
        #h = W'X
        hazard_freq = self._hazard_freq(X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            estimated_w, estimated_Va, estimated_Vb, estimated_Vdr)
        
        return torch.exp(hazard_freq) #exp(h)
    
    def _hazard_freq(self, X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            estimated_w, estimated_Va, estimated_Vb, estimated_Vdr):
        
        if not self.interaction:

            h = torch.matmul(X_train, estimated_w)
            return h
        else:

            X_concat = torch.cat((X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train), dim=1)
            wV_concat = torch.cat((estimated_w, estimated_Va, estimated_Vb, estimated_Vdr))

            h = torch.matmul(X_concat, wV_concat)
            return h

        
    def _hazard(self, X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            estimated_w, estimated_Va, estimated_Vb, estimated_Vdr):
        '''
        This function calculates the linear combination of the input features W'(w transpose)X
        V is flattened
        This calcualtes W'X
        '''
        # this if is for the situation without interaction term
        if not self.interaction:
            #w[0] is beta_0 (intecept)
            #Xbar = torch.mean(X_train,dim=0)
            h = torch.matmul(X_train, estimated_w)
            #print('h:',h)
            return h

        else:
            X_concat = torch.cat((X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train), dim=1)
            #some of the elements of estimated_Va is zero (if it's not getting optimized) and I remove those elemenst so that the shape of matrix and the length of the weight for matrix multiplication matches
            squeezed_estimated_Va = torch.squeeze(estimated_Va)  # squeeze to align the size of estimated_Va and self.hla_a_pairs_mask_flattened_bool
            squeezed_estimated_Vb = torch.squeeze(estimated_Vb)
            squeezed_estimated_Vdr = torch.squeeze(estimated_Vdr)
            wV_concat = torch.cat((estimated_w, squeezed_estimated_Va[self.hla_a_pairs_mask_flattened_bool], squeezed_estimated_Vb[self.hla_b_pairs_mask_flattened_bool], squeezed_estimated_Vdr[self.hla_dr_pairs_mask_flattened_bool]))

            h = torch.matmul(X_concat, wV_concat)
            return h

    def _l2_penalty(self):

        if self.interaction:
            return torch.sum(self.estimated_w**2) + torch.sum((self.estimated_Va* self.hla_a_pairs_mask_flattened)**2) + torch.sum((self.estimated_Vb*self.hla_b_pairs_mask_flattened)**2) + torch.sum((self.estimated_Vdr*self.hla_dr_pairs_mask_flattened)**2)
        else:
            return torch.sum(self.estimated_w**2)

    def _l2_penalty_freq(self):

        if self.interaction:
            '''  
            first neutralizing inactive hla pairs by self.hla_a_pairs_mask_flattened
            then removing the inactive pairs Va[Va !=0]
            finally remove the infrequent pairs by multiplying self.freq_mask_a
            '''

            Va = self.estimated_Va* self.hla_a_pairs_mask_flattened
            Vb = self.estimated_Vb*self.hla_b_pairs_mask_flattened
            Vdr =  self.estimated_Vdr*self.hla_dr_pairs_mask_flattened

            Va = Va[self.hla_a_pairs_mask_flattened_bool] * self.freq_mask_a
            Vb = Vb[self.hla_b_pairs_mask_flattened_bool] * self.freq_mask_b
            Vdr = Vdr[self.hla_dr_pairs_mask_flattened_bool] * self.freq_mask_dr
            return torch.sum(self.estimated_w**2) + torch.sum((Va)**2) + torch.sum((Vb)**2) + torch.sum((Vdr)**2)
        else:
            return torch.sum(self.estimated_w**2)
        
    def _l1_penalty(self):
        if self.interaction:
            return torch.sum(torch.abs(self.estimated_w)) + torch.sum(torch.abs((self.estimated_Va*self.hla_a_pairs_mask_flattened))) + torch.sum(torch.abs((self.estimated_Vb* self.hla_b_pairs_mask_flattened))) + torch.sum(torch.abs((self.estimated_Vdr*self.hla_dr_pairs_mask_flattened)))
        else:
            return torch.sum(torch.abs(self.estimated_w))
        
    def _l1_penalty_freq(self):
        if self.interaction:
            Va = self.estimated_Va* self.hla_a_pairs_mask_flattened
            Vb =self.estimated_Vb*self.hla_b_pairs_mask_flattened
            Vdr =  self.estimated_Vdr*self.hla_dr_pairs_mask_flattened

            estimated_Va = Va[self.hla_a_pairs_mask_flattened_bool] * self.freq_mask_a
            estimated_Vb = Vb[self.hla_b_pairs_mask_flattened_bool] * self.freq_mask_b
            estimated_Vdr = Vdr[self.hla_dr_pairs_mask_flattened_bool] * self.freq_mask_dr
            return torch.sum(torch.abs(self.estimated_w**2)) + torch.sum(torch.abs((estimated_Va)**2)) + torch.sum(torch.abs((estimated_Vb)**2)) + torch.sum(torch.abs((estimated_Vdr)**2))
        else:
            return torch.sum(torch.abs(self.estimated_w**2))


    def _lsm_penalty(self):
        '''
        V = [v11, v12, v13
            v21, v22, v23
            v31, v32, v33
            v41, v42, v43]
                            (4,3)
        V_flattedned = [v11, v12, v13, v21, v22, v23,v31, v32, v33,v41, v42, v43] 
                                                                                (1,12)
        
        mask = [1,1,0
                0,0,0
                0,1,0
                1,1,0]
                    (4,3)

        mask_flattened = [1,1,0,0,0,0,0,1,0,1,1,0]
                                                (1,12)

        Zd = [zd11, zd12
            zd21, zd22
            zd31, zd32
            zd41, zd42]
                        (4,2)


        Zr = [zr11, zr12
            zr21, zr22
            zr31, zr32]
                        (3,2)

        C = Zd.Zr.t() = [c11,c12,c13
                        c21,c22,c23
                        c31,c32,c33
                        c41,c42,c43]
                                    (4,3)
        C_flattened = [c11,c12,c13,c21,c22,c23,c31,c32,c33,c41,c42,c43]

        loss = || ([v11, v12, v13, v21, v22, v23,v31, v32, v33,v41, v42, v43] * [1,1,0,0,0,0,0,1,0,1,1,0]) - 
        ([c11,c12,c13,c21,c22,c23,c31,c32,c33,c41,c42,c43] * [1,1,0,0,0,0,0,1,0,1,1,0]) ||_2^2
        where * is elementwise product
        
        '''

        if self.ls_penalty_type == 'lowRank':
            if not self.interaction:
                raise ValueError('reduced ranked pernalty is only used when the interaction terms are incorporated')
            
            else:
                #Za: (pd, d) . (pr,d).T ---> (pd, pr) ----> flatten ----> diff = Zda_Zra_T_flattened - self.estimated_Va
                Ca_flattened = torch.matmul(self.estimated_Za_don, torch.t(self.estimated_Za_rec)).view(self.estimated_Za_don.shape[0]*self.estimated_Za_rec.shape[0])
                diff_a = self.estimated_Va * self.hla_a_pairs_mask_flattened - Ca_flattened * self.hla_a_pairs_mask_flattened
                penalty_a = torch.sum(diff_a**2)

                Cb_flattened = torch.matmul(self.estimated_Zb_don, torch.t(self.estimated_Zb_rec)).view(self.estimated_Zb_don.shape[0]*self.estimated_Zb_rec.shape[0])
                diff_b = self.estimated_Vb * self.hla_b_pairs_mask_flattened - Cb_flattened * self.hla_b_pairs_mask_flattened
                penalty_b = torch.sum(diff_b**2)

                Cdr_flattened = torch.matmul(self.estimated_Zdr_don, torch.t(self.estimated_Zdr_rec)).view(self.estimated_Zdr_don.shape[0]*self.estimated_Zdr_rec.shape[0])
                diff_dr = self.estimated_Vdr * self.hla_dr_pairs_mask_flattened - Cdr_flattened * self.hla_dr_pairs_mask_flattened
                penalty_dr = torch.sum(diff_dr**2)

                penalty = penalty_a + penalty_b + penalty_dr

        if self.ls_penalty_type == 'latent_distance':
            if not self.interaction:
                raise ValueError('reduced ranked pernalty is only used when the interaction terms are incorporated')
            else:

                penalty_a = self._latent_distance_penalty(self.estimated_Za_don, self.estimated_Za_rec, self.estimated_alpha_0_a, self.estimated_Va, self.hla_a_pairs_mask_flattened)
                penalty_b = self._latent_distance_penalty(self.estimated_Zb_don, self.estimated_Zb_rec, self.estimated_alpha_0_b, self.estimated_Vb, self.hla_b_pairs_mask_flattened)
                penalty_dr = self._latent_distance_penalty(self.estimated_Zdr_don, self.estimated_Zdr_rec, self.estimated_alpha_0_dr, self.estimated_Vdr, self.hla_dr_pairs_mask_flattened)

                penalty = penalty_a + penalty_b + penalty_dr

        return penalty
    
    
    def _lsm_penalty_freq(self):
        '''
        V = [v11, v12, v13
            v21, v22, v23
            v31, v32, v33
            v41, v42, v43]
                            (4,3)
        V_flattedned = [v11, v12, v13, v21, v22, v23,v31, v32, v33,v41, v42, v43] 
                                                                                (1,12)
        
        mask = [1,1,0
                0,0,0
                0,1,0
                1,1,0]
                    (4,3)

        mask_flattened = [1,1,0,0,0,0,0,1,0,1,1,0]
                                                (1,12)

        Zd = [zd11, zd12
            zd21, zd22
            zd31, zd32
            zd41, zd42]
                        (4,2)


        Zr = [zr11, zr12
            zr21, zr22
            zr31, zr32]
                        (3,2)

        C = Zd.Zr.t() = [c11,c12,c13
                        c21,c22,c23
                        c31,c32,c33
                        c41,c42,c43]
                                    (4,3)
        C_flattened = [c11,c12,c13,c21,c22,c23,c31,c32,c33,c41,c42,c43]

        loss = || ([v11, v12, v13, v21, v22, v23,v31, v32, v33,v41, v42, v43] * [1,1,0,0,0,0,0,1,0,1,1,0]) - 
        ([c11,c12,c13,c21,c22,c23,c31,c32,c33,c41,c42,c43] * [1,1,0,0,0,0,0,1,0,1,1,0]) ||_2^2
        where * is elementwise product
        
        '''

        if self.ls_penalty_type == 'lowRank':
            if not self.interaction:
                raise ValueError('reduced ranked pernalty is only used when the interaction terms are incorporated')
            
            else:
                #Za: (pd, d) . (pr,d).T ---> (pd, pr) ----> flatten ----> diff = Zda_Zra_T_flattened - self.estimated_Va
                Ca_flattened = torch.matmul(self.estimated_Za_don, torch.t(self.estimated_Za_rec)).view(self.estimated_Za_don.shape[0]*self.estimated_Za_rec.shape[0])
                diff_a = self.estimated_Va * self.hla_a_pairs_mask_flattened - Ca_flattened * self.hla_a_pairs_mask_flattened
                #print('shape of diff_a before removing inactive:', diff_a.shape)
                diff_a = (diff_a[self.hla_a_pairs_mask_flattened_bool])
                #print('shape of diff_a after removing inactive:', diff_a.shape)
                diff_a = diff_a * self.freq_mask_a
                #print('shape of diff_a after freq_mask:', diff_a.shape)
                penalty_a = torch.sum(diff_a**2)

                Cb_flattened = torch.matmul(self.estimated_Zb_don, torch.t(self.estimated_Zb_rec)).view(self.estimated_Zb_don.shape[0]*self.estimated_Zb_rec.shape[0])
                diff_b = self.estimated_Vb * self.hla_b_pairs_mask_flattened - Cb_flattened * self.hla_b_pairs_mask_flattened
                #print('shape of diff_b before removing inactive:', diff_b.shape)
                diff_b = (diff_b[self.hla_b_pairs_mask_flattened_bool]) 
                #print('shape of diff_b after removing inactive:', diff_b.shape)
                diff_b = diff_b *  self.freq_mask_b
                #print('shape of diff_b after freq_mask:', diff_b.shape)
                penalty_b = torch.sum(diff_b**2)

                Cdr_flattened = torch.matmul(self.estimated_Zdr_don, torch.t(self.estimated_Zdr_rec)).view(self.estimated_Zdr_don.shape[0]*self.estimated_Zdr_rec.shape[0])
                diff_dr = self.estimated_Vdr * self.hla_dr_pairs_mask_flattened - Cdr_flattened * self.hla_dr_pairs_mask_flattened
                #print('shape of diff_dr before removing inactive:', diff_dr.shape)
                diff_dr = (diff_dr[self.hla_dr_pairs_mask_flattened_bool]) 
                #print('shape of diff_dr after removing inactive:', diff_dr.shape)
                diff_dr = diff_dr * self.freq_mask_dr
                #print('shape of diff_dr after freq_mask:', diff_dr.shape)
                penalty_dr = torch.sum(diff_dr**2)

                penalty = penalty_a + penalty_b + penalty_dr

        if self.ls_penalty_type == 'latent_distance':
            if not self.interaction:
                raise ValueError('reduced ranked pernalty is only used when the interaction terms are incorporated')
            else:

                penalty_a = self._latent_distance_penalty_freq(self.estimated_Za_don, self.estimated_Za_rec, self.estimated_alpha_0_a,  self.estimated_Va, self.hla_a_pairs_mask_flattened, self.freq_mask_a)
                penalty_b = self._latent_distance_penalty_freq(self.estimated_Zb_don, self.estimated_Zb_rec, self.estimated_alpha_0_b , self.estimated_Vb, self.hla_b_pairs_mask_flattened, self.freq_mask_b)
                penalty_dr = self._latent_distance_penalty_freq(self.estimated_Zdr_don, self.estimated_Zdr_rec, self.estimated_alpha_0_dr, self.estimated_Vdr, self.hla_dr_pairs_mask_flattened, self.freq_mask_dr)

                penalty = penalty_a + penalty_b + penalty_dr

        return penalty
    def _compatibility(self, estimated_alpha_0,estimated_Zd, estimated_Zr):
        Zd_expanded = estimated_Zd.unsqueeze(1)
        Zr_expanded = estimated_Zr.unsqueeze(0)
        compat = estimated_alpha_0 + torch.norm(Zd_expanded - Zr_expanded, dim=2, p=2)**2
        return compat
    def _latent_distance_penalty(self, estimated_Zd, estimated_Zr, estimated_alpha_0, estimated_V, mask_flattened):
        #Zd_expanded = estimated_Zd.unsqueeze(1)
        #Zr_expanded = estimated_Zr.unsqueeze(0)
        #compatibility = estimated_alpha_0 -  torch.norm(Zd_expanded - Zr_expanded, dim=2, p=2)**2
        compatibility = self._compatibility(estimated_alpha_0, estimated_Zd, estimated_Zr)
        compatibility = compatibility.view(estimated_Zd.shape[0]*estimated_Zr.shape[0])
        #print('shape of compatibility', compatibility.shape)
        #print('shape of estimated_v', estimated_V.shape)
        #print(mask_flattened.view(1,-1).shape)
        diff = estimated_V[mask_flattened.view(1,-1).bool()] - compatibility.view(1,-1)[mask_flattened.view(1,-1).bool()]
        penalty = torch.sum(diff**2)
        return penalty
    def _latent_distance_penalty_freq(self, estimated_Zd, estimated_Zr, estimated_alpha_0, estimated_V, mask_flattened, freq_mask):
        Zd_expanded = estimated_Zd.unsqueeze(1)
        Zr_expanded = estimated_Zr.unsqueeze(0)
        compatibility = estimated_alpha_0 -  torch.norm(Zd_expanded - Zr_expanded, dim=2, p=2)**2
        compatibility = compatibility.view(estimated_Zd.shape[0]*estimated_Zr.shape[0])
        diff = estimated_V * mask_flattened - compatibility * mask_flattened
        mask_bool = mask_flattened.bool().view(1,-1)
        diff = (diff[mask_bool]) * freq_mask
        penalty = torch.sum(diff**2)
        return penalty

    def _loss(self,X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            survival_censoring_time_train, events_train):
        
        
        
        if   not self.interaction and not self.lsm_penalty:
            return self._negative_partial_log_liklihood(X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            survival_censoring_time_train, events_train) + self.alpha * self._l2_penalty() 
        
        elif  not self.interaction and self.lsm_penalty:
            raise ValueError('reduced rank regression works only when there is interaction')
        
        elif  self.interaction and not self.lsm_penalty:
            return self._negative_partial_log_liklihood(X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            survival_censoring_time_train, events_train) + self.alpha * self._l2_penalty() + self.kappa 
        
        elif  self.interaction and self.lsm_penalty:
            return self._negative_partial_log_liklihood(X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            survival_censoring_time_train, events_train) + self.alpha * self._l2_penalty() + self.gamma * self._lsm_penalty() 


    def _loss_freq(self,X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            survival_censoring_time_train, events_train):
        

        
        if   self.interaction and not self.lsm_penalty:
            return self._negative_partial_log_liklihood_freq(X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            survival_censoring_time_train, events_train) + self.alpha * self._l2_penalty_freq() 
        
        elif  self.interaction and self.lsm_penalty:
            raise ValueError('reduced rank regression works only when there is interaction')
        
        elif  self.interaction and not self.lsm_penalty:
            return self._negative_partial_log_liklihood_freq(X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            survival_censoring_time_train, events_train) + self.alpha * self._l2_penalty_freq() 
        
        elif  self.interaction and self.lsm_penalty:
            return self._negative_partial_log_liklihood_freq(X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            survival_censoring_time_train, events_train) + self.alpha * self._l2_penalty_freq() + self.kappa * self._l1_penalty() + self.gamma * self._lsm_penalty_freq() 


    def _negative_partial_log_liklihood(self, X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            survival_censoring_time_train, events_train):

        
        #X.w + X_interaction.v --> (n,p).(p,1) + (n,p^2).(p^2,1) ---> (n,1)   
        h = self._hazard(X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            self.estimated_w, self.estimated_Va, self.estimated_Vb, self.estimated_Vdr)

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
    
    def _negative_partial_log_liklihood_freq(self, X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            survival_censoring_time_train, events_train):
        '''  
            survival_censoring_time       event       h
        1-    10                          T           h1
        2-    5                           T           h2
        3-    2                           F           h3
        4-    12                          F           h4
        5-    8                           T           h5
        6-    3                           F           h6

            sort the data based on survival censoring time

            survival_censoring_time    event        h       h_reversed
            2                          F            h3      h4
            3                          F            h6      h1
            5                          T            h2      h5
            8                          T            h5      h2
            10                         T            h1      h6
            12                         F            h4      h3


            torch.logcumsumexp(h_reveresed,dim=0)

            log(exp(h4))
            log(exp(h4)+exp(h1))
            log(exp(h4)+exp(h1)+exp(h5))
            log(exp(h4)+exp(h1)+exp(h5)+exp(h2))
            log(exp(h4)+exp(h1)+exp(h5)+exp(h2)+exp(h6))
            log(exp(h4)+exp(h1)+exp(h5)+exp(h2)+exp(h6)+exp(h3))

            
            Then reverse this:
            log(exp(h4)+exp(h1)+exp(h5)+exp(h2)+exp(h6)+exp(h3))  
            log(exp(h4)+exp(h1)+exp(h5)+exp(h2)+exp(h6))  
            log(exp(h4)+exp(h1)+exp(h5)+exp(h2))  
            log(exp(h4)+exp(h1)+exp(h5))  
            log(exp(h4)+exp(h1))  
            log(exp(h4))




            PLL

            
            h3              log(exp(h4)+exp(h1)+exp(h5)+exp(h2)+exp(h6)+exp(h3))  
            h6              log(exp(h4)+exp(h1)+exp(h5)+exp(h2)+exp(h6))  
            h2       -      log(exp(h4)+exp(h1)+exp(h5)+exp(h2))  
            h5              log(exp(h4)+exp(h1)+exp(h5))  
            h1              log(exp(h4)+exp(h1))  
            h4              log(exp(h4))        

            
            Then I need to sum over the uncensored elements to get back the PLL
        
        
        '''
        
        #X.w + X_interaction.v --> (n,p).(p,1) + (n,p^2).(p^2,1) ---> (n,1)   
        Va = self.estimated_Va[self.hla_a_pairs_mask_flattened_bool]
        Vb = self.estimated_Vb[self.hla_b_pairs_mask_flattened_bool]
        Vdr = self.estimated_Vdr[self.hla_dr_pairs_mask_flattened_bool]

        estimated_Va = Va * self.freq_mask_a
        estimated_Vb = Vb * self.freq_mask_b
        estimated_Vdr = Vdr * self.freq_mask_dr
        h = self._hazard_freq(X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            self.estimated_w, estimated_Va, estimated_Vb, estimated_Vdr)


        '''I can use h and calcualte exp_h ---> change it later'''
        #exp_h = self.predict(X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
        #      self.estimated_w, self.estimated_Va, self.estimated_Vb, self.estimated_Vdr)

        # Initialize negative log-partial likelihood with zero value. values will be added in each iteration
        neg_log_partial_likelihood = 0.0


        # we sort all the tensors based on the order of survival or censoring time
        _, indices = torch.sort(survival_censoring_time_train)
    

        # sorting survival censoring time
        #survival_censoring_time_sorted = survival_censoring_time_train[indices]
        #print('surviva time sorted:', survival_censoring_time_sorted)

        # sorting events based on the order of survival or censoring time
        events_sorted = events_train[indices]

        # sorting h based on the order of survival or censoring time
        h_sorted = h[indices]
        #print('h sorted', h_sorted)
        h_sorted_reversed = torch.flip(h_sorted, dims=[0])
        logcumsumexp = torch.logcumsumexp(h_sorted_reversed,dim=0)
        logcumsumexp_reversed = torch.flip(logcumsumexp,dims=[0])

        # sorting exp_h based on the order of survival_censoring time ---> [e^h0, e^h1, e^h2]  h0 corresponded to the h of sample with lowest survival or censoring time and h2 correspond to the highest
        #exp_h_sorted = exp_h[indices]
        #print('exp_h sorted', exp_h_sorted)
        #reverse exp_h_sorted ---> [e^h2, e^h1, e^h0]
        #exp_h_reversed = torch.flip(exp_h_sorted, dims=[0])
        #print('exp_h_reversed', exp_h_reversed)
        # cumulative sum of reverse of exp_h_soreted ---> [e^h2, e^h2+^h1, e^h2+e^h1+e^h0]
        #cum_sum_exp_h_reversed = torch.cumsum(exp_h_reversed, dim=0)

        # for the vectorized calculation, we need the reverse of this:
        #[e^h2+e^h1+e^h0, e^h2+e^h1, e^h0]
        #reversed_cum_sum_exp_h_reversed = torch.flip(cum_sum_exp_h_reversed, dims=[0])
        #print('reversed_cum_sum_exp_h_reversed', reversed_cum_sum_exp_h_reversed)
    

        #this is only for printing to make sure things are correct
        #log_reversed_cum_sum_exp_h_reversed = torch.log(reversed_cum_sum_exp_h_reversed)
        # Calculate risk set size
        risk_set_sizes = torch.sum(events_sorted == 1, dim=0)

        #print('1:',logcumsumexp_reversed)
        #print('2:', log_reversed_cum_sum_exp_h_reversed)

        # negative log partial likelihood
        #neg_log_partial_likelihood = -1 * torch.sum(  (h_sorted - torch.log(reversed_cum_sum_exp_h_reversed))[events_sorted] )/risk_set_sizes

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
#
#
        #print('---------------------------------------------------')


        

        if  not self.interaction and not self.lsm_penalty: 
            optimizer = optim.Adam([self.estimated_w], lr=learning_rate)
            #print('this model only optimizes NLL')
        elif self.interaction and not self.lsm_penalty:
            optimizer = optim.Adam([self.estimated_w, self.estimated_Va, self.estimated_Vb, self.estimated_Vdr], lr=learning_rate)
        elif self.interaction and self.lsm_penalty: 
            #print('this model considers Latent distance position of HLA types')
            if self.ls_penalty_type == 'lowRank':
                optimizer = optim.Adam([self.estimated_w, self.estimated_Va, self.estimated_Vb, self.estimated_Vdr,self.estimated_Za_don,self.estimated_Za_rec,self.estimated_Zb_don,self.estimated_Zb_rec, self.estimated_Zdr_don,self.estimated_Zdr_rec], lr=learning_rate)
            if self.ls_penalty_type == 'latent_distance':
                print('latent space penlaty optimization')
                optimizer = optim.Adam([self.estimated_w, self.estimated_Va, self.estimated_Vb, self.estimated_Vdr,self.estimated_Za_don,self.estimated_Za_rec,self.estimated_Zb_don,self.estimated_Zb_rec, self.estimated_Zdr_don,self.estimated_Zdr_rec, self.estimated_alpha_0_a, self.estimated_alpha_0_b, self.estimated_alpha_0_dr], lr=learning_rate)

        else:
            raise ValueError('Choose a correct combination of self.interaction and self.rrr')
        

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
                        self.estimated_Va.copy_(torch.sign(self.estimated_Va) * torch.clamp(torch.abs(self.estimated_Va) - learning_rate * self.kappa, min=0.0))
                        self.estimated_Vb.copy_(torch.sign(self.estimated_Vb) * torch.clamp(torch.abs(self.estimated_Vb) - learning_rate * self.kappa, min=0.0))
                        self.estimated_Vdr.copy_(torch.sign(self.estimated_Vdr) * torch.clamp(torch.abs(self.estimated_Vdr) - learning_rate * self.kappa, min=0.0)

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
        return self.estimated_w, self.estimated_Va, self.estimated_Vb, self.estimated_Vdr


    def fit_freq(self, X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            survival_censoring_time_train, events_train,
            X_val, hla_a_pairs_val, hla_b_pairs_val, hla_dr_pairs_val, 
            survival_censoring_time_val, events_val, 
            hla_a_pairs_mask, hla_b_pairs_mask, hla_dr_pairs_mask,
            freq_mask_a, freq_mask_b, freq_mask_dr,
            learning_rate=0.1, num_epochs=5000, ES_threshold = 20):
        '''X_train: data_basic + MM + hla types'''

        # flattening the mask matrices for easier computation 
        #print('flattening the mask matrices')
        self.hla_a_pairs_mask_flattened = hla_a_pairs_mask.reshape(-1)
        self.hla_a_pairs_mask_flattened_bool = self.hla_a_pairs_mask_flattened.bool()
        self.hla_a_pairs_mask_flattened_bool = self.hla_a_pairs_mask_flattened_bool.view(1, -1)
        #print(f'shape of hla_a_pairs_mask: {hla_a_pairs_mask.shape}, shape of flattened: {self.hla_a_pairs_mask_flattened.shape}')
        self.hla_b_pairs_mask_flattened = hla_b_pairs_mask.reshape(-1)
        self.hla_b_pairs_mask_flattened_bool = self.hla_b_pairs_mask_flattened.bool()
        self.hla_b_pairs_mask_flattened_bool = self.hla_b_pairs_mask_flattened_bool.view(1, -1)
        #print(f'shape of hla_b_pairs_mask: {hla_b_pairs_mask.shape}, shape of flattened: {self.hla_b_pairs_mask_flattened.shape}')
        self.hla_dr_pairs_mask_flattened = hla_dr_pairs_mask.reshape(-1)
        self.hla_dr_pairs_mask_flattened_bool = self.hla_dr_pairs_mask_flattened.bool()
        self.hla_dr_pairs_mask_flattened_bool = self.hla_dr_pairs_mask_flattened_bool.view(1, -1)
        #print(f'shape of hla_dr_pairs_mask: {hla_dr_pairs_mask.shape}, shape of flattened: {self.hla_dr_pairs_mask_flattened.shape}')

        # this freq_mask sets the infrequent coefficients to zero
        self.freq_mask_a = freq_mask_a
        self.freq_mask_b = freq_mask_b
        self.freq_mask_dr = freq_mask_dr
        #print('---------------------------------------------------')


        

        if  not self.interaction and not self.lsm_penalty: 
            optimizer = optim.Adam([self.estimated_w], lr=learning_rate)
            #print('this model only optimizes NLL')
        elif self.interaction and not self.lsm_penalty:
            optimizer = optim.Adam([self.estimated_w, self.estimated_Va, self.estimated_Vb, self.estimated_Vdr], lr=learning_rate)
        elif self.interaction and self.lsm_penalty: 
            #print('this model considers Latent distance position of HLA types')
            if self.ls_penalty_type == 'lowRank':
                optimizer = optim.Adam([self.estimated_w, self.estimated_Va, self.estimated_Vb, self.estimated_Vdr,self.estimated_Za_don,self.estimated_Za_rec,self.estimated_Zb_don,self.estimated_Zb_rec, self.estimated_Zdr_don,self.estimated_Zdr_rec], lr=learning_rate)
            if self.ls_penalty_type == 'latent_distance':
                print('latent space penlaty optimization')
                optimizer = optim.Adam([self.estimated_w, self.estimated_Va, self.estimated_Vb, self.estimated_Vdr,self.estimated_Za_don,self.estimated_Za_rec,self.estimated_Zb_don,self.estimated_Zb_rec, self.estimated_Zdr_don,self.estimated_Zdr_rec, self.estimated_alpha_0_a, self.estimated_alpha_0_b, self.estimated_alpha_0_dr], lr=learning_rate)

        else:
            raise ValueError('Choose a correct combination of self.interaction and self.rrr')
        
        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(num_epochs):
            #self.like.append(-1*self._negative_log_likelihood(X,y))
        
            #self._data_collection() #only when the interaction is considered!
            optimizer.zero_grad()
            train_loss = self._loss_freq(X_train, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train,
            survival_censoring_time_train, events_train)
            #val_loss =   self._loss(X_val  , X_val_interaction  , survival_censoring_time_val  , events_val)
            self.train_losses.append(train_loss)
            train_loss.backward()
            optimizer.step()
            # Explicitly release GPU memory
            torch.cuda.empty_cache()

            #adding early stopping
            if X_val != None and hla_a_pairs_val != None and hla_b_pairs_val!=None and hla_dr_pairs_val!= None and survival_censoring_time_val!= None and events_val != None:
                #X_val, X_val_interaction, survival_censoring_time_val, events_val  = X_val.to(self.device), X_val_interaction.to(self.device), survival_censoring_time_val.to(self.device), events_val.to(self.device)

                with torch.no_grad():
                    val_loss = self._loss(X_val, hla_a_pairs_val, hla_b_pairs_val, hla_dr_pairs_val,
            survival_censoring_time_val, events_val)
                #print(val_loss)
                self.val_losses.append(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= ES_threshold:
                    print(f"Early stopping at epoch {epoch} with patience {ES_threshold}")
                    break  # This break is inside the if statement
            if (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch + 1}")
                
        return self.estimated_w, self.estimated_Va, self.estimated_Vb, self.estimated_Vdr
    

    def compute_baseline_hazard_cox_lsm(self,
        model,
        X_train, hla_a_train, hla_b_train, hla_dr_train,
        T_train, E_train
    ):
        """
        Estimate the baseline hazard H0(t) for your CoxPH_LSM model using Breslow's method.
        - model: your CoxPH_LSM instance (already fit)
        - (X_train, hla_a_train, hla_b_train, hla_dr_train): training data features
        - T_train, E_train: survival times and event indicators (1=event, 0=censor) for training set
        
        Returns:
        unique_event_times: 1D tensor of distinct event times in ascending order
        cum_baseline: 1D tensor of the same length => cumulative baseline hazard
        """
        device = X_train.device

        # 1) Sort by ascending time
        order = torch.argsort(T_train)
        T_sorted = T_train[order]
        E_sorted = E_train[order]

        # Let's build a "hazard" for each subject => h_i = model._hazard(...)
        # We'll do a loop or a direct vector approach. Then risk_i = exp(h_i).
        # We'll define a small function to get h_i for subject i:

        def get_hazard_for_subject(idx):
            # Single-subject input
            x_ = X_train[idx].unsqueeze(0)  # shape [1, p]
            a_ = hla_a_train[idx].unsqueeze(0)
            b_ = hla_b_train[idx].unsqueeze(0)
            dr_ = hla_dr_train[idx].unsqueeze(0)
            # print(x_.shape)
            # print(a_.shape)
            # print(b_.shape)
            # print(dr_.shape)
            
            # The model's hazard is a 1D tensor => shape [1], we want just the scalar
            h_i = model._hazard(
                x_,
                a_,
                b_,
                dr_,
                model.estimated_w,
                model.estimated_Va,
                model.estimated_Vb,
                model.estimated_Vdr
            )[0]
            return h_i

        # Build hazard array in sorted order
        n = len(T_train)
        h_sorted = torch.zeros(n, dtype=torch.float32, device=device)
        for i in range(n):
            idx = order[i]
            h_sorted[i] = get_hazard_for_subject(idx)

        # hazard => risk = exp(h)
        risk_sorted = torch.exp(h_sorted)

        # We'll do a running sum of the risk in reverse to get the risk set sum
        cumsum = torch.zeros(n, dtype=torch.float32, device=device)
        cumsum[n-1] = risk_sorted[n-1]
        for i in reversed(range(n-1)):
            cumsum[i] = cumsum[i+1] + risk_sorted[i]

        # 2) Breslow approach:
        unique_times_list = []
        cum_baseline_list = []
        current_H0 = 0.0

        i = 0
        while i < n:
            if E_sorted[i] == 1:
                # There's an event at T_sorted[i]
                current_time = T_sorted[i].item()
                # Count how many events at exactly this time
                count_events = 1
                j = i+1
                while j<n and abs(T_sorted[j].item() - current_time)<1e-12 and E_sorted[j]==1:
                    count_events += 1
                    j+=1

                # sum of risk among the risk set => cumsum[i]
                denom = cumsum[i].item()
                # increment
                delta = count_events/denom
                current_H0 += delta

                unique_times_list.append(current_time)
                cum_baseline_list.append(current_H0)

                i = j  # skip over multiple events at this same time
            else:
                i+=1

        if len(unique_times_list)==0:
            # Means no events => baseline hazard is 0
            return torch.empty(0), torch.empty(0)

        unique_event_times = torch.tensor(unique_times_list, dtype=torch.float32, device=device)
        cum_baseline = torch.tensor(cum_baseline_list, dtype=torch.float32, device=device)
        return unique_event_times, cum_baseline




    def predict_survival_cox_lsm(self,
        model,
        t,
        x, x_a, x_b, x_dr,
        unique_times, cum_baseline
    ):
        """
        For a single subject's feature data (x, x_a, x_b, x_dr),
        compute S(t | x) = exp( - e^{h(x)} * H0(t) ), stepwise approach.
        
        'h(x)' is your model._hazard(...) [the linear predictor].
        'risk' = exp(h(x)).
        'H0(t)' is from (unique_times, cum_baseline).

        Return a float survival probability at time t.
        """
        if len(unique_times)==0:
            return 1.0  # no events in training => trivial

        # hazard => h(x)
        with torch.no_grad():
            # shape [1, ...]
            h_val = model._hazard(
                x.unsqueeze(0),
                x_a.unsqueeze(0),
                x_b.unsqueeze(0),
                x_dr.unsqueeze(0),
                model.estimated_w,
                model.estimated_Va,
                model.estimated_Vb,
                model.estimated_Vdr
            )[0].item()
        risk = math.exp(h_val)

        # find largest j s.t. unique_times[j] <= t
        if t < unique_times[0].item():
            H0_t = 0.0
        else:
            mask = (unique_times <= t)
            idxs = mask.nonzero(as_tuple=True)[0]
            if len(idxs)==0:
                H0_t = 0.0
            else:
                j = idxs[-1].item()
                H0_t = cum_baseline[j].item()

        surv = math.exp(- risk * H0_t)
        return surv




    def predict_survival_curve_cox_lsm(self,
            model,
            # TRAIN set (for baseline hazard)
            X_train, A_train, B_train, DR_train, T_train, E_train,
            # EVAL set (for which we want the survival curves)
            X_eval, A_eval, B_eval, DR_eval,
            t_min=0.0,
            t_max=None,
            num_points=5
        ):
        """
        Compute survival probabilities S(t | x_i) for *each subject* i in X_eval
        across time points t in [t_min, t_max].

        Returns:
        times: 1D tensor [num_points], the time grid
        surv_matrix: 2D tensor [N_eval, num_points], where
                    surv_matrix[i, j] = S(times[j] | X_eval[i], A_eval[i], ...)
        ----------------------------------------------------------------------------
        This matches the approach used in integrated_brier_score_cox_lsm:
        1) Compute baseline hazard using the entire training set (Breslow).
        2) Decide t_max (if None, pick 90th percentile of T_train where E=1).
        3) For each subject i in X_eval:
            - for each time in time grid:
                compute S(t | x_i) by calling predict_survival_cox_lsm.
            - store into a row of 'surv_matrix'.
        4) Return (times, surv_matrix).
        """

        device = X_train.device
        N_eval = len(X_eval)

        # -----------------------------
        # 1) Compute or infer t_max
        # -----------------------------
        if t_max is None:
            event_mask = (E_train == 1)
            if event_mask.sum() == 0:
                # no events => trivial
                times = torch.linspace(t_min, t_min + 1, steps=num_points, device=device)
                surv_matrix = torch.ones((N_eval, num_points), device=device)
                return times, surv_matrix

            # pick 90th percentile of observed event times
            T_train_tensor = (T_train if isinstance(T_train, torch.Tensor)
                            else torch.tensor(T_train, dtype=torch.float32, device=device))
            E_train_tensor = (E_train if isinstance(E_train, torch.Tensor)
                            else torch.tensor(E_train, dtype=torch.float32, device=device))
            t_max = torch.quantile(T_train_tensor[E_train_tensor == 1], 0.95).item()

        # -----------------------------
        # 2) Compute baseline hazard
        # -----------------------------
        unique_times, cum_baseline = self.compute_baseline_hazard_cox_lsm(
            model,
            X_train, A_train, B_train, DR_train,
            T_train, E_train
        )

        # -----------------------------
        # 3) Build the time grid
        # -----------------------------
        times = torch.linspace(t_min, t_max, steps=num_points, device=device)

        # -----------------------------
        # 4) For each subject in X_eval, compute S(t|x_i)
        # -----------------------------
        surv_matrix = torch.zeros((N_eval, num_points), dtype=torch.float32, device=device)

        for i in range(N_eval):
            # We'll pass a *single subject* to predict_survival_cox_lsm
            # so that it lines up with how the IBS loop works.
            x_i   = X_eval[i]
            a_i   = A_eval[i]
            b_i   = B_eval[i]
            dr_i  = DR_eval[i]

            # for each time point
            row_surv = []
            for t_val in times:
                s_prob = self.predict_survival_cox_lsm(
                    model,
                    t_val.item(),
                    x_i, a_i, b_i, dr_i,
                    unique_times, cum_baseline
                )
                row_surv.append(s_prob)

            # store in surv_matrix row
            surv_matrix[i, :] = torch.tensor(row_surv, device=device)

        # Done: times is shape [num_points], surv_matrix is [N_eval, num_points]
        return times, surv_matrix


        



##############################################
##############################################
##############################################

    def brier_score_cox_time_lsm(self,
        model,
        t,
        # data for evaluation
        X_eval, A_eval, B_eval, DR_eval,
        T_eval, E_eval,
        # baseline hazard arrays
        unique_times, cum_baseline,
        # censor-KM arrays
        censor_times, censor_survival
    ):
        """
        Compute Brier Score at time t for your CoxPH_LSM model.
        We do the IPCW approach:
        BS(t) = 1/N * sum over i of ...
            if T_i <= t & E_i=1 => (S(t|x_i))^2 / G(T_i^-)
            if T_i > t          => (1 - S(t|x_i))^2 / G(t)
            if T_i <= t & E_i=0 => 0
        """
        N = len(T_eval)
        total = 0.0

        for i in range(N):
            Ti = T_eval[i].item()
            Ei = E_eval[i].item()
            # predict S(t|x_i)
            si_t = self.predict_survival_cox_lsm(
                model,
                t,
                X_eval[i], A_eval[i], B_eval[i], DR_eval[i],
                unique_times, cum_baseline
            )
            if Ti <= t:
                if Ei == 1:
                    # event by t => label is 0 => error = (0 - si_t)^2
                    Gval = self.get_G_value_lsm(censor_times, censor_survival, Ti, minus=True)
                    if Gval < 1e-12: Gval = 1e-12
                    total += (si_t**2)/Gval
                else:
                    # censored before t => no term
                    pass
            else:
                # T_i>t => label is 1 => error = (1 - si_t)^2
                Gval_t = self.get_G_value_lsm(censor_times, censor_survival, t, minus=False)
                if Gval_t<1e-12: Gval_t=1e-12
                total += ((1.0 - si_t)**2)/Gval_t

        return total / N
    def integrated_brier_score_cox_lsm(self,
        model,
        # TRAIN set (for baseline hazard + censor-KM)
        X_train, A_train, B_train, DR_train, T_train, E_train,
        # EVAL set (where we measure Brier Score)
        X_eval, A_eval, B_eval, DR_eval, T_eval, E_eval,
        t_min=0.0,
        t_max=None,
        num_points=20
    ):
        """
        All-in-one function:
        1) compute baseline hazard from training set => (unique_times, cum_baseline)
        2) compute censor-KM from training set => (censor_times, censor_survival)
        3) evaluate Brier Score at times t_min..t_max on the *EVAL* set
        4) integrate => IBS
        Returns: IBS (scalar float)
        """
        device = X_train.device
        if t_max is None:
            # pick max event time from E_eval
            event_mask = (E_eval==1)
            if event_mask.sum()==0:
                return 0.0  # no events => trivial
            # t_max = T_eval[event_mask].max().item()
            # t_max = min(torch.quantile(T_eval[E_eval == 1], 0.95).item(), 3000)

            # t_max = torch.quantile(T_eval[E_eval == 1], 0.95).item()

            T_eval_tensor = torch.tensor(T_eval, dtype=torch.float32) if isinstance(T_eval, np.ndarray) else T_eval
            E_eval_tensor = torch.tensor(E_eval, dtype=torch.float32) if isinstance(E_eval, np.ndarray) else E_eval

            t_max = torch.quantile(T_eval_tensor[E_eval_tensor == 1], 0.95).item()
        # 1) Baseline hazard
        unique_times, cum_baseline = self.compute_baseline_hazard_cox_lsm(
            model,
            X_train, A_train, B_train, DR_train,
            T_train, E_train
        )

        # 2) Censor-KM
        censor_times, censor_survival = self.compute_censor_km_lsm(T_train, E_train)

        # 3) Evaluate Brier Score on a grid
        times = torch.linspace(t_min, t_max, steps=num_points, device=device)
        bs_vals = []
        self.t_ls = []
        for i in range(len(times)):
            t_val = times[i].item()
            bs_t = self.brier_score_cox_time_lsm(
                model,
                t_val,
                X_eval, A_eval, B_eval, DR_eval,
                T_eval, E_eval,
                unique_times, cum_baseline,
                censor_times, censor_survival
            )
            bs_vals.append(bs_t)
            self.t_ls.append(t_val)
        # 4) trapezoid integration
        ibs = 0.0
        for i in range(len(times)-1):
            w_ = (times[i+1] - times[i]).item()
            avg_ = 0.5*(bs_vals[i] + bs_vals[i+1])
            ibs += (w_*avg_)

        # divide by (t_max - t_min) => standard IBS
        ibs = ibs/(t_max - t_min)
        self.bs_list = bs_vals
        return ibs
        


    def compute_censor_km_lsm(self,T, E):
        """
        Build the KM curve for censoring distribution from training set
        removing subjects from risk set at their event time.
        
        T: [N] times
        E: [N], 1=event, 0=censor
        """
        device = T.device
        data = [(T[i].item(), E[i].item()) for i in range(len(T))]
        data.sort(key=lambda x: x[0])  # ascending time

        at_risk = len(data)
        current_surv = 1.0
        censor_times_list = []
        censor_surv_list = []

        last_time = None
        n_censor_here = 0
        n_total_here = 0

        for (time_i, e_i) in data:
            if last_time is None:
                last_time = time_i
                if e_i==0:
                    n_censor_here=1
                else:
                    n_censor_here=0
                n_total_here=1
            else:
                if abs(time_i - last_time)<1e-12:
                    # same time => accumulate
                    if e_i==0:
                        n_censor_here+=1
                    n_total_here+=1
                else:
                    # finalize the old time
                    dq = n_censor_here / at_risk
                    current_surv *= (1.0 - dq)
                    censor_times_list.append(last_time)
                    censor_surv_list.append(current_surv)

                    at_risk -= n_total_here
                    last_time = time_i
                    n_censor_here = 1 if e_i==0 else 0
                    n_total_here = 1

        # finalize
        if n_total_here>0 and at_risk>0:
            dq = n_censor_here/at_risk
            current_surv *= (1.0 - dq)
            censor_times_list.append(last_time)
            censor_surv_list.append(current_surv)
            at_risk -= n_total_here

        censor_times = torch.tensor(censor_times_list, dtype=torch.float32, device=device)
        censor_survival = torch.tensor(censor_surv_list, dtype=torch.float32, device=device)
        return censor_times, censor_survival


    def get_G_value_lsm(self,censor_times, censor_survival, t, minus=False):
        """
        Return G(t) or G(t^-), from the piecewise censor-KM.
        """
        if len(censor_times)==0:
            return 1.0
        if t < censor_times[0].item():
            return 1.0
        mask = (censor_times <= t)
        idxs = mask.nonzero(as_tuple=True)[0]
        if len(idxs)==0:
            return 1.0

        idx = idxs[-1].item()
        if minus and abs(t - censor_times[idx].item())<1e-12:
            idx-=1
            if idx<0:
                return 1.0
        return censor_survival[idx].item()

