import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt
import copy
import os
import torch.nn.init as init
from torch.utils.data import TensorDataset, DataLoader
from utils import GridSearch
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from itertools import chain





manual_seed = 43421654
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)

class CoxPH(nn.Module):
    def __init__(self, estimated_w, regularization=True,
                    alpha = 0.5, kappa = 0.5, verbose=False):
        super(CoxPH, self).__init__()

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
        

        self.verbose = verbose

        #whether we want to do regularization or not
        self.regularization = regularization

        # alpha is regularization coefficient for vector of weights (w: individual features)
        self.alpha = alpha
        #kappa: L1 regularization coefficient
        self.kappa = kappa
        self.train_losses = []
        self.val_losses = []

    

    



    def _reset_parameters(self):
        # Reset estimated_w to its initial value
        
        self.estimated_w.data = self.initial_w.data.clone()

    def _l2_penalty(self):
            return torch.sum(self.estimated_w**2)
        
    def _l1_penalty(self):
            return torch.sum(torch.abs(self.estimated_w))



    def _hazard(self, X, estimated_w):
        '''
        This function calculates the linear combination of the input features W'(w transpose)X
        V is flattened
        This calcualtes W'X
        '''

        h = torch.matmul(X, estimated_w)

        return h




    def predict(self, X, estimated_w):
        '''This Calculates exp(W'X)'''
        
        #h = W'X
        hazard = self._hazard(X, estimated_w)
        return hazard


    def _negative_partial_log_likelihood(self, X,survival_censoring_time, events):
        #X.w + X_interaction.v --> (n,p).(p,1) + (n,p^2).(p^2,1) ---> (n,1)
        # 
        
        
        h = self._hazard(X, self.estimated_w)
        #g is equal to the gumble noise

        # we sort all the tensors based on the order of survival or censoring time
        _, indices = torch.sort(survival_censoring_time)

        # sorting events based on the order of survival or censoring time
        events_sorted = events[indices]

        # sorting h based on the order of survival or censoring time
        h_sorted = h[indices]
        #print('h sorted', h_sorted)
        h_sorted_reversed = torch.flip(h_sorted, dims=[0])
        logcumsumexp = torch.logcumsumexp(h_sorted_reversed,dim=0)
        logcumsumexp_reversed = torch.flip(logcumsumexp,dims=[0])

        risk_set_sizes = torch.sum(events_sorted == 1, dim=0)



        neg_log_partial_likelihood = -1 * torch.sum(  (h_sorted - logcumsumexp_reversed)[events_sorted] )/risk_set_sizes


        return neg_log_partial_likelihood


    def _loss(self,X, survival_censoring_time, events):
        if not self.regularization:
            return self._negative_partial_log_likelihood(X, survival_censoring_time, events)
        elif self.regularization:
            return self._negative_partial_log_likelihood(X, survival_censoring_time, events) + self.alpha * self._l2_penalty() + self.kappa * self._l1_penalty()

    
    def fit(self, X_train, 
            survival_censoring_time_train, events_train,
            X_val, 
            survival_censoring_time_val, events_val, 
            learning_rate=0.005, num_epochs=5000, ES_threshold = 20):
        '''X_train: data_basic + MM + hla types'''



        optimizer = optim.AdamW([self.estimated_w], lr=learning_rate)      

        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(num_epochs):
            #self.like.append(-1*self._negative_log_likelihood(X,y))
        
            optimizer.zero_grad()
            train_loss = self._loss(X_train, survival_censoring_time_train, events_train)
            #val_loss =   self._loss(X_val    , survival_censoring_time_val  , events_val)
            self.train_losses.append(train_loss)
            train_loss.backward()
            optimizer.step()
            self.estimated_w.data = torch.sign(self.estimated_w.data) * torch.max(torch.zeros_like(self.estimated_w), torch.abs(self.estimated_w.data) -  self.kappa)
        
            #adding early stopping
            if X_val != None  and survival_censoring_time_val!= None and events_val != None:
                #X_val, survival_censoring_time_val, events_val  = X_val.to(self.device), survival_censoring_time_val.to(self.device), events_val.to(self.device)

                with torch.no_grad():
                    val_loss = self._loss(X_val,
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
                
        return self.estimated_w
    
    

    def compute_baseline_hazard_cox(self,X_train, T_train, E_train, w):
        """
        Estimate the baseline hazard H0(t) using Breslow's method, from the *training set*.
        Inputs:
        X_train: [Ntrain, p] float tensor
        T_train: [Ntrain] times (event or censor)
        E_train: [Ntrain] binary (1=event, 0=censor)
        w:       [p] the learned Cox coefficients
        Returns:
        unique_event_times: 1D tensor of distinct event times in ascending order
        cum_baseline: 1D tensor of the same length => cumulative baseline hazard at each of those times
        """
        # 1) Sort by ascending time
        order = torch.argsort(T_train)
        T_sorted = T_train[order]
        E_sorted = E_train[order]
        X_sorted = X_train[order]

        # 2) hazard contributions
        # hazard[i] = exp(w^T x_i)
        hazards = torch.exp(X_sorted @ w)  # shape [Ntrain]

        # We'll maintain a running sum in reverse so that at index i, cumsum[i] = sum of hazards for all j>=i
        n = len(T_train)
        cumsum = torch.zeros(n, dtype=torch.float32, device=X_train.device)
        cumsum[n-1] = hazards[n-1]
        for i in reversed(range(n-1)):
            cumsum[i] = cumsum[i+1] + hazards[i]

        unique_times_list = []
        cum_baseline_list = []

        current_H0 = 0.0

        i = 0
        while i < n:
            if E_sorted[i] == 1:
                # an event at T_sorted[i]
                current_time = T_sorted[i].item()
                # how many events occur EXACTLY at this time
                count_events = 1
                j = i+1
                while j<n and abs(T_sorted[j].item()-current_time)<1e-12 and E_sorted[j]==1:
                    count_events += 1
                    j+=1

                # sum of risk for that time is cumsum[i]
                # Breslow increment
                delta = count_events / cumsum[i].item()

                current_H0 += delta
                unique_times_list.append(current_time)
                cum_baseline_list.append(current_H0)

                i = j  # skip past all events at this time
            else:
                i+=1

        if len(unique_times_list)==0:
            return (torch.empty(0), torch.empty(0))

        device = X_train.device
        unique_event_times = torch.tensor(unique_times_list, dtype=torch.float32, device=device)
        cum_baseline = torch.tensor(cum_baseline_list, dtype=torch.float32, device=device)
        return unique_event_times, cum_baseline


    def predict_survival_cox(self,t, x, w, unique_times, cum_baseline):
        """
        Predict S(t | x) = exp( - e^{w^T x} * H0(t) ), using stepwise baseline hazard.
        Inputs:
        t: scalar float
        x: [p]  features
        w: [p]  coefficients
        unique_times: 1D tensor of event times
        cum_baseline: 1D tensor, same length => cumulative hazard
        """
        if len(unique_times)==0:
            return 1.0  # no events => trivial

        linear_pred = torch.dot(x, w).item()
        risk = math.exp(linear_pred)

        # find largest j where unique_times[j] <= t
        # if t < unique_times[0], H0(t)=0
        if t < unique_times[0].item():
            H0_t = 0.0
        else:
            mask = (unique_times <= t)
            idxs = mask.nonzero(as_tuple=True)[0]
            if len(idxs)==0:
                H0_t=0.0
            else:
                j = idxs[-1].item()
                H0_t = cum_baseline[j].item()

        # survival = exp(- risk * H0(t))
        surv = math.exp(- risk * H0_t)
        return surv


    def compute_censor_km(self,T, E):
        """
        Build the KM curve for the censoring distribution from *training set*,
        removing subjects from the risk set once they have the event.
        T: [N]
        E: [N]  1=event, 0=censor
        Returns:
        censor_times:   sorted times at which a censor occurs
        censor_survival: G(t) at those times
        """
        device = T.device
        data = [(T[i].item(), E[i].item()) for i in range(len(T))]
        data.sort(key=lambda x: x[0])

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
                    # same time
                    if e_i==0:
                        n_censor_here+=1
                    n_total_here+=1
                else:
                    # finalize old time
                    dq = n_censor_here/at_risk
                    current_surv *= (1.0 - dq)
                    censor_times_list.append(last_time)
                    censor_surv_list.append(current_surv)

                    at_risk -= n_total_here
                    last_time = time_i
                    if e_i==0:
                        n_censor_here=1
                    else:
                        n_censor_here=0
                    n_total_here=1

        if n_total_here>0 and at_risk>0:
            dq = n_censor_here / at_risk
            current_surv *= (1.0 - dq)
            censor_times_list.append(last_time)
            censor_surv_list.append(current_surv)
            at_risk -= n_total_here

        censor_times = torch.tensor(censor_times_list, dtype=torch.float32, device=device)
        censor_survival = torch.tensor(censor_surv_list, dtype=torch.float32, device=device)
        return censor_times, censor_survival


    def get_G_value(self,censor_times, censor_survival, t, minus=False):
        """
        Return G(t) or G(t^-) from the piecewise censor-KM.
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


    def brier_score_cox_time(self,
        t,
        X_eval, T_eval, E_eval,
        w,
        unique_times, cum_baseline,
        censor_times, censor_survival
    ):
        """
        Single-time Brier Score for Cox model at time t.
        - X_eval: [N_eval, p]
        - T_eval: [N_eval]
        - E_eval: [N_eval]
        - w: [p]
        - unique_times, cum_baseline => baseline hazard from training
        - censor_times, censor_survival => censor-KM from training
        """
        N = len(T_eval)
        total = 0.0

        for i in range(N):
            Ti = T_eval[i].item()
            Ei = E_eval[i].item()
            surv_i = self.predict_survival_cox(t, X_eval[i], w, unique_times, cum_baseline)

            if Ti <= t:
                if Ei==1:
                    # event by t => true label "0" => squared error (0 - S_i)^2
                    Gval = self.get_G_value(censor_times, censor_survival, Ti, minus=True)
                    if Gval<1e-12: Gval=1e-12
                    total += (surv_i**2)/Gval
                else:
                    # censored before t => no contribution
                    pass
            else:
                # T_i>t => alive at t => label "1"
                Gval_t = self.get_G_value(censor_times, censor_survival, t, minus=False)
                if Gval_t<1e-12: Gval_t=1e-12
                total += ((1.0 - surv_i)**2)/Gval_t

        return total / N


    def integrated_brier_score_cox(self,
        X_train, T_train, E_train,
        X_eval,  T_eval,  E_eval,
        w,
        t_min=0.0,
        t_max=None,
        num_points=20
    ):
        """
        All-in-one function:
        1) compute baseline hazard from (X_train,T_train,E_train,w)
        2) compute censor-KM from (T_train,E_train)
        3) evaluate Brier Score at times from t_min..t_max on (X_eval,T_eval,E_eval)
        4) trapezoid integrate => IBS
        """
        device = X_train.device
        if t_max is None:
            # pick max event time from training or from E_eval => your choice
            # common is to pick from *test events* or overall
            event_mask = (E_eval==1)
            if event_mask.sum()==0:
                return 0.0  # no events => trivial
            t_max = T_eval[event_mask].max().item()

        # 1) Baseline hazard from training
        unique_times, cum_baseline = self.compute_baseline_hazard_cox(X_train, T_train, E_train, w)

        # 2) Censor-KM from training
        censor_times, censor_survival = self.compute_censor_km(T_train, E_train)

        # 3) Evaluate on grid
        times = torch.linspace(t_min, t_max, steps=num_points, device=device)
        bs_vals = []
        for i in range(len(times)):
            t_val = times[i].item()
            bs_t = self.brier_score_cox_time(
                t_val,
                X_eval, T_eval, E_eval,
                w,
                unique_times, cum_baseline,
                censor_times, censor_survival
            )
            bs_vals.append(bs_t)

        # 4) trapezoid integral
        ibs = 0.0
        for i in range(len(times)-1):
            width = (times[i+1] - times[i]).item()
            avg_height = 0.5*(bs_vals[i] + bs_vals[i+1])
            ibs += (width * avg_height)

        # optionally divide by (t_max - t_min) => standard IBS
        ibs = ibs/(t_max - t_min)

        return ibs



