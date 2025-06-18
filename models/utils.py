from itertools import product
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
import gc
import os
try:
    from lifelines.utils import concordance_index
except:
    pass
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

manual_seed = 21123334
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)


class GridSearch():

    def __init__(self, num_folds = 2, epochs = 2000, learning_rate = 0.1, ES_threshold=40, batch_size = 10000):
        
        self.num_folds = num_folds
        self.lr = learning_rate
        self.num_epochs = epochs
        self.ES_threshold = ES_threshold
        self.batch_size = batch_size

    def custom_grid_search_mse(self, model, param_grid, X, X_interaction, y):
        #best_score = float('-inf') if self.eval_metric is None else float('inf')
        best_score = float('inf')
        best_estimator = None
        best_hyperparameters = None

        kf = KFold(n_splits= self.num_folds, shuffle=True, random_state=42)

        # Generate all combinations of hyperparameters
        all_hyperparameter_combinations = list(product(*param_grid.values()))

        # Check if TPU is available
        if "TPU_NAME" in os.environ:
            device = torch.device("xla")  # XLA is PyTorch's TPU device

        else:
            # Check if GPU is available
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
    
        
        for hyperparameters in all_hyperparameter_combinations:
            hyperparameter_dict = dict(zip(param_grid.keys(), hyperparameters))
            #print(hyperparameter_dict)

            # Update model attributes with hyperparameters
            for key, value in hyperparameter_dict.items():
                setattr(model, key, value)

            #print(model.alpha)
            #print(model.gamma)
            scores = []
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                X_train_interaction, X_val_interaction = X_interaction[train_idx], X_interaction[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                #print(f'shape of training', X_train.shape)
                #print(f'shape of validation', X_val.shape)


        
                estimated_w,estimated_V = model.fit(X_train = X_train, X_train_interaction = X_train_interaction, y_train=y_train,
                                X_val = None, X_val_interaction=None, 
                                y_val=None, learning_rate=self.lr, num_epochs=self.num_epochs, ES_threshold = self.ES_threshold)
                #y_pred = model.predict(X_val)
                y_pred = model.predict(X_val, X_val_interaction, estimated_w, estimated_V)


                score = self.eval_metric_performance_mse(y_val, y_pred)


                scores.append(score)
                # Release GPU memory after each fold's training and evaluation
                del X_train, X_val, X_train_interaction, X_val_interaction, y_train, y_val, estimated_w, estimated_V
                gc.collect()
                break
            
            #print('list of scores in cross validation',scores)
            avg_score = np.mean(scores)

            if avg_score < best_score:
                best_score = avg_score
                # Create a new instance of the model with the best hyperparameters
                #best_estimator = model.__class__(**hyperparameter_dict)
                best_hyperparameters = hyperparameter_dict
            #resetting the model parameters
            model._reset_parameters()
        for key, value in best_hyperparameters.items():
                setattr(model, key, value)
        print('best chosen hyperparameterss', best_hyperparameters)
        #print('------------------------------------')
        return model, best_hyperparameters
    
    def custom_grid_search_cph(self, model, param_grid, X, hla_a_pairs, hla_b_pairs,hla_dr_pairs, 
                            hla_a_pairs_mask, hla_b_pairs_mask, hla_dr_pairs_mask, events, survival_censoring_time):
        best_score = float('-inf')
        best_estimator = None
        best_hyperparameters = None

        kf = KFold(n_splits= self.num_folds, shuffle=True, random_state=42)

        # Generate all combinations of hyperparameters
        all_hyperparameter_combinations = list(product(*param_grid.values()))

        # Check if TPU is available
        if "TPU_NAME" in os.environ:
            device = torch.device("xla")  # XLA is PyTorch's TPU device

        else:
            # Check if GPU is available
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        for hyperparameters in all_hyperparameter_combinations:
            hyperparameter_dict = dict(zip(param_grid.keys(), hyperparameters))
            print(hyperparameter_dict)

            # Update model attributes with hyperparameters
            for key, value in hyperparameter_dict.items():
                setattr(model, key, value)

            fold_number = 0
            scores = []
            scores_C = []
            for train_idx, val_idx in kf.split(X):
                fold_number += 1
                print('fold ', fold_number)
                X_train, X_val = X[train_idx], X[val_idx]
                hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train = hla_a_pairs[train_idx], hla_b_pairs[train_idx], hla_dr_pairs[train_idx]
                hla_a_pairs_val, hla_b_pairs_val, hla_dr_pairs_val = hla_a_pairs[val_idx], hla_b_pairs[val_idx], hla_dr_pairs[val_idx]
                events_train, events_val = events[train_idx], events[val_idx]
                survival_censoring_time_train, survival_censoring_time_val = survival_censoring_time[train_idx], survival_censoring_time[val_idx]
                

                # resetting the initial weights
                model._reset_parameters()
                estimated_w, estimated_Va, estimated_Vb, estimated_Vdr = model.fit(X_train=X_train, 
                    hla_a_pairs_train=hla_a_pairs_train, hla_b_pairs_train=hla_b_pairs_train, hla_dr_pairs_train=hla_dr_pairs_train,
                    survival_censoring_time_train=survival_censoring_time_train, events_train=events_train,
                    X_val=X_val, hla_a_pairs_val=hla_a_pairs_val, hla_b_pairs_val=hla_b_pairs_val, hla_dr_pairs_val=hla_dr_pairs_val, 
                    survival_censoring_time_val=survival_censoring_time_val, events_val=events_val, 
                    hla_a_pairs_mask=hla_a_pairs_mask, hla_b_pairs_mask=hla_b_pairs_mask, hla_dr_pairs_mask=hla_dr_pairs_mask,
                    learning_rate=self.lr, num_epochs=self.num_epochs, ES_threshold = self.ES_threshold)
                

                partial_hazard = model.predict(X_val, hla_a_pairs_val, hla_b_pairs_val, hla_dr_pairs_val,
                estimated_w, estimated_Va, estimated_Vb, estimated_Vdr)

                score_C = self.eval_metric_performance_cph(survival_censoring_time_val,partial_hazard, events_val)
                npll = model._negative_partial_log_liklihood(X_val, hla_a_pairs_val, hla_b_pairs_val, hla_dr_pairs_val, survival_censoring_time_val, events_val)
                score = -npll # score is partial log likelihood

                scores.append(score.detach().cpu().numpy())
                scores_C.append(score_C)
                # Release GPU memory after each fold's training and evaluation
                del X_train, X_val, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train, hla_a_pairs_val, hla_b_pairs_val, hla_dr_pairs_val, events_train, events_val, survival_censoring_time_train, survival_censoring_time_val, estimated_w, estimated_Va, estimated_Vb, estimated_Vdr
                gc.collect()
                torch.cuda.empty_cache()
                
                
            
            avg_score = np.mean(scores)
            avg_score_C = np.mean(scores_C)
            print('average score of k-fold CV:', avg_score)

            if avg_score > best_score:
                best_score = avg_score
                best_score_C = avg_score_C
                # Create a new instance of the model with the best hyperparameters
                #best_estimator = model.__class__(**hyperparameter_dict)
                best_hyperparameters = hyperparameter_dict

        for key, value in best_hyperparameters.items():
                setattr(model, key, value)
        print('best chosen hyperparameterss', best_hyperparameters)

        print('------------------------------------')
        print('best C_index related to the best partial likelihood:',best_score_C)
        return model, best_hyperparameters
    
    def custom_grid_search_cph_FM(self, model, param_grid, X, hla_a_pairs, hla_b_pairs,hla_dr_pairs, 
                            hla_a_pairs_mask, hla_b_pairs_mask, hla_dr_pairs_mask, events, survival_censoring_time):
        best_score = float('-inf')
        best_estimator = None
        best_hyperparameters = None

        kf = KFold(n_splits= self.num_folds, shuffle=True, random_state=42)

        # Generate all combinations of hyperparameters
        all_hyperparameter_combinations = list(product(*param_grid.values()))

        # Check if TPU is available
        if "TPU_NAME" in os.environ:
            device = torch.device("xla")  # XLA is PyTorch's TPU device

        else:
            # Check if GPU is available
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        for hyperparameters in all_hyperparameter_combinations:
            hyperparameter_dict = dict(zip(param_grid.keys(), hyperparameters))
            print(hyperparameter_dict)

            # Update model attributes with hyperparameters
            for key, value in hyperparameter_dict.items():
                setattr(model, key, value)

            fold_number = 0
            scores = []
            scores_C = []
            for train_idx, val_idx in kf.split(X):
                fold_number += 1
                print('fold ', fold_number)
                X_train, X_val = X[train_idx], X[val_idx]
                hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train = hla_a_pairs[train_idx], hla_b_pairs[train_idx], hla_dr_pairs[train_idx]
                hla_a_pairs_val, hla_b_pairs_val, hla_dr_pairs_val = hla_a_pairs[val_idx], hla_b_pairs[val_idx], hla_dr_pairs[val_idx]
                events_train, events_val = events[train_idx], events[val_idx]
                survival_censoring_time_train, survival_censoring_time_val = survival_censoring_time[train_idx], survival_censoring_time[val_idx]
                

                # resetting the initial weights
                model._reset_parameters()
                estimated_w, estimated_Va, estimated_Vb, estimated_Vdr = model.fit(X_train=X_train, 
                    hla_a_pairs_train=hla_a_pairs_train, hla_b_pairs_train=hla_b_pairs_train, hla_dr_pairs_train=hla_dr_pairs_train,
                    survival_censoring_time_train=survival_censoring_time_train, events_train=events_train,
                    X_val=X_val, hla_a_pairs_val=hla_a_pairs_val, hla_b_pairs_val=hla_b_pairs_val, hla_dr_pairs_val=hla_dr_pairs_val, 
                    survival_censoring_time_val=survival_censoring_time_val, events_val=events_val, 
                    hla_a_pairs_mask=hla_a_pairs_mask, hla_b_pairs_mask=hla_b_pairs_mask, hla_dr_pairs_mask=hla_dr_pairs_mask,
                    learning_rate=self.lr, num_epochs=self.num_epochs, ES_threshold = self.ES_threshold)
                

                partial_hazard = model.predict(X_val, hla_a_pairs_val, hla_b_pairs_val, hla_dr_pairs_val,
                estimated_w, estimated_Va, estimated_Vb, estimated_Vdr)

                score_C = self.eval_metric_performance_cph(survival_censoring_time_val,partial_hazard, events_val)
                npll = model._negative_partial_log_liklihood(X_val, hla_a_pairs_val, hla_b_pairs_val, hla_dr_pairs_val, survival_censoring_time_val, events_val)
                score = -npll # score is partial log likelihood

                scores.append(score.detach().cpu().numpy())
                scores_C.append(score_C)
                # Release GPU memory after each fold's training and evaluation
                del X_train, X_val, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train, hla_a_pairs_val, hla_b_pairs_val, hla_dr_pairs_val, events_train, events_val, survival_censoring_time_train, survival_censoring_time_val, estimated_w, estimated_Va, estimated_Vb, estimated_Vdr
                gc.collect()
                torch.cuda.empty_cache()
                
                
            
            avg_score = np.mean(scores)
            avg_score_C = np.mean(scores_C)
            print('average score of k-fold CV:', avg_score)

            if avg_score > best_score:
                best_score = avg_score
                best_score_C = avg_score_C
                # Create a new instance of the model with the best hyperparameters
                #best_estimator = model.__class__(**hyperparameter_dict)
                best_hyperparameters = hyperparameter_dict

        for key, value in best_hyperparameters.items():
                setattr(model, key, value)
        print('best chosen hyperparameterss', best_hyperparameters)

        print('------------------------------------')
        print('best C_index related to the best partial likelihood:',best_score_C)
        return model, best_hyperparameters
    
    def custom_grid_search_cph_real(self, model, param_grid, X, X_interaction,
                                events, survival_censoring_time):
        best_score = float('-inf')
        best_estimator = None
        best_hyperparameters = None

        kf = KFold(n_splits= self.num_folds, shuffle=True, random_state=42)

        # Generate all combinations of hyperparameters
        all_hyperparameter_combinations = list(product(*param_grid.values()))


        # Check if GPU is available
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        for hyperparameters in all_hyperparameter_combinations:
            hyperparameter_dict = dict(zip(param_grid.keys(), hyperparameters))
            print(hyperparameter_dict)

            # Update model attributes with hyperparameters
            for key, value in hyperparameter_dict.items():
                setattr(model, key, value)

            #print('d of the model:',model.d)
            
            #print(model.alpha)
            #print(model.gamma)
            fold_number = 0
            scores = []
            scores_C = []
            for train_idx, val_idx in kf.split(X):
                fold_number += 1
                X_train, X_val = X[train_idx], X[val_idx]
                X_interaction_train = X_interaction[train_idx]
                X_interaction_val = X_interaction[val_idx]                
                events_train, events_val = events[train_idx], events[val_idx]
                survival_censoring_time_train, survival_censoring_time_val = survival_censoring_time[train_idx], survival_censoring_time[val_idx]
                

                # resetting the initial weights
                model._reset_parameters()
                estimated_w, estimated_V = model.fit(X_train=X_train, 
                    X_train_interaction = X_interaction_train, 
                    survival_censoring_time_train=survival_censoring_time_train, events_train=events_train,
                    X_val=X_val, X_val_interaction = X_interaction_val, 
                    survival_censoring_time_val=survival_censoring_time_val, events_val=events_val, 
                    learning_rate=self.lr, num_epochs=self.num_epochs, ES_threshold = self.ES_threshold)
                
                partial_hazard = model.predict(X_val, X_interaction_val, estimated_w, estimated_V)

                score_C = self.eval_metric_performance_cph(survival_censoring_time_val,partial_hazard, events_val)
                npll = model._negative_partial_log_likelihood(X_val,X_interaction_val, survival_censoring_time_val, events_val)
                score = -npll # score is partial log likelihood

                scores.append(score.detach().cpu().numpy())
                scores_C.append(score_C)
                # Release GPU memory after each fold's training and evaluation
                del X_train, X_val, events_train, events_val, survival_censoring_time_train, survival_censoring_time_val, estimated_w, estimated_V
                gc.collect()
                torch.cuda.empty_cache()
                
                
            
            #print('list of scores in cross validation',scores)
            avg_score = np.mean(scores)
            avg_score_C = np.mean(scores_C)
            #print('average score of k-fold CV:', avg_score)

            if avg_score > best_score:
                best_score = avg_score
                best_score_C = avg_score_C
                # Create a new instance of the model with the best hyperparameters
                #best_estimator = model.__class__(**hyperparameter_dict)
                best_hyperparameters = hyperparameter_dict

        for key, value in best_hyperparameters.items():
                setattr(model, key, value)
        print('best chosen hyperparameterss', best_hyperparameters)
        #print('best alpha', model.alpha)
        #print('best alpha', model.kappa)
        #print('best alpha', model.gamma)
        print('------------------------------------')
        #print('best C_index related to the best partial likelihood:',best_score_C)
        return model, best_hyperparameters
    
    def custom_grid_search_cph_gumbel(self, model, param_grid, X,
                                    events, survival_censoring_time):
            best_score = float('-inf')
            best_estimator = None
            best_hyperparameters = None

            kf = KFold(n_splits= self.num_folds, shuffle=True, random_state=42)

            # Generate all combinations of hyperparameters
            all_hyperparameter_combinations = list(product(*param_grid.values()))


            # Check if GPU is available
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            for hyperparameters in all_hyperparameter_combinations:
                hyperparameter_dict = dict(zip(param_grid.keys(), hyperparameters))
                print(hyperparameter_dict)

                # Update model attributes with hyperparameters
                for key, value in hyperparameter_dict.items():
                    setattr(model, key, value)
                

                fold_number = 0
                scores = []
                scores_C = []
                for train_idx, val_idx in kf.split(X):
                    fold_number += 1
                    X_train, X_val = X[train_idx], X[val_idx]
                    events_train, events_val = events[train_idx], events[val_idx]
                    survival_censoring_time_train, survival_censoring_time_val = survival_censoring_time[train_idx], survival_censoring_time[val_idx]
                    

                    # resetting the initial weights
                    model._reset_parameters()
                    estimated_w = model.fit(X_train=X_train, X_val=X_val,
                                                        survival_censoring_time_train=survival_censoring_time_train, 
                                                        events_train=events_train,
                                                        survival_censoring_time_val=survival_censoring_time_val,
                                                        events_val=events_val, learning_rate=self.lr,
                                                        num_epochs=self.num_epochs, ES_threshold = self.ES_threshold)
                    
                    partial_hazard = model.predict(X_val, estimated_w)

                    score_C = self.eval_metric_performance_cph(survival_censoring_time_val,partial_hazard, events_val)
                    npll = model._negative_partial_log_likelihood(X_val, survival_censoring_time_val, events_val)
                    score = -npll # score is partial log likelihood

                    scores.append(score.detach().cpu().numpy())
                    scores_C.append(score_C)
                    # Release GPU memory after each fold's training and evaluation
                    del X_train, X_val, events_train, events_val, survival_censoring_time_train, survival_censoring_time_val, estimated_w
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    
                
                #print('list of scores in cross validation',scores)
                avg_score = np.mean(scores)
                avg_score_C = np.mean(scores_C)
                #print('average score of k-fold CV:', avg_score)

                if avg_score > best_score:
                    best_score = avg_score
                    best_score_C = avg_score_C
                    # Create a new instance of the model with the best hyperparameters
                    #best_estimator = model.__class__(**hyperparameter_dict)
                    best_hyperparameters = hyperparameter_dict

            for key, value in best_hyperparameters.items():
                    setattr(model, key, value)
            print('best chosen hyperparameterss', best_hyperparameters)
            #print('best alpha', model.alpha)
            #print('best alpha', model.kappa)
            #print('best alpha', model.gamma)
            print('------------------------------------')
            #print('best C_index related to the best partial likelihood:',best_score_C)
            return model, best_hyperparameters
    
    def custom_grid_search_cph_weighted(self, model, param_grid, X,
                                    events, survival_censoring_time):
            best_score = float('-inf')
            best_estimator = None
            best_hyperparameters = None

            kf = KFold(n_splits= self.num_folds, shuffle=True, random_state=42)

            # Generate all combinations of hyperparameters
            all_hyperparameter_combinations = list(product(*param_grid.values()))


            # Check if GPU is available
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            for hyperparameters in all_hyperparameter_combinations:
                hyperparameter_dict = dict(zip(param_grid.keys(), hyperparameters))
                print(hyperparameter_dict)

                # Update model attributes with hyperparameters
                for key, value in hyperparameter_dict.items():
                    setattr(model, key, value)
                

                fold_number = 0
                scores = []
                scores_C = []
                for train_idx, val_idx in kf.split(X):
                    fold_number += 1
                    X_train, X_val = X[train_idx], X[val_idx]
                    events_train, events_val = events[train_idx], events[val_idx]
                    survival_censoring_time_train, survival_censoring_time_val = survival_censoring_time[train_idx], survival_censoring_time[val_idx]
                    

                    # resetting the initial weights
                    model._reset_parameters()
                    estimated_w = model.fit(X_train=X_train, X_val=X_val,
                                                        survival_censoring_time_train=survival_censoring_time_train, 
                                                        events_train=events_train,
                                                        survival_censoring_time_val=survival_censoring_time_val,
                                                        events_val=events_val, learning_rate=self.lr,
                                                        num_epochs=self.num_epochs, ES_threshold = self.ES_threshold)
                    
                    partial_hazard = model.predict(X_val, estimated_w)

                    score_C = self.eval_metric_performance_cph(survival_censoring_time_val,partial_hazard, events_val)
                    npll = model._negative_partial_log_likelihood(X_val, survival_censoring_time_val, events_val)
                    score = -npll # score is partial log likelihood

                    scores.append(score.detach().cpu().numpy())
                    scores_C.append(score_C)
                    # Release GPU memory after each fold's training and evaluation
                    del X_train, X_val, events_train, events_val, survival_censoring_time_train, survival_censoring_time_val, estimated_w
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    
                
                #print('list of scores in cross validation',scores)
                avg_score = np.mean(scores)
                avg_score_C = np.mean(scores_C)
                #print('average score of k-fold CV:', avg_score)

                if avg_score > best_score:
                    best_score = avg_score
                    best_score_C = avg_score_C
                    # Create a new instance of the model with the best hyperparameters
                    #best_estimator = model.__class__(**hyperparameter_dict)
                    best_hyperparameters = hyperparameter_dict

            for key, value in best_hyperparameters.items():
                    setattr(model, key, value)
            print('best chosen hyperparameterss', best_hyperparameters)
            #print('best alpha', model.alpha)
            #print('best alpha', model.kappa)
            #print('best alpha', model.gamma)
            print('------------------------------------')
            #print('best C_index related to the best partial likelihood:',best_score_C)
            return model, best_hyperparameters
    
    def custom_grid_search_coxnet(self, model, param_grid, X,
                                    events, survival_censoring_time):
            best_score = float('-inf')
            best_estimator = None
            best_hyperparameters = None

            kf = KFold(n_splits= self.num_folds, shuffle=True, random_state=42)

            # Generate all combinations of hyperparameters
            all_hyperparameter_combinations = list(product(*param_grid.values()))


            # Check if GPU is available
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            for hyperparameters in all_hyperparameter_combinations:
                hyperparameter_dict = dict(zip(param_grid.keys(), hyperparameters))
                print(hyperparameter_dict)

                # Update model attributes with hyperparameters
                for key, value in hyperparameter_dict.items():
                    setattr(model, key, value)
                

                fold_number = 0
                scores = []
                scores_C = []
                for train_idx, val_idx in kf.split(X):
                    fold_number += 1
                    X_train, X_val = X[train_idx], X[val_idx]
                    events_train, events_val = events[train_idx], events[val_idx]
                    survival_censoring_time_train, survival_censoring_time_val = survival_censoring_time[train_idx], survival_censoring_time[val_idx]
                    

                    # resetting the initial weights
                    model._reset_parameters()
                    estimated_w = model.fit(X_train=X_train, X_val=X_val,
                                                        survival_censoring_time_train=survival_censoring_time_train, 
                                                        events_train=events_train,
                                                        survival_censoring_time_val=survival_censoring_time_val,
                                                        events_val=events_val, learning_rate=self.lr,
                                                        num_epochs=self.num_epochs, ES_threshold = self.ES_threshold)
                    
                    partial_hazard = model.predict(X_val, estimated_w)

                    score_C = self.eval_metric_performance_cph(survival_censoring_time_val,partial_hazard, events_val)
                    npll = model._negative_partial_log_likelihood(X_val, survival_censoring_time_val, events_val)
                    score = -npll # score is partial log likelihood

                    scores.append(score.detach().cpu().numpy())
                    scores_C.append(score_C)
                    # Release GPU memory after each fold's training and evaluation
                    del X_train, X_val, events_train, events_val, survival_censoring_time_train, survival_censoring_time_val, estimated_w
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    
                
                #print('list of scores in cross validation',scores)
                avg_score = np.mean(scores)
                avg_score_C = np.mean(scores_C)
                #print('average score of k-fold CV:', avg_score)

                if avg_score > best_score:
                    best_score = avg_score
                    best_score_C = avg_score_C
                    # Create a new instance of the model with the best hyperparameters
                    #best_estimator = model.__class__(**hyperparameter_dict)
                    best_hyperparameters = hyperparameter_dict

            for key, value in best_hyperparameters.items():
                    setattr(model, key, value)
            print('best chosen hyperparameterss', best_hyperparameters)
            #print('best alpha', model.alpha)
            #print('best alpha', model.kappa)
            #print('best alpha', model.gamma)
            print('------------------------------------')
            #print('best C_index related to the best partial likelihood:',best_score_C)
            return model, best_hyperparameters

    def custom_grid_search_cph_freq(self, model, param_grid, 
                                    X, 
                                    hla_a_pairs, hla_b_pairs,hla_dr_pairs, 
                            hla_a_pairs_mask, hla_b_pairs_mask, hla_dr_pairs_mask,
                                freq_mask_a, freq_mask_b, freq_mask_dr
                                , events, survival_censoring_time):
        best_score = float('-inf')
        best_estimator = None
        best_hyperparameters = None

        kf = KFold(n_splits= self.num_folds, shuffle=True, random_state=42)

        # Generate all combinations of hyperparameters
        all_hyperparameter_combinations = list(product(*param_grid.values()))

        # Check if TPU is available
        if "TPU_NAME" in os.environ:
            device = torch.device("xla")  # XLA is PyTorch's TPU device

        else:
            # Check if GPU is available
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        for hyperparameters in all_hyperparameter_combinations:
            hyperparameter_dict = dict(zip(param_grid.keys(), hyperparameters))
            print(hyperparameter_dict)

            # Update model attributes with hyperparameters
            for key, value in hyperparameter_dict.items():
                setattr(model, key, value)

            print('d of the model:',model.d)
            
            #print(model.alpha)
            #print(model.gamma)
            fold_number = 0
            scores = []
            for train_idx, val_idx in kf.split(X):
                fold_number += 1
                print('fold ', fold_number)
                X_train, X_val = X[train_idx], X[val_idx]
                hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train = hla_a_pairs[train_idx], hla_b_pairs[train_idx], hla_dr_pairs[train_idx]
                hla_a_pairs_val, hla_b_pairs_val, hla_dr_pairs_val = hla_a_pairs[val_idx], hla_b_pairs[val_idx], hla_dr_pairs[val_idx]
                events_train, events_val = events[train_idx], events[val_idx]
                survival_censoring_time_train, survival_censoring_time_val = survival_censoring_time[train_idx], survival_censoring_time[val_idx]
                

                # resetting the initial weights
                model._reset_parameters()
                estimated_w, estimated_Va, estimated_Vb, estimated_Vdr = model.fit_freq(X_train=X_train, 
                    hla_a_pairs_train=hla_a_pairs_train, hla_b_pairs_train=hla_b_pairs_train, hla_dr_pairs_train=hla_dr_pairs_train,
                    survival_censoring_time_train=survival_censoring_time_train, events_train=events_train,
                    X_val=X_val, hla_a_pairs_val=hla_a_pairs_val, hla_b_pairs_val=hla_b_pairs_val, hla_dr_pairs_val=hla_dr_pairs_val, 
                    survival_censoring_time_val=survival_censoring_time_val, events_val=events_val, 
                    hla_a_pairs_mask=hla_a_pairs_mask, hla_b_pairs_mask=hla_b_pairs_mask, hla_dr_pairs_mask=hla_dr_pairs_mask,
                    freq_mask_a=freq_mask_a, freq_mask_b=freq_mask_b, freq_mask_dr=freq_mask_dr,
                    learning_rate=self.lr, num_epochs=self.num_epochs, ES_threshold = self.ES_threshold)
                



                #score = self.eval_metric_performance_cph(survival_censoring_time_val,partial_hazard, events_val)
                npll = model._negative_partial_log_liklihood_freq(X_val, hla_a_pairs_val, hla_b_pairs_val, hla_dr_pairs_val, survival_censoring_time_val, events_val)
                score = -npll # score is partial log likelihood

                scores.append(score.detach().cpu().numpy())
                # Release GPU memory after each fold's training and evaluation
                del X_train, X_val, hla_a_pairs_train, hla_b_pairs_train, hla_dr_pairs_train, hla_a_pairs_val, hla_b_pairs_val, hla_dr_pairs_val, events_train, events_val, survival_censoring_time_train, survival_censoring_time_val, estimated_w, estimated_Va, estimated_Vb, estimated_Vdr
                gc.collect()
                torch.cuda.empty_cache()
                break
                
            
            #print('list of scores in cross validation',scores)
            avg_score = np.mean(scores)
            print('average score of k-fold CV:', avg_score)

            if avg_score > best_score:
                best_score = avg_score
                # Create a new instance of the model with the best hyperparameters
                #best_estimator = model.__class__(**hyperparameter_dict)
                best_hyperparameters = hyperparameter_dict

        for key, value in best_hyperparameters.items():
                setattr(model, key, value)
        print('best chosen hyperparameterss', best_hyperparameters)
        #print('best alpha', model.alpha)
        #print('best alpha', model.kappa)
        #print('best alpha', model.gamma)
        print('------------------------------------')
        return model, best_hyperparameters





    def custom_grid_search_logreg(self, model, param_grid, X, X_interaction, y, mask_bool):
        best_score = float('-inf')
        best_hyperparameters = None

        # Use StratifiedKFold instead of KFold
        kf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        # Generate all combinations of hyperparameters
        all_hyperparameter_combinations = list(product(*param_grid.values()))

        # Device selection logic remains the same
        if "TPU_NAME" in os.environ:
            device = torch.device("xla")
        else:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        for hyperparameters in all_hyperparameter_combinations:
            hyperparameter_dict = dict(zip(param_grid.keys(), hyperparameters))
            #print(hyperparameter_dict)

            # Update model attributes with hyperparameters
            for key, value in hyperparameter_dict.items():
                setattr(model, key, value)

            scores = []
            # Adapted for StratifiedKFold
            X = X.cpu().numpy()
            y = y.cpu().numpy()
            for train_idx, val_idx in kf.split(X, y):
                X = torch.tensor(X, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)
                X,y = X.to(device), y.to(device)
                X_train, X_val = X[train_idx], X[val_idx]
                
                X_train_interaction, X_val_interaction = X_interaction[train_idx], X_interaction[val_idx]
                
            
                y_train, y_val = y[train_idx], y[val_idx]

                model._reset_parameters()
                estimated_w, estimated_V = model.fit(X_train=X_train, X_train_interaction=X_train_interaction,
                                                    y_train=y_train, X_val=None, X_val_interaction=None,
                                                    y_val=None, learning_rate=self.lr, num_epochs=self.num_epochs,
                                                    ES_threshold=self.ES_threshold, batch_size=self.batch_size)

                y_pred = model.predict_proba(X_val, X_val_interaction, estimated_w, estimated_V, mask_bool)
                score = self.eval_metric_performance_logreg(y_val, y_pred)
                scores.append(score)

                # Memory cleanup
                del X_train, X_val, X_train_interaction, X_val_interaction, y_train, y_val, estimated_w, estimated_V
                gc.collect()
                torch.cuda.empty_cache()

            avg_score = np.mean(scores)
            #print('average score of k-fold CV:', avg_score)

            if avg_score > best_score:
                best_score = avg_score
                best_hyperparameters = hyperparameter_dict

        for key, value in best_hyperparameters.items():
            setattr(model, key, value)

        print('best chosen hyperparameters:', best_hyperparameters)
        print('------------------------------------')
        return model, best_hyperparameters


    def custom_grid_search_logreg_FM(self, model, param_grid, X, X_interaction, y, mask_bool):
        best_score = float('-inf')
        best_hyperparameters = None

        # Use StratifiedKFold instead of KFold
        kf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        # Generate all combinations of hyperparameters
        all_hyperparameter_combinations = list(product(*param_grid.values()))

        # Device selection logic remains the same
        if "TPU_NAME" in os.environ:
            device = torch.device("xla")
        else:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        for hyperparameters in all_hyperparameter_combinations:
            hyperparameter_dict = dict(zip(param_grid.keys(), hyperparameters))
            #print(hyperparameter_dict)

            # Update model attributes with hyperparameters
            for key, value in hyperparameter_dict.items():
                setattr(model, key, value)

            scores = []
            # Adapted for StratifiedKFold
            X = X.cpu().numpy()
            y = y.cpu().numpy()
            for train_idx, val_idx in kf.split(X, y):
                X = torch.tensor(X, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)
                X,y = X.to(device), y.to(device)
                X_train, X_val = X[train_idx], X[val_idx]

                X_train_interaction, X_val_interaction = X_interaction[train_idx], X_interaction[val_idx]

                y_train, y_val = y[train_idx], y[val_idx]

                model._reset_parameters()
                estimated_w, estimated_V = model.fit(X_train=X_train, X_train_interaction=X_train_interaction,
                                                    y_train=y_train, X_val=X_val, X_val_interaction=X_val_interaction,
                                                    y_val=y_val, learning_rate=self.lr, num_epochs=self.num_epochs,
                                                    ES_threshold=self.ES_threshold, batch_size=self.batch_size)

                y_pred = model.predict_proba(X_val, X_val_interaction, estimated_w, estimated_V, mask_bool)
                score = self.eval_metric_performance_logreg(y_val, y_pred)
                scores.append(score)

                # Memory cleanup
                del X_train, X_val, X_train_interaction, X_val_interaction, y_train, y_val, estimated_w, estimated_V
                gc.collect()
                torch.cuda.empty_cache()

            avg_score = np.mean(scores)
            #print('average score of k-fold CV:', avg_score)

            if avg_score > best_score:
                best_score = avg_score
                best_hyperparameters = hyperparameter_dict

        for key, value in best_hyperparameters.items():
            setattr(model, key, value)

        print('best chosen hyperparameters:', best_hyperparameters)
        print('------------------------------------')
        return model, best_hyperparameters





    def eval_metric_performance_mse(self, y_true, y_pred):
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        return mean_squared_error(y_true, y_pred)
        
    def eval_metric_performance_cph(self, survival_censoring_time_val, partial_hazard, events_val):
        partial_hazard=partial_hazard.detach().cpu().numpy()
        survival_censoring_time_val = survival_censoring_time_val.detach().cpu().numpy()
        events_val= events_val.detach().cpu().numpy()
        c_index = concordance_index(survival_censoring_time_val, -partial_hazard, events_val)
        return c_index
    
    def eval_metric_performance_logreg(self, y_true, y_pred):
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()

        auc = roc_auc_score(y_true_np, y_pred_np)
        return auc

