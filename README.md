## Real Data Experiments – Logistic Regression

This experiment evaluates different logistic regression models (including our proposed LIT-LVM variants) on real-world datasets using interaction terms.

### Script: `logreg_experiment.py`

###  Requirements

Make sure the following dependencies are installed:

```bash
pip install torch scikit-learn pandas matplotlib
```

Ensure the following files are available in the repository:

- `utils.py` (should define `GridSearch`)
- `dataset.py` (should define `data_loader`)
- Model file: `logreg_LIT_LVM_V1.py`

###  How to Run

```bash
python logreg_experiment.py --dataset <dataset_name> --model <model_name> --d <latent_dim> --lr <learning_rate> --note "<your notes>"
```

- `--dataset`: Name of the dataset file (e.g., `tecator.arff`)
- `--model`: One of the following:
  - `litlvmV1`
  - `elasticnet`: Elastic Net baseline
- `--d`: Latent dimension (default: 2)
- `--lr`: Learning rate (default: 0.05)
- `--note`: Optional note to describe the experiment setup

###  Example

```bash
python logreg_experiment.py --dataset tecator.arff --model litlvmV2 --d 2 --lr 0.01 --note "Running LIT-LVM v2 with d=2"
```

###  Modifiable Parameters

You can tune the following parameters directly **within the script**:

| Parameter            | Variable Name     | Location in Script         |
|----------------------|-------------------|-----------------------------|
| Latent dimension     | `d`               | Set via CLI and used throughout |
| Learning rate        | `learning_rate`   | Set via `--lr` argument     |
| Number of epochs     | `num_epochs`      | Defined near top of script  |
| Number of folds      | `folds`           | Defined near top of script  |
| Number of experiments| `num_experiments` | Defined near top of script  |
| Early stopping tol   | `es_threshold`    | Defined near top of script  |
| Train/val/test sizes | `train_size`, `val_size`, `test_size` | Defined near end of script |
| Hyperparam grid      | `param_grid_1`, `param_grid_2` | Defined inside the script |

- Use `param_grid_1` for LIT-LVM models (model_1, model_2).
- Use `param_grid_2` for Elastic Net variants (model_3).

These control the search over `alpha`, `kappa`, `gamma`, etc.

###  Output

Results are automatically saved in a text file under the specified output folder, with a name like:

```
<dataset>_model:<model_name>_lr:<learning_rate>_d:<d>_tol:<tol>_folds:<folds>.txt
```

The output includes the average AUC scores for each repetition and the best hyperparameters selected during tuning.


##  Real Data Experiments – Linear Regression

These experiments evaluate different linear regression models on real datasets with interaction terms.

The setup is **similar to the logistic regression experiments**, but the script and models are tailored for regression tasks.

###  Script: `linreg_experiment.py`

###  Requirements

Ensure the same dependencies and structure as the logistic regression experiment are available, including:

- `utils.py` (defines `GridSearch`)
- `dataset.py` (defines `data_loader`)
- Model files: `linreg_LIT_LVM_V1.py`

###  How to Run

```bash
python linreg_experiment.py --dataset <dataset_name> --model <model_name>
```

- `--dataset`: Name of the dataset file (e.g., `tecator.arff`)
- `--model`: Choose one of the following:
  - `litlvmV1`
  - `elasticnet`: Elastic Net baseline

###  Example

```bash
python linreg_experiment.py --dataset tecator.arff --model litlvmV1
```

###  Modifiable Parameters

As with the logistic regression script, the following parameters can be modified **within `linreg_experiment.py`**:

| Parameter            | Variable Name     | Description                       |
|----------------------|-------------------|-----------------------------------|
| Latent dimension     | `d`               | Dimension of latent embedding     |
| Learning rate        | `learning_rate`   | Learning rate                     |
| Number of epochs     | `num_epochs`      | Training epochs                   |
| Number of folds      | `folds`           | For cross-validation              |
| Number of experiments| `num_experiments` | Number of repeated runs           |
| Early stopping tol   | `es_threshold`    | Patience for early stopping       |
| Train/val/test sizes | `train_size`, `val_size`, `test_size` | Data splitting ratios       |
| Hyperparam grid      | `param_grid_1`, `param_grid_2` | For tuning LIT-LVM or Elastic Net |

- Use `param_grid_1` for LIT-LVM models.
- Use `param_grid_2` for Elastic Net models.

###  Output

Results will be saved under:

```
experiments/results/Linear_Regression/<DATE>/<dataset_name>_model:<model_name>_lr:<learning_rate>_threshold:<es_threshold>_tol:<tol>_folds:<folds>.txt
```

Each file includes R² scores and RMSE across runs, along with best hyperparameters per trial.

## Real Data Experiments – SparseFM Baselines

To evaluate SparseFM baselines on real data, we use the **same scripts** as for the LIT-LVM models:

- `logreg_experiment.py` for classification tasks
- `linreg_experiment.py` for regression tasks

The overall workflow—data loading, model training, evaluation—remains unchanged. The only difference is in the hyperparameter configuration.

###  Parameter Configuration

Hyperparameter tuning is controlled via `param_grid_1` in both scripts. For SparseFM, we **fix all other hyperparameters** and only **search over values of `gamma`**, which corresponds to the **regularization strength on the latent structure** in our paper.

To enforce SparseFM, we use **very high values** of this regularization strength:

```python
'gamma': [1e5, 1e6, 1e7, 1e8]
```

###  Notes

- `gamma` in the code corresponds to the latent structure regularization strength discussed in the paper.
- No modifications to the script are needed—only the `param_grid_1` setting must be adjusted.

## Simulated Data Experiments – Logistic Regression

We perform simulation experiments to evaluate the performance of **LIT-LVM**, **ElasticNet**, and **Factorization Machines (FM)** under controlled conditions. Each method has its own experiment file:

- `sim_logreg_experiment_litlvm.py` — for LIT-LVM
- `sim_logreg_experiment_elasticnet.py` — for ElasticNet
- `sim_logreg_experiment_FMs.py` — for Factorization Machines

All experiments use the same simulation class (`LogisticRegression_Simulator`) and share a similar procedure to the real data experiments. The key difference is that **data is synthetically generated** based on a known low-rank structure with configurable noise and sparsity.

### How to Run

Each script supports CLI arguments to control the simulation setup:

```bash
python sim_logreg_experiment_<model>.py --p <feature_dim> --d <latent_dim> --V_noise <noise_level> --sparsity <0_or_1> --noise <0_or_1> --interaction <0_or_1> --sigma <sparsity_strength>
```

Example for LIT-LVM:

```bash
python sim_logreg_experiment_litlvm.py --p 60 --d 2 --V_noise 0.01 --sparsity 1 --noise 1 --interaction 1 --sigma 0.0001 
```

Example for FM:

```bash
python sim_logreg_experiment_FMs.py --p 60 --d 5 --V_noise 0.01 --sparsity 0 --noise 1 --interaction 1 --sigma 0.0001 --gamma 1e6
```

Example for ElasticNet:

```bash
python sim_logreg_experiment_elasticnet.py --p 60 --d 2 --V_noise 0.01 --sparsity 1 --noise 1 --interaction 1 --sigma 0.0001
```

###  Controllable Parameters (Inside Each Script)

The following values can be **manually changed from within each script** (not from command-line arguments):

| Parameter             | Variable Name     | Description                                                                 |
|-----------------------|-------------------|-----------------------------------------------------------------------------|
| Training samples      | `num_samples`     | Number of samples per train/val/test set                                   |
| Learning rate         | `learning_rate`   | Step size for gradient descent                                             |
| Number of epochs      | `num_epochs`      | Training duration                                                           |
| Early stopping        | `es_threshold`    | Number of epochs without improvement to stop                               |
| Number of splits      | `data_split` loop | Number of independent train/val/test runs (set to 5 by default)            |
| Model depth (FM, LIT-LVM) | `d`           | Latent dimension, also varies across models internally (2, 5, 10) for FM/LIT-LVM |
| Param grid            | `param_grid`      | Hyperparameter grid for model selection, including `alpha`, `kappa`, `gamma` |
| Regularization strength (`gamma`) | `gamma` (LIT-LVM) | Controlled via CLI and used to penalize latent structure                  |

- In **LIT-LVM**, we search over values of `gamma`, corresponding to **latent structure regularization strength**.
- In **ElasticNet**, `gamma` is fixed at 0 (no latent structure penalty).
- In **FM**, the dimension `d` is varied manually`.

###  Output

Each script logs results (AUC, MSE of coefficients) to a file like:

```
experiments/results/simulations/logistic_regression/<DATE>/<MODEL>_p:<p>_d:<d>_lr:<lr>_Noise:<...>.txt
```

The logs include:
- AUC on held-out test set
- MSE between true and estimated coefficients (`w`) and interaction matrices (`V`)
- Standard errors across runs

```python
# Example output summary in file
result_2: [0.91, 0.89, 0.92, ...], mean: 0.905, SE: 0.012
result_2_w_MSE: [...], result_2_V_MSE: [...]
```

###  Summary

- Use different scripts for each baseline.
- Set model and experiment type via CLI.
- Modify training/control settings inside the script.
- Results are automatically logged with reproducible configurations.

##  Simulated Data Experiments – Linear Regression

The linear regression simulations follow **exactly the same structure** as the logistic regression experiments. The only difference is that they use regression-specific simulators and evaluation metrics.

The corresponding scripts are:

- `sim_linreg_experiment_litlvm.py`
- `sim_linreg_experiment_elasticnet.py`
- `sim_linreg_experiment_FMs.py`




