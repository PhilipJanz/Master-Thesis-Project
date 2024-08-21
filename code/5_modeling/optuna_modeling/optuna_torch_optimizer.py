from copy import deepcopy

import numpy as np
import optuna

import lightning as pl
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data_assembly import group_years


class Tensor(Dataset):
    def __init__(self, X, y):
        # Example conversion of X_train and y_train
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.to_numpy()

        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.to_numpy().ravel()

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LitModel(pl.LightningModule):
    def __init__(self, input_shape, trial=None, params=None):
        super(LitModel, self).__init__()

        # Check if either trial or params is provided
        assert (not trial) ^ (not params), "Choose either params OR give an optuna trial (it's xor)"

        layers = []
        in_features = input_shape  # Initialize in_features with input_shape

        # Hidden layers
        if trial:
            n_layers = trial.suggest_int('n_layers', 1, 1)
            for i in range(n_layers):
                units = trial.suggest_int(f'units_layer{i + 1}', 1, 32)
                layers.append(nn.Linear(in_features, units))
                layers.append(nn.Sigmoid())
                dropout_rate = trial.suggest_float(f'dropout_layer{i + 1}', 0.0, 0.5)
                layers.append(nn.Dropout(dropout_rate))
                in_features = units  # Update in_features for the next layer
        else:
            for i in range(params["n_layers"]):
                units = params[f'units_layer{i + 1}']
                layers.append(nn.Linear(in_features, units))
                layers.append(nn.Sigmoid())
                dropout_rate = params[f'dropout_layer{i + 1}']
                layers.append(nn.Dropout(dropout_rate))
                in_features = units  # Update in_features for the next layer

        # Output layer
        layers.append(nn.Linear(in_features, 1))

        self.model = nn.Sequential(*layers)

        # Store learning rate
        if trial:
            self.learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
        else:
            self.learning_rate = params["learning_rate"]

    def predict(self, X):
        # test on validation data
        X_val_tensor = torch.tensor(X, dtype=torch.float32)
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            y_pred = self(X_val_tensor).T[0].detach().numpy()  # Predict on the entire validation set
        self.train()
        return y_pred

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class OptunaOptimizer:
    """

    :param X: (n x p)-array
    :param y: (n)-array
    :param years: (n)-array: important since we so 'leave-year-out-CV'
    :param predictor_names: (p)-array: column names of X. Is used to make feature selection and feature shrinking
    :param best_params: usually None because that's the goal of optimization. But it can be used for model transfer
    :param feature_set_selection: (bool)
    :param feature_len_shrinking: (bool)
    :param model_types: list of  model names to be considered. Subset of ['svr', 'rf', 'lasso', 'nn', 'xgb']
    :param sampler: optuna sampler
    :param num_folds: n-fold for CV. num_folds > n_unique_years results into leave-one-year-out-CV
    :param repetition_per_fold: make more reliable parameter choice by training models multiple times
    :param seed: for splitting years into folds for CV
    """
    def __init__(self,
                 X,
                 y,
                 years,
                 predictor_names,
                 best_params=None,
                 sampler=optuna.samplers.TPESampler,
                 num_folds=5,
                 repetition_per_fold=1,
                 seed=42):
        # init
        self.X = X
        self.y = y
        self.years = years
        self.predictor_names = predictor_names
        self.best_params = best_params
        self.num_folds = num_folds
        self.seed = seed
        self.repetition_per_fold = repetition_per_fold

        # Create a new Optuna study
        self.study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(self, trial):
        epochs = trial.suggest_int('epochs', 200, 1000)
        batch_size = trial.suggest_int('batch_size', 50, len(self.X))

        model = LitModel(input_shape=self.X.shape[1], trial=trial)

        # shrink number of folds if wanted else make loyoCV
        if self.num_folds < len(np.unique(self.years)):
            folds = group_years(years=self.years, n=self.num_folds, seed=self.seed)
        else:
            folds = self.years.copy()

        # CV
        # Save the error of each fold in a list
        mse_ls = []
        for _ in range(self.repetition_per_fold):
            for fold_out in np.unique(folds):
                fold_out_bool = (folds == fold_out)

                X_train, y_train = self.X[~fold_out_bool], self.y[~fold_out_bool]
                X_val, y_val = self.X[fold_out_bool], self.y[fold_out_bool]

                # copy model to not train it for other hold-out-years
                model_copy = deepcopy(model)

                # Convert your numpy arrays to a Dataset
                train_dataset = Tensor(X_train, y_train)

                # Create DataLoader with the suggested batch size
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                # Set up the trainer with the number of epochs
                trainer = pl.Trainer(max_epochs=epochs, accelerator='cpu',
                                     logger=False,  # Disables the logger
                                     enable_progress_bar=False,  # Disables the progress bar
                                     enable_model_summary=False)

                # Train the model
                trainer.fit(model_copy, train_loader)
                y_pred = model_copy.predict(X=X_val)
                mse = np.mean((y_pred - y_val) ** 2)
                mse_ls.append(mse)

        return np.mean(mse_ls)

    def optimize(self, timeout=600, n_trials=100,
                 n_jobs=-1, show_progress_bar=True, print_result=True):
        self.study.optimize(self.objective, n_trials=n_trials, timeout=timeout,
                            n_jobs=n_jobs, show_progress_bar=show_progress_bar,
                            gc_after_trial=True)

        self.best_mse, self.best_params = self.study.best_trial.value, self.study.best_trial.params

        if print_result:
            # Print the best performance
            print(f"  Best MSE: {self.best_mse}")

            # Print the best hyperparameters
            print(f"  Best Params: ")
            for key, value in self.best_params.items():
                print("    {}: {}".format(key, value))

        # return (best models mse, best models params
        return self.best_mse, self.best_params

    def train_best_model(self):
        """
        This method applies the best parameters (found by optimization) to any dataset of X and y
        trains the optimal model and exports the model as well as the transformed X
        :return: trained_model
        """
        assert self.best_params, ".apply() applies optimal parameters that are found by .optimize() or by setting them: self.best_params = {...}"

        model = LitModel(input_shape=self.X.shape[1], params=self.best_params)

        # Convert your numpy arrays to a Dataset
        train_dataset = Tensor(self.X, self.y)

        # Create DataLoader with the suggested batch size
        train_loader = DataLoader(train_dataset, batch_size=self.best_params["batch_size"], shuffle=True)

        # Set up the trainer with the number of epochs
        trainer = pl.Trainer(max_epochs=self.best_params["epochs"],
                             logger=False,  # Disables the logger
                             enable_progress_bar=False,  # Disables the progress bar
                             enable_model_summary=False)  # Disables the model summary)

        # Train the model
        trainer.fit(model, train_loader)

        return model
