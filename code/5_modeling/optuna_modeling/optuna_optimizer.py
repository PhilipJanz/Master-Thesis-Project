from optuna import TrialPruned

from config import SEED
from copy import deepcopy

import optuna
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.utils._testing import ignore_warnings
from xgboost import XGBRegressor

from data_assembly import group_years

import numpy as np
from scipy.spatial.distance import pdist, squareform


class GaussianProcess:
    def __init__(self, sigma=1, l_s=0.5, l_t=1.5, noise=0.1, const=0.01):
        # Hyperparameters of the Gaussian process
        self.sigma = sigma
        self.l_s = l_s  # spatial length scale
        self.l_t = l_t  # temporal length scale
        self.noise = noise
        self.const = const
        self.X_mean = None
        self.X_scale = None

    def fit(self, X, y, years):
        """
        Fits the Gaussian Process model using the provided training data.

        Parameters:
        X (array-like): Feature matrix of shape (n_samples, n_features)
        y (array-like): Target vector of shape (n_samples,)
        years (array-like): Years associated with the data points (n_samples,)
        """
        # Normalize years (temporal data)
        years = years.values[:, np.newaxis]
        self.year_mean = np.mean(years, axis=0, keepdims=True)
        self.year_scale = np.amax(years, axis=0) - np.amin(years, axis=0)
        self.years_train = (years - self.year_mean) / self.year_scale

        # Feature normalization (z-score normalization)
        self.X_mean = np.mean(X, axis=0)
        self.X_scale = np.std(X, axis=0)
        X_normalized = (X - self.X_mean) / self.X_scale

        bias = np.ones([X_normalized.shape[0], 1])
        self.X_train = np.concatenate((X_normalized, bias), axis=1)

        self.y_train = y.values[:, np.newaxis]

        # Calculate the spatial and temporal kernel matrices
        pairwise_dists_loc = squareform(pdist(self.X_train, 'euclidean')) ** 2 / self.l_s ** 2
        pairwise_dists_year = squareform(pdist(self.years_train, 'euclidean')) ** 2 / self.l_t ** 2

        # Compute the kernel matrix with noise
        n_train = X_normalized.shape[0]
        noise_matrix = self.noise * np.identity(n_train)
        self.K_train = self.sigma * (np.exp(-pairwise_dists_loc) * np.exp(-pairwise_dists_year)) + noise_matrix

        # Precompute the inverse of the kernel matrix
        self.K_inv = np.linalg.inv(self.K_train)

    def predict(self, X, years):
        """
        Predicts the target values for new inputs using the Gaussian Process model.

        Parameters:
        X (array-like): Feature matrix of shape (n_samples, n_features)
        years (array-like): Years associated with the data points (n_samples,)

        Returns:
        y_pred (array-like): Predicted target values of shape (n_samples,)
        """
        # Normalize the years for the test set
        years = years.values[:, np.newaxis]
        years_test = (years - self.year_mean) / self.year_scale

        # Feature normalization (using the mean and scale from training set)
        X_normalized = (X - self.X_mean) / self.X_scale
        bias = np.ones([X_normalized.shape[0], 1])
        X_test = np.concatenate((X_normalized, bias), axis=1)

        # Compute the spatial and temporal kernel matrices for the test set
        pairwise_dists_loc_test = squareform(
            pdist(np.concatenate((self.X_train, X_test), axis=0), 'euclidean')) ** 2 / self.l_s ** 2
        pairwise_dists_year_test = squareform(
            pdist(np.concatenate((self.years_train, years_test), axis=0), 'euclidean')) ** 2 / self.l_t ** 2

        n_train = self.X_train.shape[0]
        n_test = X_test.shape[0]

        # Kernel matrix for the test points and between train and test points
        K_test_train = self.sigma * (np.exp(-pairwise_dists_loc_test[n_train:, :n_train]) * np.exp(
            -pairwise_dists_year_test[n_train:, :n_train]))

        # Compute predictions using the GP formula
        y_pred = K_test_train.dot(self.K_inv).dot(self.y_train)

        return y_pred.flatten()


def init_model(model_name, trial=None, params=None):
    # check if inputs are reasonable: xor on trial and params
    assert (not trial) ^ (not params), "Choose either params OR give an optuna trial (it's xor)"

    # Define model parameters
    if model_name == 'svr':
        if trial:
            model_params = {
                'C': trial.suggest_float('C', 1e-7, 100.0, log=True),
                'epsilon': trial.suggest_float('epsilon', 1e-5, 10.0, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            }
        else:
            model_params = {key: params[key] for key in ["C", "epsilon", "gamma", "kernel"]}
        return SVR(**model_params), model_params
    elif model_name == 'rf':
        if trial:
            model_params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 500),
                'max_depth': trial.suggest_int('max_depth', 1, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20)
            }
        else:
            model_params = {key: params[key] for key in
                            ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]}
        return RandomForestRegressor(**model_params), model_params
    elif model_name == 'gp':
        if trial:
            model_params = {
                'sigma': trial.suggest_float('sigma', 1e-2, 10.0, log=True),
                'l_s': trial.suggest_float('l_s', 1e-2, 100.0, log=True),
                'l_t': trial.suggest_float('l_t', 1e-2, 10.0, log=True),
                'noise': trial.suggest_float('noise', 1e-6, 10.0, log=True),
                'const': trial.suggest_float('const', 1e-4, 10.0, log=True)
            }
        else:
            model_params = {key: params[key] for key in
                            ["sigma", "l_s", "l_t", "noise", "const"]}
        return GaussianProcess(**model_params), model_params
    elif model_name == 'lasso':
        if trial:
            model_params = {
                'alpha': trial.suggest_float('alpha', 1e-9, 100.0, log=True)
            }
        else:
            model_params = {"alpha": params["alpha"]}
        return Lasso(**model_params), model_params
    elif model_name == 'xgb':
        if trial:
            model_params = {
                'max_depth': trial.suggest_int('max_depth', 1, 20),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 1e-9, 1, log=True),
                'alpha': trial.suggest_float('alpha', 1e-9, 100.0, log=True)
            }
        else:
            model_params = {key: params[key] for key in
                            ["max_depth", "learning_rate", "subsample", 'colsample_bytree', 'gamma', "alpha"]}
        model_params["lambda"] = 0
        model_params["n_estimators"] = 300
        return XGBRegressor(**model_params, random_state=SEED), model_params
    else:
        raise AssertionError(f"Model name '{model_name}' is not in: ['svr', 'rf', 'lasso', 'nn', 'lstm']")


class OptunaOptimizer:
    """

    :param X: (n x p)-array
    :param y: (n)-array
    :param years: (n)-array: important since we so 'leave-year-out-CV'
    :param best_params: usually None because that's the goal of optimization. But it can be used for model transfer
    :param model_type:model name. from ['svr', 'rf', 'lasso', 'xgb']
    :param sampler: optuna sampler
    :param num_folds: n-fold for CV. num_folds > n_unique_years results into leave-one-year-out-CV
    :param repetition_per_fold: make more reliable parameter choice by training models multiple times
    """
    def __init__(self,
                 study_name,
                 X,
                 y,
                 years,
                 best_params=None,
                 max_feature_len=10,
                 model_type=None,
                 sampler=optuna.samplers.TPESampler,
                 pruner=None,
                 num_folds=5,
                 repetition_per_fold=1):
        # init
        self.X = X
        self.y = y
        self.years = years
        self.best_params = best_params
        self.model_type = model_type
        self.num_folds = num_folds
        self.repetition_per_fold = repetition_per_fold
        self.max_feature_len = max_feature_len

        # Create a new Optuna study
        self.study = optuna.create_study(study_name=study_name, direction="minimize", sampler=sampler, pruner=pruner)

    @ignore_warnings(category=ConvergenceWarning)
    def objective(self, trial):
        # It might happen that each and every feature is filtered out.
        # In that case we are left with the naive average predictor:
        if len(self.X) == 0:
            # mse of average predictor is the variance
            return np.mean(self.y ** 2)

        # Generate params for model
        model, params = init_model(model_name=self.model_type, trial=trial)

        # shrink number of folds if wanted else make loyoCV
        if self.num_folds < len(np.unique(self.years)):
            folds = group_years(years=self.years, n=self.num_folds)
        else:
            folds = self.years.copy()

        # CV
        # Save the error of each fold in a list
        mse_ls = []
        for _ in range(self.repetition_per_fold):
            for j, fold_out in enumerate(np.unique(folds)):
                fold_out_bool = (folds == fold_out)

                if self.model_type == "gp":
                    X_train, y_train, years_train = self.X[~fold_out_bool], self.y[~fold_out_bool], self.years[~fold_out_bool]
                    X_val, y_val, years_val = self.X[fold_out_bool], self.y[fold_out_bool], self.years[fold_out_bool]

                    # Fit the model and differentiate between the model classes
                    model_copy = deepcopy(model)
                    # Fit the model
                    model_copy.fit(X_train, y_train, years_train)

                    # Predict the target values
                    preds = model_copy.predict(X_val, years_val)
                else:
                    X_train, y_train = self.X[~fold_out_bool], self.y[~fold_out_bool]
                    X_val, y_val = self.X[fold_out_bool], self.y[fold_out_bool]

                    # Fit the model and differentiate between the model classes
                    model_copy = deepcopy(model)
                    # Fit the model
                    model_copy.fit(X_train, y_train)

                    # Predict the target values
                    preds = model_copy.predict(X_val)

                # Calculate the RMSE and append it to the list
                fold_mse = mean_squared_error(y_val, preds)
                mse_ls.append(fold_mse)

                trial.report(fold_mse, step=j)

                if trial.should_prune():
                    raise TrialPruned()

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

    @ignore_warnings(category=ConvergenceWarning)
    def train_best_model(self):
        """
        This method applies the best parameters (found by optimization) to any dataset of X and y
        trains the optimal model and exports the model as well as the transformed X
        :return: trained_model
        """
        assert self.best_params, ".apply() applies optimal parameters that are found by .optimize() or by setting them: self.best_params = {...}"

        model, _ = init_model(model_name=self.model_type, params=self.best_params)

        # Fit the model
        if self.model_type == "gp":
            model.fit(self.X, self.y, self.years)
        else:
            model.fit(self.X, self.y)

        return model
