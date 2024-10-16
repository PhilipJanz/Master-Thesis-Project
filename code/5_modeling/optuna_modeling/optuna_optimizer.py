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
                'eta': trial.suggest_float('learning_rate', 1e-3, 1, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 1e-7, 1, log=True),
                'alpha': trial.suggest_float('alpha', 1e-7, 100.0, log=True)
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
    def train_best_model(self, X, y):
        """
        This method applies the best parameters (found by optimization) to any dataset of X and y
        trains the optimal model and exports the model as well as the transformed X
        :param X: 2D np.array with predictors
        :param y: np.array of yields
        :param years: np.array of harvest years
        :return: trained_model
        """
        assert self.best_params, ".apply() applies optimal parameters that are found by .optimize() or by setting them: self.best_params = {...}"

        model, _ = init_model(model_name=self.model_type, params=self.best_params)

        # Fit the model
        model.fit(X, y)

        return model
