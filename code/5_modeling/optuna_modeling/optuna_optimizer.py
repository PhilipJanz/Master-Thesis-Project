from config import RESULTS_DATA_DIR, SEED
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

import tensorflow as tf

#tf.keras.config.disable_interactive_logging()
tf.random.set_seed(SEED)
from tensorflow.keras import Sequential
from tensorflow.keras.models import clone_model, Model
from tensorflow.keras.layers import Input, Dropout, LSTM, Dense, concatenate

from data_assembly import rescale_array, group_years
from optuna_modeling.feature_sets_for_optuna import feature_sets


def create_nn(trial=None, params=None):
    """
    This function creates a neural network with variable number of hidden layers and drop-out regularisation
    The parameter can be given by 'params' or will be chosen by optuna using 'trial'. Only one is possible
    :param trial: trial from optuna for hyperparameter optimization
    :param params: specific hyperparameters
    :return: Keras-nn ready to be trained. Make it run the extra mile!
    """
    # check if inputs are reasonable: xor on trial and params
    assert (not trial) ^ (not params), "Choose either params OR give an optuna trial (it's xor)"

    model = Sequential()

    # Hidden layers
    if trial:
        n_layers = trial.suggest_int('n_layers', 1, 1)
        for i in range(n_layers):
            model.add(Dense(trial.suggest_int(f'units_layer{i + 1}', 1, 32), activation='sigmoid'))
            model.add(Dropout(trial.suggest_float(f'dropout_layer{i + 1}', 0.0, 0.5)))
    else:
        for i in range(params["n_layers"]):
            model.add(Dense(params[f'units_layer{i + 1}'], activation='sigmoid'))
            model.add(Dropout(params[f'dropout_layer{i + 1}']))

    # Output layer
    model.add(Dense(1, activation='linear'))

    # Compile model
    model.compile(optimizer="Adam", loss='mean_squared_error')

    return model


def create_lstm(t, f, s, trial=None, params=None):
    """
    This function creates a LSTM neural network with variable number of hidden layers and drop-out regularisation
    The parameter can be given by 'params' or will be chosen by optuna using 'trial'. Only one is possible
    :param t: length of timeseries
    :param f: number of temporal features
    :param s: number of static (non-temporal) features
    :param trial: trial from optuna for hyperparameter optimization
    :param params: specific hyperparameters
    :return: Keras-nn ready to be trained. Make it run the extra mile!
    """
    # check if inputs are reasonable: xor on trial and params
    assert (not trial) ^ (not params), "Choose either params OR give an optuna trial (it's xor)"

    # Define input shapes
    time_sensitive_input = Input(shape=(t, f), name='time_sensitive_input')
    static_input = Input(shape=(s,), name='static_input')

    # Hidden layers with dropout
    if trial:
        # dropout after input
        input_dropout = trial.suggest_float(f'input_dropout', 0.2, 0.5)
        time_sensitive_input = Dropout(rate=input_dropout)(time_sensitive_input)
        static_input = Dropout(rate=input_dropout)(static_input)

        # LSTM layer for time-sensitive data
        lstm_out = LSTM(units=trial.suggest_int(f'lstm_units', 1, 5),
                        dropout=trial.suggest_float(f'lstm_dropout', 0.2, 0.5),
                        recurrent_dropout=trial.suggest_float(f'lstm_recurrent_dropout', 0.2, 0.5),
                        return_sequences=False)(time_sensitive_input)
        # Concatenate with static data
        merged = concatenate([static_input, lstm_out])
        # init last hidden layer
        dense_out = Dense(units=trial.suggest_int(f'hidden_units', 1, 32), activation='sigmoid')(merged)
    else:
        # dropout after input
        input_dropout = params[f'input_dropout']
        time_sensitive_input = Dropout(rate=input_dropout)(time_sensitive_input)
        static_input = Dropout(rate=input_dropout)(static_input)

        # LSTM layer for time-sensitive data
        lstm_out = LSTM(units=params[f'lstm_units'],
                        dropout=params[f'lstm_dropout'],
                        recurrent_dropout=params[f'lstm_recurrent_dropout'],
                        return_sequences=False)(time_sensitive_input)
        # Concatenate with static data
        merged = concatenate([static_input, lstm_out])
        # init last hidden layer
        dense_out = Dense(units=params[f'hidden_units'], activation='sigmoid')(merged)

    # Output layer (regression example)
    output = Dense(units=1, activation='linear')(dense_out)

    # Create and compile the model
    model = Model(inputs=[static_input, time_sensitive_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')

    return model


def select_features(X, predictor_names, trial=None, params=None):
    # check if inputs are reasonable: xor on trial and params
    assert (not trial) ^ (not params), "Choose either params OR give an optuna trial (it's xor)"

    # create list of bools representing predictors from X to keep
    predictor_selection_bool = np.repeat(True, X.shape[1])

    # filter feature-set list to the feature-sets that are actually in the data (e.g. soil is missing for region model)
    relevant_feature_sets = feature_sets.copy()
    if "clay" not in predictor_names:
        del relevant_feature_sets["soil"]

    # Select True / False for each feature or get it from params
    if trial:
        feature_set_selection_bool = [trial.suggest_categorical(name, [True, False]) for name in relevant_feature_sets]
    else:
        feature_set_selection_bool = [params[name] for name in relevant_feature_sets]

    # list names of features not selected to filter them out
    left_out_feature_sets = [name for name, selected in zip(relevant_feature_sets, feature_set_selection_bool) if
                             not selected]

    # set predictors False when they were not selected
    for left_out_feature_set in left_out_feature_sets:
        left_out_features = relevant_feature_sets[left_out_feature_set]
        predictor_selection_bool = predictor_selection_bool * [np.all([x not in predictor for x in left_out_features])
                                                               for predictor in predictor_names]

    # make new X with selected feature sets
    X_sel = X[:, predictor_selection_bool]
    sel_predictor_names = predictor_names[predictor_selection_bool]

    return X_sel, sel_predictor_names


def init_model(X, model_name, trial=None, params=None):
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
    elif model_name == 'nn':
        if trial:
            params = {
                "epochs": trial.suggest_int('epochs', 200, 1000),
                "batch_size": trial.suggest_int('batch_size', 50, len(X))
            }
            return create_nn(trial=trial), params
        else:
            return create_nn(params=params), params
    elif model_name == 'lstm':
        X_static, X_time = X
        _, s = X_static.shape
        _, t, f = X_time.shape
        if trial:
            params = {
                "epochs": trial.suggest_int('epochs', 200, 1000),
                "batch_size": trial.suggest_int('batch_size', 50, len(X_static))
            }
            return create_lstm(t=t, f=f, s=s, trial=trial), params
        else:
            return create_lstm(t=t, f=f, s=s, params=params), params
    elif model_name == 'xgb':
        if trial:
            model_params = {
                'max_depth': trial.suggest_int('max_depth', 1, 20),
                'eta': trial.suggest_float('learning_rate', 1e-5, 1, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
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
    :param predictor_names: (p)-array: column names of X. Is used to make feature selection and feature shrinking
    :param best_params: usually None because that's the goal of optimization. But it can be used for model transfer
    :param feature_set_selection: (bool)
    :param feature_len_shrinking: (bool)
    :param model_types: list of  model names to be considered. Subset of ['svr', 'rf', 'lasso', 'nn', 'xgb', 'lstm']
    :param sampler: optuna sampler
    :param num_folds: n-fold for CV. num_folds > n_unique_years results into leave-one-year-out-CV
    :param repetition_per_fold: make more reliable parameter choice by training models multiple times
    """
    def __init__(self,
                 X,
                 y,
                 years,
                 predictor_names=None,
                 best_params=None,
                 feature_set_selection=True,
                 feature_len_shrinking=True,
                 max_feature_len=10,
                 model_types=['svr', 'rf', 'lasso', 'nn', "lstm", 'xgb'],
                 sampler=optuna.samplers.TPESampler,
                 num_folds=5,
                 repetition_per_fold=1):
        # init
        self.X = X
        self.y = y
        self.years = years
        self.predictor_names = predictor_names
        self.best_params = best_params
        self.feature_set_selection = feature_set_selection
        self.model_types = model_types
        self.num_folds = num_folds
        self.repetition_per_fold = repetition_per_fold
        self.feature_set_selection = feature_set_selection
        self.feature_len_shrinking = feature_len_shrinking
        self.max_feature_len = max_feature_len

        # Create a new Optuna study
        self.study = optuna.create_study(direction="minimize", sampler=sampler)

    @ignore_warnings(category=ConvergenceWarning)
    def objective(self, trial):

        if self.feature_set_selection | self.feature_len_shrinking:
            # prepare X: feature selection & feature shrinking
            X_trans, _ = self.transform_X(X=self.X, predictor_names=self.predictor_names, trial=trial)
        else:
            X_trans = self.X

        # It might happen that each and every feature is filtered out.
        # In that case we are left with the naive average predictor:
        if len(X_trans) == 0:
            # mse of average predictor is the variance
            return np.mean(self.y ** 2)

        # Select model type
        if type(self.model_types) == str:
            model_name = self.model_types
        elif type(self.model_types) == list:
            model_name = trial.suggest_categorical('model_type', self.model_types)
        else:
            raise AssertionError(f"Input for model_types is not str or list: {self.model_types}")

        # Generate params for model
        model, params = init_model(X=X_trans, model_name=model_name, trial=trial)

        # shrink number of folds if wanted else make loyoCV
        if self.num_folds < len(np.unique(self.years)):
            folds = group_years(years=self.years, n=self.num_folds)
        else:
            folds = self.years.copy()

        # CV
        # Save the error of each fold in a list
        mse_ls = []
        for _ in range(self.repetition_per_fold):
            for fold_out in np.unique(folds):
                fold_out_bool = (folds == fold_out)

                if model_name == "lstm":
                    X_train, y_train = (X_trans[0][~fold_out_bool], X_trans[1][~fold_out_bool]), self.y[~fold_out_bool]
                    X_val, y_val = (X_trans[0][fold_out_bool], X_trans[1][fold_out_bool]), self.y[fold_out_bool]
                else:
                    X_train, y_train = X_trans[~fold_out_bool], self.y[~fold_out_bool]
                    X_val, y_val = X_trans[fold_out_bool], self.y[fold_out_bool]

                # Fit the model and differentiate between the model classes
                if model_name in ["nn", "lstm"]:
                    model_copy = clone_model(model)
                    model_copy.set_weights(model.get_weights())
                    model_copy.compile(optimizer="adam", loss=model.loss)
                    # Train the Keras model
                    model_copy.fit(X_train, y_train,
                                   epochs=params["epochs"],
                                   batch_size=params["batch_size"],
                                   verbose=0)

                    # Predict the target values
                    preds = model_copy.predict(X_val, verbose=0).flatten()
                    # clear session
                    #tf.keras.backend.clear_session()
                else:
                    model_copy = deepcopy(model)
                    # Fit the model
                    model_copy.fit(X_train, y_train)

                    # Predict the target values
                    preds = model_copy.predict(X_val)

                # Calculate the RMSE and append it to the list
                fold_mse = mean_squared_error(y_val, preds)
                mse_ls.append(fold_mse)
                del model_copy
        #if model_name in ["nn", "lstm"]:
            # clear session
            #tf.keras.backend.clear_session()
        del model
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
    def train_best_model(self, X, y, predictor_names=None):
        """
        This method applies the best parameters (found by optimization) to any dataset of X and y
        trains the optimal model and exports the model as well as the transformed X
        :param X: 2D np.array with predictors
        :param y: np.array of yields
        :param years: np.array of harvest years
        :param predictor_names: labels for the columns of X
        :return: X_trans, trained_model
        """
        assert self.best_params, ".apply() applies optimal parameters that are found by .optimize() or by setting them: self.best_params = {...}"

        if self.feature_set_selection | self.feature_len_shrinking:
            # transform X
            X, new_predictor_names = self.transform_X(X=X, predictor_names=predictor_names, params=self.best_params)
            assert X.shape[-1] > 0, "There are no features left after selection & shrinking."

        # initialize model
        if type(self.model_types) == str:
            model_type = self.model_types
        else:
            model_type = self.best_params["model_type"]

        model, _ = init_model(X=X, model_name=model_type, params=self.best_params)

        # Fit the model and differentiate between the model classes
        if model_type in ['nn', "lstm"]:
            # Train the Keras model
            model.fit(X, y,
                      epochs=self.best_params["epochs"],
                      batch_size=self.best_params["batch_size"],
                      verbose=0)
        else:
            # Fit the model
            model.fit(X, y)

        if self.feature_set_selection | self.feature_len_shrinking:
            return X, new_predictor_names, model
        else:
            return model

    def transform_X(self, X, predictor_names, trial=None, params=None):
        # check if inputs are reasonable: xor on trial and params
        assert (not trial) ^ (not params), "Choose either params OR give an optuna trial (it's xor)"

        # Feature Selection
        if self.feature_set_selection:
            if trial:
                X_sel, sel_predictor_names = select_features(X=X, predictor_names=predictor_names, trial=trial)
            else:
                X_sel, sel_predictor_names = select_features(X=X, predictor_names=predictor_names,
                                                             params=self.best_params)
        else:
            X_sel = X.copy()
            sel_predictor_names = predictor_names.copy()

        # Feature Shrinking
        if self.feature_len_shrinking:
            if trial:
                X_trans, new_predictor_names = self.shrink_features(X=X_sel, predictor_names=sel_predictor_names,
                                                                    trial=trial)
            else:
                X_trans, new_predictor_names = self.shrink_features(X=X_sel, predictor_names=sel_predictor_names,
                                                                    params=self.best_params)
        else:
            X_trans = X_sel
            new_predictor_names = sel_predictor_names

        return X_trans, new_predictor_names

    def shrink_features(self, X, predictor_names, trial=None, params=None):
        # check if inputs are reasonable: xor on trial and params
        assert (not trial) ^ (not params), "Choose either params OR give an optuna trial (it's xor)"

        # get remaining features that are time series (those which need to get sharnk)
        ts_features = np.unique(["_".join(x.split("_")[:-1]) for x in predictor_names if x.split("_")[-1].isdigit()])

        # make bool array for featues that will be shrinked (to replace them later with X_shrink)
        transformed_feature_columns = np.repeat(False, X.shape[1])
        # list of sharnk features that will replace the old X later
        X_shrink = []
        new_predictor_names = []

        for ts_feature in ts_features:
            ts_feature_loc = np.array([ts_feature in name for name in predictor_names])
            transformed_feature_columns = transformed_feature_columns + ts_feature_loc

            if trial:
                # define feature length
                feature_len = trial.suggest_int(ts_feature + '_len', 0,
                                                np.min([self.max_feature_len, sum(ts_feature_loc)]))
            else:
                feature_len = params[ts_feature + '_len']

            if feature_len == 0:
                continue

            if feature_len == 1:
                X_shrink.append(np.mean(X[:, ts_feature_loc], 1).reshape(-1, 1))
            else:
                X_shrink.append(rescale_array(X[:, ts_feature_loc], feature_len))
            new_predictor_names.extend([f"{ts_feature}_{i + 1}" for i in range(feature_len)])
        if np.any(~transformed_feature_columns):
            X_shrink.append(X[:, ~transformed_feature_columns])
            new_predictor_names.extend(predictor_names[~transformed_feature_columns])
        if len(X_shrink) >= 1:
            X_shrink = np.hstack(X_shrink)
        else:
            X_shrink = np.array([])

        return X_shrink, new_predictor_names
