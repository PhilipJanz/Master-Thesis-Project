from optuna import TrialPruned

from config import SEED
from copy import deepcopy

import optuna
import numpy as np

import tensorflow as tf

#tf.keras.config.disable_interactive_logging()
tf.random.set_seed(SEED)
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, LSTM, Dense, Conv1D, Flatten, Bidirectional, concatenate

from data_assembly import group_years


def create_nn(c, trial=None, params=None):
    """
    This function creates a neural network with variable number of hidden layers and drop-out regularisation
    The parameter can be given by 'params' or will be chosen by optuna using 'trial'. Only one is possible
    :param trial: trial from optuna for hyperparameter optimization
    :param params: specific hyperparameters
    :return: Keras-nn ready to be trained. Make it run the extra mile!
    """
    # check if inputs are reasonable: xor on trial and params
    assert (not trial) ^ (not params), "Choose either params OR give an optuna trial (it's xor)"

    input = Input(shape=(c,), name='time_sensitive_input')
    dense_out = input
    # Hidden layers
    if trial:
        n_layers = trial.suggest_int('n_layers', 1, 3)
        for i in range(n_layers):
            dense_out = Dense(trial.suggest_int(f'units_layer{i + 1}', 1, 64), activation='sigmoid')(dense_out)
            dense_out = Dropout(trial.suggest_float(f'dropout_layer{i + 1}', 0.0, 0.5))(dense_out)
    else:
        for i in range(params["n_layers"]):
            dense_out = Dense(params[f'units_layer{i + 1}'], activation='sigmoid')(dense_out)
            dense_out = Dropout(params[f'dropout_layer{i + 1}'])(dense_out)

    # Output layer (regression example)
    output = Dense(units=1, activation='linear')(dense_out)

    # Create and compile the model
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam', loss='mse')

    return model


def create_cnn(t, f, s, trial=None, params=None):
    """
    This function creates a 1D CNN neural network with variable number of convolutional layers and dropout regularization.
    The parameters can be given by 'params' or will be chosen by Optuna using 'trial'. Only one is possible.
    :param t: length of time series
    :param f: number of temporal features (number of time series)
    :param s: number of static (non-temporal) features
    :param trial: trial from Optuna for hyperparameter optimization
    :param params: specific hyperparameters
    :return: Keras model ready to be trained.
    """

    # Check if inputs are reasonable: XOR on trial and params
    assert (trial is not None) ^ (params is not None), "Choose either params OR give an Optuna trial (it's xor)"

    # Define input shapes
    time_sensitive_input = Input(shape=(t, f), name='time_sensitive_input')
    static_input = Input(shape=(s,), name='static_input')

    # Hidden layers with dropout
    if trial:
        # Hyperparameter for static input
        static_dense_units = trial.suggest_categorical('static_dense_units', [4, 8, 16, 32, 64])
        static_dropout = trial.suggest_float('static_dropout', 0.0, 0.5)
        # Fully connected layer on static
        static_dense_out = Dense(units=static_dense_units, activation="tanh")(static_input)
        static_dense_out = Dropout(rate=static_dropout)(static_dense_out)

        # Convolutional layer hyperparameters
        num_conv_layers = trial.suggest_int('num_conv_layers', 1, 5)
        conv_filters = trial.suggest_categorical('conv_filters', [1, 2, 4, 8, 16, 32, 64])
        max_kernel_size = t / num_conv_layers
        kernel_size = trial.suggest_int('kernel_size', 2, np.min([16, max_kernel_size]))
        conv_dropout = trial.suggest_float('conv_dropout', 0.0, 0.5)

        # Build convolutional layers
        x = time_sensitive_input
        for i in range(num_conv_layers):
            x = Conv1D(filters=conv_filters,
                       kernel_size=kernel_size,
                       activation="relu",
                       padding='valid')(x)
            x = Dropout(rate=conv_dropout)(x)

        x = Flatten()(x)

        # Dense layer hyperparameters
        dense_units = trial.suggest_categorical('dense_units', [4, 8, 16, 32, 64])
    else:
        # Hyperparameter suggestions via trial
        static_dense_units = params['static_dense_units']
        static_dropout = params['static_dropout']
        static_dense_out = Dense(units=static_dense_units, activation="tanh")(static_input)
        static_dense_out = Dropout(rate=static_dropout)(static_dense_out)

        # Convolutional layer hyperparameters
        num_conv_layers = params['num_conv_layers']
        conv_filters = params['conv_filters']
        kernel_size = params['kernel_size']
        conv_dropout = params['conv_dropout']

        # Build convolutional layers
        x = time_sensitive_input
        for i in range(num_conv_layers):
            x = Conv1D(filters=conv_filters,
                       kernel_size=kernel_size,
                       activation="relu",
                       padding='valid')(x)
            x = Dropout(rate=conv_dropout)(x)

        x = Flatten()(x)

        # Dense layer hyperparameters
        dense_units = params['dense_units']

    # Concatenate with static data
    merged = concatenate([static_dense_out, x])

    # Final dense layer
    dense_out = Dense(units=dense_units, activation="tanh")(merged)

    # Output layer (regression example)
    output = Dense(units=1, activation='linear')(dense_out)

    # Create and compile the model
    model = Model(inputs=[static_input, time_sensitive_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')

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

    # Sample params if in a trial or extract them if given
    if trial:
        # sample params:

        # Hyperparameter for static input
        static_dense_units = trial.suggest_categorical('static_dense_units', [4, 8, 16, 32, 64])
        static_dropout = trial.suggest_float('static_dropout', 0.0, 0.5)

        # Hyperparameter for LSTM
        lstm_units = trial.suggest_categorical('lstm_units', [1, 2, 4, 8, 16, 32])
        lstm_dropout = trial.suggest_float('lstm_dropout', 0.0, 0.5)

        # last fully connected layer (feature layer)
        last_dense_units = trial.suggest_categorical(f'last_dense_units', [4, 8, 16, 32])
    else:
        # collect params
        static_dense_units = params[f'static_dense_units']
        static_dropout = params[f'static_dropout']
        lstm_units = params[f'lstm_units']
        lstm_dropout = params[f'lstm_dropout']
        last_dense_units = params[f'last_dense_units']


    # Define input shapes
    time_sensitive_input = Input(shape=(t, f), name='time_sensitive_input')
    static_input = Input(shape=(s,), name='static_input')

    # Fully connected layer on static input
    static_dense_out = Dense(units=static_dense_units, activation="tanh")(static_input)
    static_dense_out = Dropout(rate=static_dropout)(static_dense_out)

    # LSTM layers for time-sensitive data
    lstm_out = LSTM(units=lstm_units, return_sequences=True)(time_sensitive_input)
    lstm_out = Dropout(rate=lstm_dropout)(lstm_out)
    lstm_out = LSTM(units=lstm_units, return_sequences=True)(lstm_out)
    lstm_out = Dropout(rate=lstm_dropout)(lstm_out)
    lstm_out = LSTM(units=lstm_units)(lstm_out)
    lstm_out = Dropout(rate=lstm_dropout)(lstm_out)

    # Concatenate with static data
    merged = concatenate([static_dense_out, lstm_out])
    # init last hidden layer
    dense_out = Dense(units=last_dense_units, activation='tanh')(merged)

    # Output layer (regression example)
    output = Dense(units=1, activation='linear')(dense_out)

    # Create and compile the model
    model = Model(inputs=[static_input, time_sensitive_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')

    return model


def init_model(X, model_name, trial=None, params=None):
    # check if inputs are reasonable: xor on trial and params
    assert (not trial) ^ (not params), "Choose either params OR give an optuna trial (it's xor)"

    # Define model parameters
    if model_name == 'nn':
        if trial:
            params = {
                "epochs": trial.suggest_int('epochs', 500, 2000),
                "batch_size": trial.suggest_int('batch_size', 50, 500)
            }
            return create_nn(c=X.shape[1], trial=trial), params
        else:
            return create_nn(c=X.shape[1], params=params), params
    elif model_name == 'cnn':
        X_static, X_time = X
        _, s = X_static.shape
        _, t, f = X_time.shape
        if trial:
            params = {
                "epochs": trial.suggest_int('epochs', 100, 1000),
                "batch_size": trial.suggest_int('batch_size', 50, 500)
            }
            return create_cnn(t=t, f=f, s=s, trial=trial), params
        else:
            return create_cnn(t=t, f=f, s=s, params=params), params
    elif model_name == 'lstm':
        X_static, X_time = X
        _, s = X_static.shape
        _, t, f = X_time.shape
        if trial:
            params = {
                "epochs": trial.suggest_int('epochs', 100, 1000),
                "batch_size": trial.suggest_int('batch_size', 50, 500)
            }
            return create_lstm(t=t, f=f, s=s, trial=trial), params
        else:
            return create_lstm(t=t, f=f, s=s, params=params), params
    else:
        raise AssertionError(f"Model name '{model_name}' is not in: ['svr', 'rf', 'lasso', 'nn', 'lstm']")


class OptunaOptimizerTF:
    """

    :param X: (n x p)-array
    :param y: (n)-array
    :param years: (n)-array: important since we so 'leave-year-out-CV'
    :param best_params: usually None because that's the goal of optimization. But it can be used for model transfer
    :param model_type: model name in ['svr', 'rf', 'lasso', 'nn', 'xgb', 'lstm'
    :param sampler: optuna sampler
    :param num_folds: n-fold for CV. num_folds > n_unique_years results into leave-one-year-out-CV
    :param repetition_per_fold: make more reliable parameter choice by training models multiple times
    """
    def __init__(self,
                 study_name,
                 X,
                 y,
                 years,
                 model_type,
                 best_params=None,
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

        # Create a new Optuna study
        self.study = optuna.create_study(study_name=study_name, direction="minimize", sampler=sampler, pruner=pruner)

    def objective(self, trial):
        # Generate params for model
        model, params = init_model(X=self.X, model_name=self.model_type, trial=trial)
        # store initial weights
        init_weights = model.get_weights()

        # shrink number of folds if wanted else make loyoCV
        if self.num_folds < len(np.unique(self.years)):
            folds = group_years(years=self.years, n=self.num_folds)
        else:
            folds = self.years

        # CV
        # Save the error of each fold in a list
        mse_ls = []
        for _ in range(self.repetition_per_fold):
            for j, fold_out in enumerate(np.unique(folds)):
                fold_out_bool = (folds == fold_out)

                if self.model_type != "nn":
                    X_train, y_train = (self.X[0][~fold_out_bool], self.X[1][~fold_out_bool]), self.y[~fold_out_bool]
                    X_val, y_val = (self.X[0][fold_out_bool], self.X[1][fold_out_bool]), self.y[fold_out_bool]
                else:
                    X_train, y_train = self.X[~fold_out_bool], self.y[~fold_out_bool]
                    X_val, y_val = self.X[fold_out_bool], self.y[fold_out_bool]

                # Fit the model and differentiate between the model classes
                model_copy = model
                model_copy.set_weights(init_weights)
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

                # Calculate the RMSE and append it to the list
                fold_mse = np.mean((y_val - preds) ** 2)
                mse_ls.append(fold_mse)

                trial.report(fold_mse, step=j)

                if trial.should_prune():
                    raise TrialPruned()
        #if model_name in ["nn", "lstm"]:
            # clear session
            #tf.keras.backend.clear_session()
        #del model
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

    def train_best_model(self, X, y):
        """
        This method applies the best parameters (found by optimization) to any dataset of X and y
        trains the optimal model and exports the model as well as the transformed X
        :param X: 2D np.array with predictors
        :param y: np.array of yields
        :param years: np.array of harvest years
        :return: X_trans, trained_model
        """
        assert self.best_params, ".apply() applies optimal parameters that are found by .optimize() or by setting them: self.best_params = {...}"

        model, _ = init_model(X=X, model_name=self.model_type, params=self.best_params)

        # Train the Keras model
        model.fit(X, y,
                  epochs=self.best_params["epochs"],
                  batch_size=self.best_params["batch_size"],
                  verbose=0)

        return model
