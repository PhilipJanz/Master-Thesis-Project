import optuna
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

from config import RESULTS_DATA_DIR
from feature_sets_for_optuna import feature_sets
from loyocv import group_years


def create_nn(trial, input_shape):
    model = Sequential()

    # Input layer
    model.add(Input(shape=(input_shape,)))

    # Hidden layers
    for i in range(trial.suggest_int('n_layers', 0, 3)):
        model.add(Dense(trial.suggest_int(f'units_layer{i+1}', 4, 256), activation='relu'))
        model.add(Dropout(trial.suggest_float(f'dropout_layer{i+1}', 0.0, 0.5)))

    # Output layer
    model.add(Dense(1, activation='linear'))

    # Compile model
    model.compile(optimizer=trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd']),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    return model


class OptunaOptimizer:
    def __init__(self,
                 X,
                 y,
                 years,
                 feature_set_selection=True,
                 model_types=['svr', 'rf', 'lasso', 'nn'],
                 num_folds=5,
                 repetition_per_fold=2,
                 num_trials=100,
                 global_seed=42):
        # init
        self.X = X
        self.y = y
        self.years = years
        self.feature_set_selection = feature_set_selection
        self.model_types = model_types
        self.num_folds = num_folds
        self.repetition_per_fold = repetition_per_fold
        self.num_trials = num_trials
        self.global_seed = global_seed
        self.feature_set_selection = feature_set_selection

        # Create a new Optuna study if it doesn't exist or load the existing one
        self.study = optuna.create_study(direction="minimize")

    def objective(self, trial):
        if self.feature_set_selection:
            # Select True / False for each feature
            feature_set_selection_bool = [trial.suggest_categorical(name, [True, False]) for name in feature_sets]

            # list names of features not selected to filter them out
            left_out_feature_sets = [name for name, selected in zip(feature_sets, feature_set_selection_bool) if not selected]

            # create list of bools representing predictors from X to keep
            predictor_selection_bool = np.repeat(True, self.X.shape[1])
            # set predictors False when they were not selected
            for left_out_feature_set in left_out_feature_sets:
                left_out_features = feature_sets[left_out_feature_set]
                predictor_selection_bool = predictor_selection_bool * [np.all([x not in predictor for x in left_out_features]) for predictor in predictor_names]

            # make new X with selected feature sets
            X_sel = self.X[:, predictor_selection_bool]
        else:
            X_sel = self.X.copy()

        # Select model type
        model_name = trial.suggest_categorical('model_type', self.model_types)

        # Define model parameters
        if model_name == 'svr':
            params = {
                'C': trial.suggest_float('C', 1e-7, 100.0, log=True),
                'epsilon': trial.suggest_float('epsilon', 1e-5, 10.0, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            }
        elif model_name == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 500),
                'max_depth': trial.suggest_int('max_depth', 1, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20)
            }
        elif model_name == 'lasso':
            params = {
                'alpha': trial.suggest_float('alpha', 1e-9, 100.0, log=True)
            }
        elif model_name == 'nn':
            input_shape = X_sel.shape[1]
            epochs = trial.suggest_int('epochs', 10, 500)
            batch_size = trial.suggest_int('batch_size', 1, len(X_sel))
        else:
            raise AssertionError(f"Model name '{model_name}' is not in: ['svr', 'rf', 'lasso', 'nn']")

        # Save the error of each fold in a list
        mse_ls = []

        # shrink number of folds if wanted else make loyoCV
        if self.num_folds < len(np.unique(self.years)):
            folds = group_years(years=self.years, n=self.num_folds)
        else:
            folds = self.years.copy()

        # CV
        for _ in range(self.repetition_per_fold):
            for fold_out in np.unique(folds):
                fold_out_bool = (folds == fold_out)

                X_train, y_train = X_sel[~fold_out_bool], self.y[~fold_out_bool]
                X_val, y_val = X_sel[fold_out_bool], self.y[fold_out_bool]

                # Reinitialize the model within the loop
                if model_name == 'svr':
                    model = SVR(**params)
                elif model_name == 'rf':
                    model = RandomForestRegressor(**params)
                elif model_name == 'lasso':
                    model = Lasso(**params)
                elif model_name == 'nn':
                    model = create_nn(trial, input_shape)  # Create a fresh untrained model

                # Fit the model and diffrentiate between the model classes
                if model_name == 'nn':
                    # Train the Keras model
                    model.fit(X_train, y_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              verbose=0)

                    # Predict the target values
                    preds = model.predict(X_val, verbose=0).flatten()
                else:
                    # Fit the model
                    model.fit(X_train, y_train)

                    # Predict the target values
                    preds = model.predict(X_val)

                # Calculate the RMSE and append it to the list
                fold_mse = mean_squared_error(y_val, preds)
                mse_ls.append(fold_mse)

        return np.mean(mse_ls)

    def optimize(self, timeout=600,
                   n_jobs=-1, show_progress_bar=True, gc_after_trial=True, print_result=True):
        self.study.optimize(self.objective, n_trials=self.num_trials, timeout=timeout,
                            n_jobs=n_jobs, show_progress_bar=show_progress_bar) # TODO , gc_after_trial=gc_after_trial

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
