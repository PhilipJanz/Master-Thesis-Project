import optuna
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

from xgboost import XGBRegressor

from config import RESULTS_DATA_DIR
from data_assembly import rescale_array
from feature_sets_for_optuna import feature_sets
from loyocv import group_years


def create_nn(trial, input_shape):
    model = Sequential()

    # Input layer
    model.add(Input(shape=(input_shape,)))

    # Hidden layers
    for i in range(trial.suggest_int('n_layers', 1, 3)):
        model.add(Dense(trial.suggest_int(f'units_layer{i+1}', 8, 256), activation='relu'))
        model.add(Dropout(trial.suggest_float(f'dropout_layer{i+1}', 0.0, 0.5)))

    # Output layer
    model.add(Dense(1, activation='linear'))

    # Compile model
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    return model


class OptunaOptimizer:
    def __init__(self,
                 X,
                 y,
                 years,
                 predictor_names,
                 feature_set_selection=True,
                 feature_len_shrinking=True,
                 model_types=['svr', 'rf', 'lasso', 'nn'],
                 sampler=optuna.samplers.TPESampler,
                 num_folds=5,
                 repetition_per_fold=1,
                 seed=42):
        # init
        self.X = X
        self.y = y
        self.years = years
        self.predictor_names = predictor_names
        self.feature_set_selection = feature_set_selection
        self.model_types = model_types
        self.num_folds = num_folds
        self.seed = seed
        self.repetition_per_fold = repetition_per_fold
        self.feature_set_selection = feature_set_selection
        self.feature_len_shrinking = feature_len_shrinking

        # Create a new Optuna study if it doesn't exist or load the existing one
        self.study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(self, trial):
        # create list of bools representing predictors from X to keep
        predictor_selection_bool = np.repeat(True, self.X.shape[1])

        if self.feature_set_selection:
            # Select True / False for each feature
            feature_set_selection_bool = [trial.suggest_categorical(name, [True, False]) for name in feature_sets]

            # list names of features not selected to filter them out
            left_out_feature_sets = [name for name, selected in zip(feature_sets, feature_set_selection_bool) if not selected]

            # set predictors False when they were not selected
            for left_out_feature_set in left_out_feature_sets:
                left_out_features = feature_sets[left_out_feature_set]
                predictor_selection_bool = predictor_selection_bool * [np.all([x not in predictor for x in left_out_features]) for predictor in self.predictor_names]

            # make new X with selected feature sets
            X_sel = self.X[:, predictor_selection_bool]
        else:
            X_sel = self.X.copy()

        if self.feature_len_shrinking:
            # get remaining features that are time series (those which need to get shrinked)
            ts_features = np.unique(["_".join(x.split("_")[:-1]) for x in self.predictor_names[predictor_selection_bool] if x.split("_")[-1].isdigit()])

            # make bool array for featues that will be shrinked (to replace them later with new_X)
            transformed_feature_columns = np.repeat(False, X_sel.shape[1])
            # list of shrinked features that will replace the old X later
            new_X = []

            for ts_feature in ts_features:
                # define feature length
                feature_len = trial.suggest_int(ts_feature + '_len', 1, 10)

                ts_feature_loc = np.array([ts_feature in name for name in self.predictor_names[predictor_selection_bool]])
                transformed_feature_columns = transformed_feature_columns + ts_feature_loc

                if feature_len == 1:
                    new_X.append(np.mean(X_sel[:, ts_feature_loc], 1).reshape(-1, 1))
                else:
                    new_X.append(rescale_array(X_sel[:, ts_feature_loc], feature_len))
            new_X.append(X_sel[:, ~transformed_feature_columns])
            X_sel = np.hstack(new_X)


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
        elif model_name == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 1, 20),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 100.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 100.0, log=True)
            }
        else:
            raise AssertionError(f"Model name '{model_name}' is not in: ['svr', 'rf', 'lasso', 'nn']")

        # Save the error of each fold in a list
        mse_ls = []

        # shrink number of folds if wanted else make loyoCV
        if self.num_folds < len(np.unique(self.years)):
            folds = group_years(years=self.years, n=self.num_folds, seed=self.seed)
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
                    model = create_nn(trial, input_shape)
                elif model_name == 'xgb':
                    model = XGBRegressor(**params)
                else:
                    raise AssertionError(f"Model name '{model_name}' is not in: ['svr', 'rf', 'lasso', 'nn']")

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

    def optimize(self, timeout=600, n_trials=100,
                   n_jobs=-1, show_progress_bar=True, print_result=True):
        self.study.optimize(self.objective, n_trials=n_trials, timeout=timeout,
                            n_jobs=n_jobs, show_progress_bar=show_progress_bar)

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

    def apply(self, X, y, years, predictor_names):
        """
        This method applies the best parameters (found by optimization) to any dataset of X and y
        :param X: 2D np.array with predictors
        :param y: np.array of yields
        :param years: np.array of harvest years
        :param predictor_names: labels for the columns of X
        :return: tran_X, y_pred
        """
        pass


