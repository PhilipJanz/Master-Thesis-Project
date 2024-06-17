from sklearn.linear_model import Lasso
from sklearn.svm import SVR


# Define the parameter grid for LASSO
param_grid_lasso = {
    'alpha': [1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10, 100]
}

# Define the parameter grid for SVR
param_grid_svr = {
    'C': [1e-2, 0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

# Define the parameter grid for random forest
param_grid_rf = {
    'n_estimators': [300],
    'max_depth': [None, 2, 5, 10, 20],
    'max_features': [1,  'log2', None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}


# Define the parameter grid
param_grid_nn = {
    'neurons': [32, 64, 128],
    'dropout_rate': [0.0, 0.2, 0.5],
    'activation': ['relu', 'tanh'],
    'optimizer': ['adam', 'rmsprop'],
    'epochs': [50, 100, 200],
    'batch_size': [10, 20]
}

# define a collection of models to be used for yield estimation
model_ls = {"lasso": Lasso(),
            "svr": SVR()}
model_param_grid_ls = {"lasso": param_grid_lasso,
                       "svr": param_grid_svr}
