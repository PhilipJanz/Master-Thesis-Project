from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop


def create_nn(optimizer='adam', learning_rate=0.001, activation='relu', neurons=32, depth=1, dropout_rate=0.0, input_shape=(None,)):
    model = Sequential()
    model.add(Dense(neurons, input_shape=input_shape, activation=activation))
    model.add(Dropout(dropout_rate))
    for _ in range(depth - 1):
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # Output layer for regression

    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Optimizer not recognized")
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model


# Define the parameter grid for LASSO
param_grid_lasso = {
    'alpha': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10, 100]
}

# Define the parameter grid for SVR
param_grid_svr = {
    'C': [1e-2, 0.1, 1, 10, 100],
    'epsilon': [0.001, 0.01, 0.1, 0.5, 1],
    'kernel': ['linear', 'poly', 'rbf']
}

# Define the parameter grid for random forest
param_grid_rf = {
    'n_estimators': [300],
    'max_depth': [None, 5, 10, 20],
    'max_features': ['log2', None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


# Define the parameter grid
param_grid_nn = {
    'model__neurons': [32, 64, 128],
    'model__depth': [1, 2],
    'model__dropout_rate': [0.0, 0.2],
    'model__activation': ['relu'],
    'model__optimizer': ['adam'], # , "rmsprop"
    'model__learning_rate': [0.001], # 0.001, 0.01, 0.1
    'epochs': [500],
    'batch_size': [20, 50, 100]
}

# define a collection of models to be used for yield estimation
model_ls = {"nn": "create_nn", "lasso": Lasso(),
            "svr": SVR(),
            "rf": RandomForestRegressor(),
            }
model_param_grid_ls = {"nn": param_grid_nn, "lasso": param_grid_lasso, "svr": param_grid_svr, "rf": param_grid_rf, }
