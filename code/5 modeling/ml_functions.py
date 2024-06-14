# basic packages
import os
import numpy as np
import pandas as pd
import pickle
from copy import deepcopy
import matplotlib.pyplot as plt
from datetime import datetime

# preprocessing
from sklearn.preprocessing import StandardScaler

# models
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
#import tensorflow as tf

# parallelization
#from concurrent.futures import ProcessPoolExecutor, as_completed

# code makeup
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignore ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def train_and_predict(X_train, y_train, X_test, y_test, model, epochs=None, learning_rate=None, batch_size=None):
    """
    Train the model and predict the test set.
    This function is designed to run in parallel for each cross-validation fold.
    """
    if epochs: # Keras NN
        my_model = tf.keras.models.clone_model(model)
        my_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=tf.keras.losses.MeanSquaredError())
        history = my_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)#, validation_data=(X_test, y_test))
        """loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        plt.figure(figsize=(10, 6))
        plt.plot(loss, 'bo-', label='Training loss', alpha=0.7)
        plt.plot(val_loss, 'ro-', label='Validation loss', alpha=0.7)
        plt.hlines(xmin=0, xmax=len(loss), y=min(val_loss), color="darkgrey", label=f"Minimum val loss: {round(min(val_loss), 3)}, ({np.argmin(val_loss)}); Final val-loss: {round(val_loss[-1], 3)}")
        plt.title(f'Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.ylim([0.03, 5])  # Be cautious with log scale: 0 cannot be shown on a log scale
        plt.yscale('log')
        plt.legend()
        plt.show()"""
        y_pred = my_model.predict(X_test,verbose=0).T[0]
    else: # Sklearn
        my_model = deepcopy(model)
        my_model.fit(X_train, y_train)
        y_pred = my_model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    return y_pred, mse


def loyocv(X, y, years, model, model_name, epochs=None, batch_size=None, learning_rate=None, print_result=False):
    """
    model: initialized model with .fit and .predict
    """
    # check if years are sorted. If not the functions output y_preds would not align with the input array 'y'
    assert all(np.sort(years) == years), "Sort by year in your data preprocessing!"
    # initialize arrays to be filled
    scores = []
    y_preds = np.repeat(np.nan, len(y))
    # first level leave-one-year-out
    for year_0 in np.unique(years):
        year_0_bool = (years != year_0)
        X_train = X[year_0_bool]
        y_train = y[year_0_bool]
        X_test = X[np.invert(year_0_bool)]
        y_test = y[np.invert(year_0_bool)]

        y_pred, mse = train_and_predict(X_train, y_train, X_test, y_test,
                                        model=model,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        learning_rate=learning_rate)

        # save the score & best hyperparameter
        scores.append(mse)
        y_preds[np.invert(year_0_bool)] = y_pred

    model_mse = np.mean(scores)
    if print_result:
        print(model_name + ": Mean MSE across all folds:", np.round(model_mse, 3))
    return y_preds, model_mse


def loyocv_parallel(X, y, years, model, model_name, print_result=False):
    """
    model: initialized model with .fit and .predict
    """
    # check if years are sorted. If not the functions output y_preds would not align with the input array 'y'
    assert all(np.sort(years) ==  years), "Sort by year in your data preprocessing!"
    # initialize arrays to be filled
    scores = []
    y_preds = np.repeat(np.nan, len(y))

    # Prepare data for each fold
    tasks = []
    for year_0 in np.unique(years):
        year_0_bool = (years != year_0)
        X_train = X[year_0_bool]
        y_train = y[year_0_bool]
        X_test = X[~year_0_bool]
        y_test = y[~year_0_bool]
        tasks.append((X_train, y_train, X_test, y_test))

    # Execute tasks in parallel
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(train_and_predict, X_train, y_train, X_test, y_test, deepcopy(model)) for X_train, y_train, X_test, y_test in tasks]

        for future in as_completed(futures):
            y_pred, mse = future.result()
            # Identify the test set for the current fold
            _, _, _, y_test = tasks[futures.index(future)]
            # Update predictions and scores
            y_preds[~(years == np.unique(years)[futures.index(future)])] = y_pred
            scores.append(mse)

    model_mse = np.mean(scores)
    if print_result:
        print(model_name + ": Mean MSE across all folds:", np.round(model_mse, 3))
    return y_preds, model_mse


def nested_loyocv(X, y, years, model_ls, model_name_ls, epochs=None, batch_size=None, learning_rate=None, print_result=False, parallel=False):
    """
    model_ls:  list of initialized models with .fit and .predict.
    """
    # check if years are sorted. If not the functions output y_preds would not align with the input array 'y'
    assert all(np.sort(years) ==  years), "Sort by year in your data preprocessing!"
    # initialize arrays to be filled
    best_model_ls = []
    y_preds = np.repeat(np.nan, len(y))
    # first level leave-one-year-out
    for year_0 in np.unique(years):
        year_0_bool = (years != year_0)

        X_train = X[year_0_bool]
        y_train = y[year_0_bool]
        years_train = years[year_0_bool]
        X_test = X[np.invert(year_0_bool)]
        y_test = y[np.invert(year_0_bool)]

        scores = []
        for model, model_name in zip(model_ls, model_name_ls):
            # second level loyocv to determine the best model (hyperparameter) for that split
            if parallel:
                _, model_mse = loyocv_parallel(X=X_train, y=y_train, years=years_train, model=model, model_name=model_name, print_result=print_result)
            else:
                _, model_mse = loyocv(X=X_train, y=y_train, years=years_train, model=model, model_name=model_name, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, print_result=print_result)
            # save the scores
            scores.append(model_mse)
            # the following line is a bit cheating but it is reasonable from my personal experience. When the performance drops with higher hypers: stop process
            if len(scores) >= 2:
                if scores[-1] > scores[-2]:
                    break
        # choose the best performing hyperparameters for testing on the hold out year
        ix_best_model = np.argmin(scores)

        if epochs:  # NN based on keras
            best_model = tf.keras.models.clone_model(model_ls[ix_best_model])
            best_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=tf.keras.losses.MeanSquaredError())
            history = best_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_test, y_test))
            loss = history.history["loss"]
            val_loss = history.history["val_loss"]
            plt.figure(figsize=(10, 6))
            plt.plot(loss, 'bo-', label='Training loss', alpha=0.7)
            plt.plot(val_loss, 'ro-', label='Validation loss', alpha=0.7)
            plt.hlines(xmin=0, xmax=len(loss), y=min(val_loss), color="darkgrey", label=f"Minimum val loss: {round(min(val_loss), 3)}, ({np.argmin(val_loss)}); Final val-loss: {round(val_loss[-1], 3)}")
            plt.title(f'Training and Validation Loss, mode: {model_name_ls[ix_best_model]}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.ylim([0.03, 5])  # Be cautious with log scale: 0 cannot be shown on a log scale
            plt.yscale('log')
            plt.legend()
            plt.show()
            y_pred = best_model.predict(X_test, verbose=0).T[0]
        else: # Sklearn models
            best_model = deepcopy(model_ls[ix_best_model])
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)

        # save the score & best hyperparameter
        best_model_ls.append(model_name_ls[ix_best_model])
        y_preds[np.invert(year_0_bool)] = y_pred
        print(year_0, ", best_model: ", str(model_name_ls[ix_best_model]), ", mse: ", str(round(np.mean((y_test - y_pred) ** 2), 3)))

    print("Nested-LOYOCV finished! Mean MSE under selected models:", np.round(np.mean((y - y_preds) ** 2), 3))
    unique, counts = np.unique(best_model_ls, return_counts=True)
    print("Selected models:  (model_name: counts)", dict(zip(unique, counts)))
    return y_preds, best_model_ls

