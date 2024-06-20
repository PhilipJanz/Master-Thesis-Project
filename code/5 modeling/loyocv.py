# basic packages
import os
import numpy as np
import pandas as pd
import pickle
from copy import deepcopy
import matplotlib.pyplot as plt
from datetime import datetime
import time
import random
from collections import defaultdict

# preprocessing
from sklearn.preprocessing import StandardScaler

# models
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# parallelization
from concurrent.futures import ProcessPoolExecutor, as_completed

# code makeup
import warnings
from sklearn.exceptions import ConvergenceWarning

from training import train_and_predict

# Ignore ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def loyocv(X, y, years, model, model_name, folds=None, print_result=False):
    """
    model: initialized model with .fit and .predict
    """
    # check if years are sorted. If not the functions output y_preds would not align with the input array 'y'
    assert np.all(np.sort(years) == years), "Sort by year in your data preprocessing!"

    # shrink number of folds if wanted
    if folds:
        years = group_years(years=years, n=folds)

    # initialize arrays to be filled
    scores = []
    y_preds = np.repeat(np.nan, len(y))
    # first level leave-one-year-out
    for year_0 in np.unique(years):
        year_0_bool = (years != year_0)
        X_train = X[year_0_bool]
        y_train = y[year_0_bool]
        X_test = X[~year_0_bool]
        y_test = y[~year_0_bool]

        y_pred, mse = train_and_predict(X_train, y_train, X_test, y_test, model=model)

        # save the score & best hyperparameter
        scores.append(mse)
        y_preds[~year_0_bool] = y_pred

    model_mse = np.mean(scores)
    if print_result:
        print(model_name + ": Mean MSE across all folds:", np.round(model_mse, 3))
    return y_preds, model_mse


def loyocv_parallel(X, y, years, model, model_name, print_result=False):
    """
    model: initialized model with .fit and .predict
    """
    # check if years are sorted. If not the functions output y_preds would not align with the input array 'y'
    assert np.all(np.sort(years) == years), "Sort by year in your data preprocessing!"
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
        futures = [executor.submit(train_and_predict, X_train, y_train, X_test, y_test, model) for X_train, y_train, X_test, y_test in tasks]

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


def nested_loyocv(X, y, years, model, param_grid, print_result=False, parallel=False):
    """
    model_ls:  list of initialized models with .fit and .predict.
    """
    # check if years are sorted. If not the functions output y_preds would not align with the input array 'y'
    assert np.all(np.sort(years) == years), "Sort by year in your data preprocessing!"

    # initialize arrays to be filled
    # list for best model (hyperparameters)
    best_model_ls = []

    # list for best set of features (optional)
    if type(X) == list:
        best_feature_set_ix_ls = []

    # final predictions on unseen data
    y_preds = np.repeat(np.nan, len(y))

    # first level leave-one-year-out
    for year_0 in np.unique(years):
        year_0_bool = (years != year_0)

        # differentiate between only hyperparameter search or feature set search as well
        if type(X) != list: # only one X
            X_train = X[year_0_bool]
            y_train = y[year_0_bool]
            years_train = years[year_0_bool]
            X_test = X[np.invert(year_0_bool)]
            y_test = y[np.invert(year_0_bool)]

            best_model, _ = loyocv_grid_search(X_train, y_train, years_train, model=model, param_grid=param_grid)

            y_pred, mse = train_and_predict(X_train, y_train, X_test, y_test, model=best_model, plot_train_history=False)
        else: # multiple X
            mse_ls = []
            for X_i in X:
                X_train = X_i[year_0_bool]
                y_train = y[year_0_bool]
                years_train = years[year_0_bool]
                X_test = X_i[~year_0_bool]
                y_test = y[~year_0_bool]

                best_model, _ = loyocv_grid_search(X_train, y_train, years_train, model=model, param_grid=param_grid)

                y_pred, mse = train_and_predict(X_train, y_train, X_test, y_test, model=best_model, plot_train_history=False)

                mse_ls.append(mse)

        # save the score & best hyperparameter
        best_model_ls.append(best_model)
        y_preds[~year_0_bool] = y_pred
        print(f"{year_0}, best_model: {best_model}, mse: {np.round(mse, 3)}")

    print("Nested-LOYOCV finished! Mean MSE under selected models:", np.round(np.mean((y - y_preds) ** 2), 3))
    unique, counts = np.unique(best_model_ls, return_counts=True)
    print("Selected models:  (model_name: counts)", dict(zip(unique, counts)))
    return y_preds, best_model_ls


import numpy as np
from sklearn.model_selection import GridSearchCV, GroupKFold


def cv_grid_search(X, y, model, param_grid, cv=5):
    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=cv, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

    # Fit the model
    grid_search.fit(X, y)

    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best parameters found: ", best_params)

    # Use the best model
    return grid_search.best_estimator_


def loyocv_grid_search(X, y, years, model, param_grid, folds=None, print_result=False):
    # shrink number of folds if wanted
    if folds:
        years = group_years(years=years, n=folds)

    # Set up the GroupKFold
    group_kfold = GroupKFold(n_splits=len(np.unique(years)))

    # Set up the GridSearchCV with GroupKFold
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=group_kfold, scoring='neg_mean_squared_error')

    # Fit the grid search to the data
    grid_search.fit(X, y, groups=years)

    # Output the results
    if print_result:
        print(f"Best parameters found for {model.name}: ", grid_search.best_params_)
        print("corresponding cross-validation score: ", grid_search.best_score_)

    # Use the best model
    return grid_search.best_estimator_, np.abs(grid_search.best_score_)


def group_years(years, n):
    unique_years = list(set(years))
    random.shuffle(unique_years)

    # Create groups and distribute unique values randomly
    groups = defaultdict(list)
    for i, year in enumerate(unique_years):
        groups[i % n].append(year)

    # Map values to their groups
    year_to_group = {}
    for group_number, group_year in groups.items():
        for year in group_year:
            year_to_group[year] = group_number

    # Create the final output list with the same length as the input list
    grouped_years = [year_to_group[year] for year in years]

    return grouped_years
