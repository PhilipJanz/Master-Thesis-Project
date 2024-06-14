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


def loyocv_grid_search(X, y, years, model, param_grid):
    # Set up the GroupKFold
    group_kfold = GroupKFold(n_splits=len(np.unique(years)))

    # Set up the GridSearchCV with GroupKFold
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=group_kfold, scoring='neg_mean_squared_error')

    # Fit the grid search to the data
    grid_search.fit(X, y, groups=years)

    # Output the results
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)

    # Use the best model
    return grid_search.best_estimator_
