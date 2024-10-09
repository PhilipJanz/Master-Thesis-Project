import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

from config import SEED


def mi_vif_selection(X, y, vif_threshold=5):
    """
    Perform feature selection based on Mutual Information (MI) and Variance Inflation Factor (VIF).

    Parameters:
    X : pandas DataFrame
        The feature matrix.
    y : pandas Series or numpy array
        The response variable.
    vif_threshold : float
        The VIF threshold to remove features with multicollinearity.

    Returns:
    selected_features : list
        List of selected feature names after MI and VIF selection.
    """

    # Step 1: Calculate Mutual Information between each feature and the target variable
    if X.shape[0] > 5:
        mi = mutual_info_regression(X, y, random_state=SEED)
    else:
        mi = corr_based_mutual_info(X, y)

    mi_series = pd.Series(mi, index=X.columns)
    mi_series[mi_series > 0]

    # Step 2: Sort features based on MI in descending order
    mi_sorted = mi_series.sort_values(ascending=False)

    # Step 3: Initialize empty set for selected features
    selected_features = []

    # Standardize the feature matrix for VIF calculation
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Step 4: Forward selection based on MI, and VIF evaluation
    for feature in mi_sorted.index:
        selected_features.append(feature)
        current_X = X_scaled[selected_features]

        # the first feature will be taken without condition
        if current_X.shape[1] == 1:
            continue

        # Calculate VIF for the current set of selected features
        vif_ls = [variance_inflation_factor(current_X.values, i) for i in range(current_X.shape[1])]

        # Step 5: If any feature has VIF > vif_threshold, remove the last added feature
        if np.max(vif_ls) > vif_threshold:
            selected_features.remove(feature)

    return selected_features


def corr_based_mutual_info(X, y):
    correlations = pd.concat([X, y], axis=1).corr().values[:-1, -1]
    abs_correlation = np.abs(correlations)
    return abs_correlation
