
import numpy as np
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor


def backwards_feature_selection(X, y, feature_names):
    assert X.shape[1] == len(feature_names)

    # backward selection feature selection
    backward_fs = SequentialFeatureSelector(XGBRegressor(), forward=False, k_features="best")

    X_select = backward_fs.fit_transform(X, y)
    selected_feature_names = [feature_names[i] for i in backward_fs.k_feature_idx_]

    return X_select, selected_feature_names, backward_fs


def calculate_vif(X, feature_names):
    """
    Calculate the Variance Inflation Factor (VIF) for each feature.

    Parameters:
    X : ndarray or DataFrame
        The matrix of features.
    feature_names : list
        The list of feature names.

    Returns:
    DataFrame
        DataFrame containing the VIF values for each feature.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_names)

    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['vif'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif_data


def feature_selection_vif(X, feature_names, indicator_features, threshold=5.0):
    """
    Perform feature selection based on VIF.

    Parameters:
    X : ndarray or DataFrame
        The matrix of features.
    feature_names : list
        The list of feature names.
    threshold : float, optional
        The VIF threshold for removing features (default is 5.0).

    Returns:
    ndarray, list
        The reduced feature matrix and the list of selected feature names.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_names)

    while True:
        if X.shape[1] == 1:
            break

        vif_data = calculate_vif(X.values, X.columns)
        fixed_ix = np.array([name not in indicator_features for name in feature_names])
        max_vif = vif_data[~fixed_ix].iloc[vif_data['vif'][~fixed_ix].argmax()]

        if max_vif.vif > threshold:
            X = X.drop(columns=[max_vif.feature])
            feature_names = np.delete(feature_names, max_vif.name)
        else:
            break

    return X.values, X.columns.tolist()