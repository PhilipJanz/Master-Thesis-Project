import numpy as np


"""
This script unites metrics used as evaluation methods.
"""


def calc_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))


def calc_r2(y_true, y_pred):
    mean_estimator = []
    for i in y_true.index:
        mean_estimator.append(np.mean(y_true.drop(i)))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - mean_estimator)**2)
    return 1 - (ss_res / ss_tot)


def calc_brr2(y_true, y_pred, y_benchmark):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_benchmark = np.sum((y_true - y_benchmark)**2)
    return 1 - (ss_res / ss_benchmark)
