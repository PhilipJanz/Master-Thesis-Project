import numpy as np


"""
This script unites metrics used as evaluation methods.
"""


def calc_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))


def calc_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


def calc_r2(y_true, y_pred):
    y_mean = mean_estimator(y=y_true)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_mean)**2)
    return 1 - (ss_res / ss_tot)


def calc_brr2(y_true, y_pred, y_benchmark):
    """
    Calculated the Benchmark-realative R2 metric
    :param y_true:
    :param y_pred:
    :param y_benchmark:
    :return:
    """
    ss_res = np.sum((y_true - y_pred)**2)
    ss_benchmark = np.sum((y_true - y_benchmark)**2)
    return 1 - (ss_res / ss_benchmark)


def mean_estimator(y):
    """
    This is an out-of-sample mean estimator
    """
    y_mean = []
    for i in y.index:
        y_mean.append(np.mean(y.drop(i)))
    return y_mean
