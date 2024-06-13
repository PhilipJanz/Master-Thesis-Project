from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from config import PROCESSED_DATA_DIR
from yield_.yield_functions import load_yield_data


def make_profiles(data_df):
    # detect value columns in dataframes and save day of year for fitting later
    value_columns = [column for column in data_df.columns if column[:4].isdigit()]
    day_of_year = [get_day_of_year(date) for date in value_columns]

    # profile matrix to be filled:
    profile_mtx = []

    for i, row in data_df.iterrows():
        # extract values
        values = row[value_columns].values.astype(dtype="float")

        # fit the data by a fourier series
        days, fourier_fitted_values = fit_fourier_series(days=day_of_year, values=values, N=5)

        profile_mtx.append(fourier_fitted_values)

    return np.array(profile_mtx)


def get_day_of_year(date_str):
    if date_str[5:] == "02_30":
        date_str = date_str[:5] + "02_28"
    # Parse the date string into a datetime object
    date = datetime.strptime(date_str, '%Y_%m_%d')
    # Get the day of the year
    day_of_year = date.timetuple().tm_yday
    return day_of_year


def fit_fourier_series(days, values, N):
    """
    Fit a Fourier series to the given data.

    Parameters:
    days (array): Array of days of the year.
    values (array): Array of values corresponding to the days.
    N (int): Number of harmonics for the Fourier series.

    Returns:
    fitted_values (array): Fitted values for each day from 1 to 365.
    """
    # Define the Fourier series function
    def fourier_series(t, *a):
        ret = a[0] / 2  # a_0 / 2 term
        n_harmonics = (len(a) - 1) // 2
        for i in range(n_harmonics):
            ret += a[2 * i + 1] * np.cos(2 * np.pi * (i + 1) * t / 365) + a[2 * i + 2] * np.sin(2 * np.pi * (i + 1) * t / 365)
        return ret

    # Initial guess for the coefficients
    initial_guess = np.zeros(2 * N + 1)

    # Curve fitting
    params, params_covariance = curve_fit(fourier_series, days, values, p0=initial_guess)

    # Generate fitted values for each day from 1 to 365
    days = np.arange(1, 366)
    fitted_values = fourier_series(days, *params)

    return days, fitted_values


def fit_polynomial(days, values, degree):
    """
    Fit a polynomial to the given data.

    Parameters:
    days (array): Array of days of the year.
    values (array): Array of values corresponding to the days.
    degree (int): Degree of the polynomial.

    Returns:
    fitted_values (array): Fitted values for each day from 1 to 365.
    """
    # Fit the polynomial
    coefs = np.polyfit(days, values, degree)
    poly = np.poly1d(coefs)

    # Generate fitted values for each day from 1 to 365
    days = np.arange(1, 366)
    fitted_values = poly(days)

    return days, fitted_values
