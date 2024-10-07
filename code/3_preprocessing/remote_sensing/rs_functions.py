import os

import numpy as np
import pandas as pd
import rasterio
from scipy.optimize import curve_fit


def load_rs_data(rs_path, filter_filename=None):
    """

    :param rs_path:
    :param filter_filename:
    :return:
    """
    file_names = os.listdir(rs_path)
    if filter_filename:
        file_names = [file_name for file_name in file_names if filter_filename in file_name]

    # make lists for data collection
    date_list = []
    rs_image_list = []

    for file_name in file_names:
        # Load the remote sensing data
        with rasterio.open(rs_path / file_name) as rs_src:
            rs_image = rs_src.read(1)
            # the following shrinking makes sure the image is in the correct format (660, 520)
            #rs_image = rs_image[8:-9]
            #plt.imshow(rs_image)
            #plt.show()
            date_list.append(file_name[18:28])
            rs_image_list.append(rs_image)
            transform = rs_src.transform

            crs = rs_src.crs

    return rs_image_list, date_list


def custom_rolling_average(data, window):
    """
    Computes a rolling average for a list of numbers, filling NaN values with the average of the surrounding values.

    :param data: list or array of numbers (including NaN values)
    :param window: size of the rolling window
    :return: list of rolling averages with NaN values filled
    """

    # Convert data to a pandas Series
    series = pd.Series(data)

    # Ensure the series is of a numeric type
    series = series.astype(float)

    # Fill NaN values with the average of the surrounding values
    series = series.interpolate(method='linear', limit_direction='both')

    # Compute the rolling average
    rolling_avg = series.rolling(window=window, min_periods=1, center=True).mean()

    # Convert the result back to a list
    result = rolling_avg.tolist()
    return result


def fill_missing_values(doy_ls, data, N):
    # Define the Fourier series function
    def fourier_approximation(t, *a):
        ret = a[0] / 2  # a_0 / 2 term
        n_harmonics = (len(a) - 1) // 2
        for i in range(n_harmonics):
            ret += a[2 * i + 1] * np.sin(2 * np.pi * (i + 1) * t / 365) + a[2 * i + 2] * np.cos(2 * np.pi * (i + 1) * t / 365)
        return ret

    # setup data
    qualitative_data_loc = ~pd.isnull(data)
    qualitative_data = data[qualitative_data_loc]
    time_stamps = doy_ls
    qualitative_time_stamps = time_stamps[qualitative_data_loc]

    # Initial guess for the coefficients
    initial_guess = np.zeros(2 * N + 1)
    initial_guess[0] = np.max(qualitative_data) - np.min(qualitative_data)

    # Curve fitting
    params, params_covariance = curve_fit(fourier_approximation, qualitative_time_stamps, qualitative_data, p0=initial_guess)

    # Generate fitted values for desired time stamps
    fitted_values = fourier_approximation(time_stamps, *params)
    filled_data = data.copy()
    filled_data[~qualitative_data_loc] = fitted_values[~qualitative_data_loc]
    return time_stamps, fitted_values, filled_data
