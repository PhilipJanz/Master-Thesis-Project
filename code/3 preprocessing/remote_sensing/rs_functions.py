import os

import matplotlib.pyplot as plt
import pandas as pd
import rasterio


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
            rs_image = rs_image[8:-9]
            #plt.imshow(rs_image)
            #plt.show()
            date_list.append(file_name[18:28])
            rs_image_list.append(rs_image)

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
