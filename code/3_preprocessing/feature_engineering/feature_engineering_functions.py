import numpy as np
import pandas as pd
import math

from config import PROCESSED_DATA_DIR
from datetime import datetime, timedelta

from data_loader import make_adm_column

"""
This script collects functions that shape features out of time series.
It also processes
"""


def max_cdd(preci_array):
    """
    Max consecutive dry days (nealy zero precipitation)
    :param preci_array: precipitation data
    :return: int
    """
    max_count = 0
    current_count = 0

    for num in preci_array:
        if num < 1:
            current_count += 1
        else:
            max_count = max(max_count, current_count)
            current_count = 0

    # In case the longest streak ends at the last element
    max_count = max(max_count, current_count)

    return max_count


def process_features_dict(yield_df, cc_df, feature_location_dict, feature_engineering_dict, length,
                          start_before_sos, end_before_eos):
    """
    see process_feature_dict() docstring
    """
    processed_features_dict = {}
    for feature_name, feature_path in feature_location_dict.items():
        print(feature_name, end="\r")
        feature_df = pd.read_csv(PROCESSED_DATA_DIR / feature_path, keep_default_na=False)

        # apply CC for each yield datapoint
        processed_feature_dict = process_feature_dict(yield_df=yield_df,
                                                      cc_df=cc_df,
                                                      feature_df=feature_df,
                                                      feature_name=feature_name,
                                                      feature_engineering_func=feature_engineering_dict[feature_name],
                                                      length=length,
                                                      start_before_sos=start_before_sos,
                                                      end_before_eos=end_before_eos)
        # append to final dict
        processed_features_dict = processed_features_dict | processed_feature_dict
    return processed_features_dict


def process_feature_dict(yield_df, cc_df, feature_df, length, feature_name,
                         start_before_sos, end_before_eos, feature_engineering_func=None):
    """
    This function processes raw-data (in the form of time-series) into custom features.
    First set the length of the precessed time-series: to assure equal length no matter the underlying data.
    Set the feature_engineering_func according the aggregation on one part of the time-series eg. mean, sum, max ...

    Example:
        feature_name = "preci", length = 2, feature_engineering_func = {"sum": np.sum, "max": np.max}
        This will lead to four features: "preci_sum_1", "preci_sum_2", "preci_max_1", "preci_max_2"
        where 1 and 2 represent the first and second half of the time-series and np.sum and np.max are the funcitons
        applied on those two time-series.

    If feature_engineering_func is None it will simply take the average.

    :param yield_df: (DataFrame) processed yield df
    :param cc_df: (DataFrame) processed crop calendar df with 'sos' and 'eos' in day of year format
    :param feature_df: (DataFrame) raw time-series of feature
    :param length: (int) of shrinked timeseries
    :param feature_name: (str)
    :param start_before_sos: (int) in days
    :param end_before_eos: (int) in days
    :param feature_engineering_func: (optional) (dict) {"func_name": callable function, ...}
    :return: dict with processed feature as key and list of aggragated feature (length of yield_df) as values
    """
    # create common 'adm' column for faster matching of all dataframes
    yield_df = make_adm_column(yield_df.copy())
    cc_df = make_adm_column(cc_df.copy())
    feature_df = make_adm_column(feature_df.copy())

    # get columns of values and extract date form column name
    value_columns = np.array([column for column in feature_df.columns if column[:4].isdigit()])
    column_dates = np.array(
        [datetime.strptime(column.replace("02_30", "02_28"), '%Y_%m_%d') for column in value_columns])

    # make empty matrix to be filled in loop with feature values
    processed_feature_dict = {}
    if feature_engineering_func:
        if length == 1:
            for func_name, func in feature_engineering_func.items():
                processed_feature_dict[f"{feature_name}_{func_name}"] = []
        else:
            for func_name, func in feature_engineering_func.items():
                for i in range(length):
                    processed_feature_dict[f"{feature_name}_{func_name}_{i + 1}"] = []
    else:
        if length == 1:
            processed_feature_dict[feature_name] = []
        else:
            for i in range(length):
                processed_feature_dict[f"{feature_name}_{i + 1}"] = []

    # iterate over the yield values
    for _, row in yield_df.iterrows():
        start_point = cc_df.loc[cc_df.adm == row.adm, "sos"].values[0] - start_before_sos
        end_point = cc_df.loc[cc_df.adm == row.adm, "eos"].values[0] - end_before_eos
        start_date = year_day_to_date(year=row.harv_year - 1, day_of_year=start_point)
        end_date = year_day_to_date(year=row.harv_year, day_of_year=end_point)
        season_columns = value_columns[(column_dates >= start_date) & (column_dates <= end_date)]
        assert len(season_columns) > 0, f"Couldn't find {feature_name} data for {row}"

        # cut out the relevant data
        season_values = feature_df.loc[feature_df.adm == row.adm, season_columns].values[0].astype("float")

        # process feature (looks like it could be optimized)
        if feature_engineering_func:
            if length == 1:
                for func_name, func in feature_engineering_func.items():
                    value = func(season_values)
                    processed_feature_dict[f"{feature_name}_{func_name}"].append(value)
            else:
                segmented_timeseries = segment_array(arr=season_values, new_length=length)
                for i, partial_timeseries in enumerate(segmented_timeseries):
                    for func_name, func in feature_engineering_func.items():
                        value = func(partial_timeseries)
                        processed_feature_dict[f"{feature_name}_{func_name}_{i + 1}"].append(value)
        else:
            if length == 1:
                processed_feature_dict[feature_name].append(np.mean(season_values))
            else:
                rescale_values = rescale_array(arr=season_values, new_length=length)
                for i, value in enumerate(rescale_values):
                    processed_feature_dict[f"{feature_name}_{i + 1}"].append(value)

    return processed_feature_dict


def year_day_to_date(year, day_of_year):
    # Create a date for the first day of the given year
    first_day_of_year = datetime(year, 1, 1)

    # Add the given number of days to the first day of the year, subtracting 1 because timedelta(0) is the first day
    target_date = first_day_of_year + timedelta(days=int(day_of_year - 1))

    return target_date


def segment_array(arr, new_length):
    # Calculate the size of each segment
    segment_size = math.ceil(len(arr) / new_length)

    segmented_arr = [
        arr[int(i * segment_size):int((i + 1) * segment_size)]
        for i in range(new_length)
    ]
    if len(segmented_arr[-1]) == 0:
        segmented_arr[-1] = [arr[-1]]
    return segmented_arr


def rescale_array(arr, new_length):
    """
    Rescales a 1D or 2D numpy array to a new length using interpolation.

    Parameters:
    arr (np.ndarray): Input 1D or 2D array.
    new_length (int): The number of elements/columns in the output array.

    Returns:
    np.ndarray: The rescaled array.
    """
    if arr.ndim == 1:
        original_length = len(arr)
        if new_length == original_length:
            return arr
        elif new_length > original_length:
            # Create an array of the new indices, scaled appropriately
            new_indices = np.linspace(0, original_length - 1, new_length)

            # Interpolate the values at the new indices
            rescaled_arr = np.interp(new_indices, np.arange(original_length), arr)
        else:
            # Segment the data
            segmented_array = segment_array(arr=arr, new_length=new_length)

            rescaled_arr = np.array([np.mean(arr) for arr in segmented_array])

    elif arr.ndim == 2:
        num_rows, num_cols = arr.shape
        if new_length >= num_cols:
            return arr

        rescaled_arr = np.zeros((num_rows, new_length))

        for i in range(num_rows):
            rescaled_arr[i] = rescale_array(arr[i], new_length)

    else:
        raise ValueError("Input array must be 1D or 2D")

    return rescaled_arr
