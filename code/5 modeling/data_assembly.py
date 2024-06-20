from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler

from config import PROCESSED_DATA_DIR


def make_X(df_ls, standardize=True):
    if standardize:
        standardize_mtx_ls = []
        column_ls = []
        for df in df_ls:
            if len(df.shape) == 1:
                standardized_values = df.values - np.mean(df.values) / np.std(df.values)
                standardize_mtx_ls.append(standardized_values.reshape(len(standardized_values), 1))
                column_ls.append([df.name])
            else:
                # filter columns where all values are the same
                columns_to_keep = [col for col in df.columns if df[col].nunique() > 1]
                df = df[columns_to_keep]

                standardize_mtx_ls.append(df.values - np.mean(df.values) / np.std(df.values))
                column_ls.append(list(df.columns))

        X = np.concatenate(standardize_mtx_ls, axis=1)
        return X, np.concatenate(column_ls)
    else:
        X = pd.concat(df_ls, axis=1).values
        column_ls = np.concatenate([[df.name] if len(df.shape) == 1 else list(df.columns) for df in df_ls])
        return X, column_ls


def process_list_of_feature_df(yield_df, cc_df, feature_dict, length, start_before_sos, end_before_eos):
    processed_feature_df_dict = {}
    for feature_name, feature_path in feature_dict.items():
        feature_df = pd.read_csv(PROCESSED_DATA_DIR / feature_path, keep_default_na=False)

        # apply CC for each yield datapoint
        processed_feature_df = process_feature_df(yield_df=yield_df,
                                                  cc_df=cc_df,
                                                  feature_df=feature_df,
                                                  feature_name=feature_name,
                                                  length=length,
                                                  start_before_sos=start_before_sos,
                                                  end_before_eos=end_before_eos)
        processed_feature_df_dict[feature_name] = processed_feature_df
    return processed_feature_df_dict


def process_feature_df(yield_df, cc_df, feature_df, length, feature_name, start_before_sos, end_before_eos):
    # create common 'adm' column for faster matching of all dataframes
    yield_df = make_adm_column(yield_df.copy())
    cc_df = make_adm_column(cc_df.copy())
    feature_df = make_adm_column(feature_df.copy())

    # get columns of values and extract date form column name
    value_columns = np.array([column for column in feature_df.columns if column[:4].isdigit()])
    column_dates = np.array([datetime.strptime(column.replace("02_30", "02_28"), '%Y_%m_%d') for column in value_columns])

    # make empty matrix to be filled in loop with feature values
    processed_feature_mtx = []

    # iterate over the yield values
    for i, row in yield_df.iterrows():
        start_point = cc_df.loc[cc_df.adm == row.adm, "sos"].values[0] - start_before_sos
        end_point = cc_df.loc[cc_df.adm == row.adm, "eos"].values[0] - end_before_eos
        start_date = year_day_to_date(year=row.harv_year - 1, day_of_year=start_point)
        end_date = year_day_to_date(year=row.harv_year, day_of_year=end_point)
        season_columns = value_columns[(column_dates >= start_date) & (column_dates <= end_date)]
        assert len(season_columns) > 0

        # cut  out the relevant data
        season_values = feature_df.loc[feature_df.adm == row.adm, season_columns].values[0].astype("float")

        # rescale the time series to target length
        if length == 1:
            season_values = np.mean(season_values)
        elif length == "mmm":
            # max mean min
            season_values = np.array([np.max(season_values), np.mean(season_values), np.min(season_values)])
        else:
            season_values = rescale_array(season_values, length)

        processed_feature_mtx.append(season_values)

    if length == 1:
        return pd.DataFrame(processed_feature_mtx, columns=[feature_name + "_" + "mean"])
    elif length == "mmm":
        return pd.DataFrame(processed_feature_mtx, columns=[feature_name + "_" + str(i) for i in ["max", "mean", "min"]])
    else:
        return pd.DataFrame(processed_feature_mtx, columns=[feature_name + "_" + str(i) for i in range(1, length + 1)])


def make_dummies(yield_df):
    yield_df = make_adm_column(yield_df.copy())
    dummy_df = pd.get_dummies(yield_df.adm) * 1
    return dummy_df


def make_adm_column(df):
    """
    Create common 'adm' column for faster matching of dataframes
    It units country, adm1 and adm2
    """
    df["adm"] = [str(x).replace("'", "").replace(" None", "") for x in df[["country", "adm1", "adm2"]].values]
    return df


def year_day_to_date(year, day_of_year):
    # Create a date for the first day of the given year
    first_day_of_year = datetime(year, 1, 1)

    # Add the given number of days to the first day of the year, subtracting 1 because timedelta(0) is the first day
    target_date = first_day_of_year + timedelta(days=int(day_of_year - 1))

    return target_date


def rescale_array(arr, new_length):
    original_length = len(arr)
    if new_length == original_length:
        return arr

    # Calculate the ratio of the new length to the original length
    ratio = new_length / original_length

    # Create an array of the new indices, scaled appropriately
    new_indices = np.linspace(0, original_length - 1, new_length)

    # Interpolate the values at the new indices
    rescaled_arr = np.interp(new_indices, np.arange(original_length), arr)

    return rescaled_arr


def make_X_y(df, dummies=True, include_year=True, features=None):
    assert all(df.columns[:5] == ['country', 'adm1', 'adm2', 'harv_year', 'yield'])
    y = np.array(df["yield"])
    X = []  # Features (2D array)

    years = np.array(df.harv_year)

    X_df = df.iloc[:, 5:]

    if features:
        X_df = X_df.loc[:, features]

    if include_year:
        # the last columns of X will be always the year
        X_df["harv_year"] = years.astype("float")

    # standardize metrical columns
    scaler = StandardScaler()
    columns_to_scaled = np.where(np.max(X_df, axis=0) > 10)[0]
    X_df.iloc[:, columns_to_scaled] = scaler.fit_transform(X_df.iloc[:, columns_to_scaled])
    column_names = X_df.columns
    #if include_year:
    # multiplying by a large number means the coeeficient can be very small and wont have any weight for the penelization term
    #X_df["harv_year"] = X_df["harv_year"] * 1e6
    return X, y, years, column_names
