from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import random
from collections import defaultdict

from sklearn.preprocessing import StandardScaler

from config import PROCESSED_DATA_DIR, SEED
from data_loader import make_adm_column


def make_X(df_ls, standardize=True):
    if standardize:
        standardize_mtx_ls = []
        column_ls = []
        for df in df_ls:
            if len(df.shape) == 1:
                standardized_values = (df.values - np.mean(df.values)) / np.std(df.values)
                standardize_mtx_ls.append(standardized_values.reshape(len(standardized_values), 1))
                column_ls.append([df.name])
            else:
                # filter columns where all values are the same
                columns_to_keep = [col for col in df.columns if df[col].nunique() > 1]
                df = df[columns_to_keep]

                standardize_mtx_ls.append((df.values - np.mean(df.values)) / np.std(df.values))
                column_ls.append(list(df.columns))

        X = np.concatenate(standardize_mtx_ls, axis=1)
        return X, np.concatenate(column_ls)
    else:
        X = pd.concat(df_ls, axis=1).values
        column_ls = np.concatenate([[df.name] if len(df.shape) == 1 else list(df.columns) for df in df_ls])
        return X, column_ls


def make_dummies(yield_df):
    yield_df = make_adm_column(yield_df.copy())
    dummy_df = pd.get_dummies(yield_df.adm) * 1
    return dummy_df


def make_X_y(df, include_year=True, features=None):
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
    return X, y, years, column_names


def group_years(years, n):
    unique_years = list(set(years))

    # Set the random seed
    rng = random.Random(SEED)
    rng.shuffle(unique_years)

    # Create groups and distribute unique values randomly
    groups = defaultdict(list)
    for i, year in enumerate(unique_years):
        groups[i % n].append(year)

    # Map values to their groups
    year_to_group = {}
    for group_number, group_year in groups.items():
        for year in group_year:
            year_to_group[year] = group_number

    # Create the final output list with the same length as the input list
    grouped_years = [year_to_group[year] for year in years]

    return np.array(grouped_years)
