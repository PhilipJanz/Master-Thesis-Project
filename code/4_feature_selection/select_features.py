import pickle

import numpy as np
import pandas as pd

from algorithm import mi_vif_selection
from config import PROCESSED_DATA_DIR, FEATURE_SELECTION_DIR
from data_loader import load_yield_data

"""
"""

# INITIALIZATION ###############

# indicator for feature file name given preprocessing specifications (see process_designed_features.py)
feature_file_name = "3_60_60"

# geographic scale of FS (country, adm, ...)
data_split = "country"

# VIF threshold
vif_threshold = 5


# DATA LOADING #####################

# load yield data and benchmark
yield_df = load_yield_data()

# load processed features
feature_path = PROCESSED_DATA_DIR / f"features/processed_designed_features_df_{feature_file_name}.csv"
feature_df = pd.read_csv(feature_path, keep_default_na=False)

# test if datasets have same row order
assert np.all(feature_df[["adm", "harv_year"]] == yield_df[["adm", "harv_year"]])

# add target variable to feature df
feature_df["y"] = yield_df["standardized_yield"]


# EXECUTION ########################

# create dict of selected varibles
feature_selection_dict = {}
for name, subset_df in feature_df.groupby(data_split):
    feature_selection_dict[name] = {}
    for leave_out_year in subset_df["harv_year"].unique():
        print(name, leave_out_year, end="\r")
        train_subset_df = subset_df[subset_df["harv_year"] != leave_out_year]
        y = train_subset_df["y"]
        X = train_subset_df.drop(['country', 'adm1', 'adm2', 'adm', 'harv_year', 'y'], axis=1)
        feature_selection_dict[name][leave_out_year] = mi_vif_selection(X=X, y=y, vif_threshold=vif_threshold)


with open(FEATURE_SELECTION_DIR / f"feature_selection_dir_{data_split}_vif{vif_threshold}_{feature_file_name}.pkl", 'wb') as f:
    # Pickle using the highest protocol available.
    pickle.dump(feature_selection_dict, f, pickle.HIGHEST_PROTOCOL)
