import time

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from sklearn.ensemble import RandomForestRegressor

from config import PROCESSED_DATA_DIR, RESULTS_DATA_DIR
from crop_calendar.crop_calendar_functions import load_my_cc
from cv_grid_search import cv_grid_search, loyocv_grid_search
from feature_sets import feature_sets
from models import model_ls, model_param_grid_ls
from data_assembly import process_feature_df, process_list_of_feature_df
from loyocv import loyocv, loyocv_parallel
from soil.soil_functions import load_soil_data

"""
This script is the final script that brings together all processed data to make groundbreaking yield predictions.
It searches for the best combination of: predictive model (architecture and hyperparameters), sets of data cluster and feature set

To make the code more slim we dont loop over the sets of clusters (all, country, adm, ndvi-cluster, ...)
This has to be specified in the beginning of the code.

There are two main loops:
1. model architecture (Lasso, RF, NN ...)
    2. loop over clusters (eg. single countries)
        inside the loop a nested-LOYOCV strategy is applied
        the inner LOYOCV estimates the parameters performance and picks the best hyperparameters and feature set
        the outer LOYOCV evaluated the best models performance on entirely unseen data
        the picked hyperparameters and feature set are saved  for later investigation
"""

# LOAD & PREPROCESS ############

# load yield data and benchmark
yield_df = pd.read_csv(RESULTS_DATA_DIR / f"benchmark/yield_benchmark_prediction.csv", keep_default_na=False)

# load crop calendar (CC)
cc_df = load_my_cc()

# load and process features
processed_feature_df_dict = process_list_of_feature_df(yield_df=yield_df, cc_df=cc_df, feature_dict=feature_sets["all"],
                                                       length="mmm", start_before_sos=30, end_before_eos=60)

# load soil characteristics
soil_df = load_soil_data()
soil_df = pd.merge(yield_df[["country", "adm1", "adm2"]], soil_df, on=["country", "adm1", "adm2"], how="left")


# INITIALIZATION ##############

cluster_set = "all"

# 1. loop over model architecture
for model in model_ls:

    # 2. loop over cluster
    if cluster_set == ""

