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
from grid_search_modeling.models import param_grid_rf
from data_assembly import process_feature_df
from loyocv import loyocv, loyocv_parallel

"""
This script supports the feature selection by assessing the isolated predictive power of each feature.
The model is a random forest. Initial hyperparameter selection is recommended.
Also choose the length of feature timeseries per season (eg. 1, 2, 4, 8) each feature might have a different 
optimal length.
"""

# load yield data and benchmark
yield_df = pd.read_csv(RESULTS_DATA_DIR / f"benchmark/yield_benchmark_prediction.csv", keep_default_na=False)

# load crop calendar (CC)
cc_df = load_my_cc()

# load features and make feature dataframe for yield data with respect to the CC
feature_ls = ["remote sensing/smooth_ndvi_regional_matrix.csv", "remote sensing/si_smooth_ndvi_regional_matrix.csv",
              "remote sensing/smooth_evi_regional_matrix.csv", "remote sensing/si_smooth_evi_regional_matrix.csv",
              "climate/pr_sum_regional_matrix.csv", "climate/si_pr_sum_regional_matrix.csv",
              "climate/pr_max_regional_matrix.csv", "climate/pr_cddMax_regional_matrix.csv",
              "climate/pr_belowP01_regional_matrix.csv", "climate/pr_aboveP99_regional_matrix.csv",
              "climate/tas_median_regional_matrix.csv", "climate/si_tas_median_regional_matrix.csv",
              "climate/tasmin_median_regional_matrix.csv", "climate/tasmax_median_regional_matrix.csv",
              "climate/tasmin_belowP01_regional_matrix.csv", "climate/tasmax_aboveP99_regional_matrix.csv"] #
feature_name_ls = ["ndvi", "si-ndvi", "evi", "si-evi",
                   "preci-sum", "si-preci-sum", "pr-max", "pr-cdd", "pr-belowP01", "pr-aboveP99",
                   "temp-median", "si-temp-median", "min-temp-median", "max-temp-median",
                   "min-temp-belowP01", "max-temp-aboveP99"]  # ,
# length of feature timeseries per season
length = 10 # "mmm"

processed_feature_df_ls = []
for feature, feature_name in zip(feature_ls, feature_name_ls):
    feature_df = pd.read_csv(PROCESSED_DATA_DIR / feature, keep_default_na=False)

    # apply CC for each yield datapoint
    processed_feature_df = process_feature_df(yield_df=yield_df,
                                              cc_df=cc_df,
                                              feature_df=feature_df,
                                              feature_name=feature_name,
                                              length=length,
                                              start_before_sos=30,
                                              end_before_eos=60)
    processed_feature_df_ls.append(processed_feature_df)

# define the predictive model ones (make a grid search initially to get a feeling for the params)
randy = RandomForestRegressor(max_depth=5, max_features='log2', n_estimators=300, n_jobs=-1, random_state=42) # opt on ndvi-2 Dedza
randy = RandomForestRegressor(max_depth=5, max_features=None, n_estimators=300, n_jobs=-1, random_state=42) # opt on ndvi-4/8 Dedza
randy = RandomForestRegressor(max_features=None, n_estimators=300, n_jobs=-1, random_state=42) # opt on ndvi-mmm Dedza
#randy = loyocv_grid_search(X, y, years=X[:, 0], model=randy, param_grid=rf_param_grid)

# columns to be filled with predictions
for feature_name in feature_name_ls:
    yield_df[feature_name + "_pred"] = np.nan

# loop over the admins
for adm, adm_yield_df in yield_df.groupby(["country", "adm1", "adm2"]):  #
    adm = str(adm).replace(", ", "-").replace("'", "").replace(" None", "")

    print("\n")

    for processed_feature_df, feature_name in zip(processed_feature_df_ls, feature_name_ls):
        adm_feature_df = processed_feature_df.loc[adm_yield_df.index]

        X = pd.concat([adm_yield_df["harv_year"] - 2010, adm_feature_df], axis=1).values
        y = adm_yield_df["yield"]

        start = time.time()
        y_preds, model_mse = loyocv(X, y, years=adm_yield_df["harv_year"], model=randy, model_name=f"{adm}, {feature_name}", print_result=True)
        print(time.time() - start)
        #assert False
        # fill into df
        if np.any(y_preds < 0):
            y_preds[y_preds < 0] = 0
        yield_df.loc[adm_yield_df.index, feature_name + "_pred"] = y_preds

# save
#yield_df.to_csv(RESULTS_DATA_DIR / f"feature_selection/yield_feature_prediction_{length}.csv", index=False)


# VISUALIZE ########
#yield_df = pd.read_csv(RESULTS_DATA_DIR / f"feature_selection/yield_feature_prediction_mmm.csv", keep_default_na=True)

# calculate and plot mse for each feature and country
mse_matrix = []
mse_ls = []
for feature in feature_name_ls:
    y_pred = yield_df[feature + "_pred"]
    bench_mse = np.mean((yield_df["yield"] - yield_df["bench_lin_reg"]) ** 2)
    mse = np.mean((yield_df["yield"] - y_pred) ** 2)
    mse_ls.append(1 - mse / bench_mse)
mse_matrix.append(mse_ls)
country_ls = []
for country, country_yield_df in yield_df.groupby("country"):
    country_ls.append(country)
    mse_ls = []
    for feature in feature_name_ls:
        y_pred = country_yield_df[feature + "_pred"]
        bench_mse = np.mean((country_yield_df["yield"] - country_yield_df["bench_lin_reg"]) ** 2)
        mse = np.mean((country_yield_df["yield"] - y_pred) ** 2)
        mse_ls.append(1 - mse / bench_mse)
    mse_matrix.append(mse_ls)
mse_df = pd.DataFrame(np.array(mse_matrix).T, index=feature_name_ls, columns=np.concatenate([["all"], country_ls]))

# Plotting the dataframe with individually normalized columns in the same plot without colorbar and flipped colormap
plt.figure(figsize=(13, 8))

heatmap = sns.heatmap(mse_df, annot=True, cmap='RdYlGn', fmt=".2f",
                      vmax=1.0, vmin=-1.0)

if length == 1:
    plt.title(f'Individual feature performance by RMSE using average over season')
elif length == "mmm":
    plt.title(f'Individual feature performance by RMSE using mean, min & max of season')
else:
    plt.title(f'Individual feature performance by RMSE with feature length of {length}')
plt.savefig(RESULTS_DATA_DIR / f"feature_selection/plots/indi_feature_{length}_mse.png", dpi=300)
plt.show()
