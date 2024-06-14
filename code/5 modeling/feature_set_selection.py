import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from config import PROCESSED_DATA_DIR, RESULTS_DATA_DIR
from crop_calendar.crop_calendar_functions import load_my_cc
from cv_grid_search import cv_grid_search, loyocv_grid_search
from cv_parameter import rf_param_grid
from data_assembly import process_feature_df, make_dummies
from ml_functions import loyocv
from yield_.yield_functions import load_yield_data

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
processed_feature_df_ls = []
for feature, feature_name in zip(feature_ls, feature_name_ls):
    feature_df = pd.read_csv(PROCESSED_DATA_DIR / feature, keep_default_na=False)

    # apply CC for each yield datapoint
    processed_feature_df = process_feature_df(yield_df=yield_df,
                                              cc_df=cc_df,
                                              feature_df=feature_df,
                                              feature_name=feature_name,
                                              length=1,
                                              start_before_sos=0,
                                              end_before_eos=60)
    processed_feature_df_ls.append(processed_feature_df)

# define the predictive model ones (make a grid search initially to get a feeling for the params)
#randy = loyocv_grid_search(X, y, years=X[:, 0], model=randy, param_grid=rf_param_grid, cv=4)
randy = RandomForestRegressor(max_features=1, min_samples_split=3, n_estimators=300, n_jobs=-1, random_state=42)

# columns to be filled with predictions
for model_name in feature_name_ls:
    yield_df[model_name + "_pred"] = np.nan

# loop over the countries
for adm, adm_yield_df in yield_df.groupby(["country", "adm1", "adm2"]):  #
    adm = str(adm).replace(", ", "-").replace("'", "").replace(" None", "")

    print("\n")

    for processed_feature_df, feature_name in zip(processed_feature_df_ls, feature_name_ls):
        #break
        adm_feature_df = processed_feature_df.loc[adm_yield_df.index]

        X = pd.concat([adm_yield_df["harv_year"], adm_yield_df["bench_lin_reg"], adm_feature_df], axis=1).values
        y = adm_yield_df["yield"]

        y_preds, model_mse = loyocv(X, y, years=adm_yield_df["harv_year"], model=randy, model_name=f"{adm}, {feature_name}", print_result=True)
        # fill into df
        if np.any(y_preds < 0):
            y_preds[y_preds < 0] = 0
        yield_df.loc[adm_yield_df.index, model_name + "_pred"] = y_preds

# save
yield_df.to_csv(RESULTS_DATA_DIR / f"feature_selection/yield_feature_prediction.csv", index=False)
