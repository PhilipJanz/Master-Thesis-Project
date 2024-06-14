import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from config import PROCESSED_DATA_DIR
from crop_calendar.crop_calendar_functions import load_my_cc
from cv_grid_search import cv_grid_search, loyocv_grid_search
from cv_parameter import rf_param_grid
from data_assembly import process_feature_df, make_dummies
from ml_functions import loyocv
from yield_.yield_functions import load_yield_data

# load yield data
yield_df = load_yield_data()
yield_df = yield_df[~yield_df.country.isin(["Ethiopia", "Kenya"])]
yield_df = yield_df[yield_df.harv_year > 2001]
yield_df.sort_values(["country", "adm1", "adm2", "harv_year"], inplace=True, ignore_index=True)

# load crop calendar (CC)
cc_df = load_my_cc()

# load features and make feature dataframe for yield data with respect to the CC
feature_ls = ["remote sensing/smooth_ndvi_regional_matrix.csv", "remote sensing/si_smooth_ndvi_regional_matrix.csv",
              "remote sensing/smooth_evi_regional_matrix.csv", "remote sensing/si_smooth_evi_regional_matrix.csv"] # ,"climate/pr_sum_regional_matrix.csv", "climate/si_pr_sum_regional_matrix.csv"
feature_name_ls = ["ndvi", "si-ndvi", "evi", "si-evi"]  # , "preci-sum", "si-preci-sum"
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


# loop over the countries
for country, country_yield_df in yield_df.groupby("country"): # ["adm1", "adm2"]
    print("\n")
    # make dummies for admins
    dummy_df = make_dummies(country_yield_df)

    # first run without any features, just year and admin
    X = pd.concat([country_yield_df["harv_year"], dummy_df], axis=1).values
    y = country_yield_df["yield"]

    y_preds, model_mse = loyocv(X, y, years=country_yield_df["harv_year"], model=randy_model_0, model_name=f"{country}, model-0", print_result=True)

    for processed_feature_df, feature_name in zip(processed_feature_df_ls, feature_name_ls):
        #break
        country_feature_df = processed_feature_df.loc[country_yield_df.index]
        country_feature_df["model-0_pred"] = y_preds
        X = pd.concat([country_yield_df["harv_year"], dummy_df, country_feature_df], axis=1).values #
        y = country_yield_df["yield"]

        y_preds, model_mse = loyocv(X, y, years=country_yield_df["harv_year"], model=randy, model_name=f"{country}, {feature_name}", print_result=True)


randy = cv_grid_search(X, y, model=randy, param_grid=rf_param_grid, cv=4)
randy = loyocv_grid_search(X, y, years=X[:, 0], model=randy, param_grid=rf_param_grid)
