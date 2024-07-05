import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

#from cluster_functions import load_cluster_data
from config import PROCESSED_DATA_DIR, RESULTS_DATA_DIR
from feature_sets_for_optuna import feature_location_dict
from models import model_ls, model_param_grid_ls
from data_assembly import process_feature_df, process_list_of_feature_df, make_adm_column, make_X, make_dummies
from loyocv import loyocv, loyocv_parallel, loyocv_grid_search, nested_loyocv
#from soil.soil_functions import load_soil_data

from optuna_hypersearch import OptunaOptimizer
import optuna

"""
This script is the final script that brings together all processed data to make groundbreaking yield predictions.
It searches for the best combination of: predictive model (architecture and hyperparameters), sets of data cluster and feature set

To make the code more slim we dont loop over the sets of clusters (all, country, adm, ndvi-cluster, ...)
This has to be specified in the beginning of the code.

There are two main loops:
# TODO!
    2. loop over clusters (eg. single countries)
        inside the loop a nested-LOYOCV strategy is applied
        the inner LOYOCV estimates the parameters performance and picks the best hyperparameters and feature set
        the outer LOYOCV evaluated the best models performance on entirely unseen data
        the picked hyperparameters and feature set are saved  for later investigation
"""

# LOAD & PREPROCESS ############

# load yield data and benchmark
#yield_df = pd.read_csv(RESULTS_DATA_DIR / f"benchmark/yield_benchmark_prediction.csv", keep_default_na=False)
yield_df = pd.read_csv(PROCESSED_DATA_DIR / "yield/processed_comb_yield.csv", keep_default_na=False)
yield_df = make_adm_column(yield_df)

# load crop calendar (CC)
cc_df = pd.read_csv(PROCESSED_DATA_DIR / f"crop calendar/my_crop_calendar.csv", keep_default_na=False) # load_my_cc()

# load and process features
length = 10
processed_feature_df_dict = process_list_of_feature_df(yield_df=yield_df, cc_df=cc_df, feature_dict=feature_location_dict,
                                                       length=length, start_before_sos=30, end_before_eos=60)

# load soil characteristics
soil_df = pd.read_csv(PROCESSED_DATA_DIR / "soil/soil_property_region_level.csv", keep_default_na=False) # load_soil_data()
soil_df = make_adm_column(soil_df)
soil_df = pd.merge(yield_df["adm"], soil_df, on=["adm"], how="left")
soil_df = soil_df[['clay', 'elevation', 'nitrogen', 'phh2o', 'sand', 'silt', 'soc']]
# scale each column (soil property) individually
soil_df.iloc[:, :] = StandardScaler().fit_transform(soil_df.values)

# load clusters
cluster_df = pd.read_csv(PROCESSED_DATA_DIR / f"data clustering/cluster_data.csv", keep_default_na=False) # load_cluster_data()
yield_df = pd.merge(yield_df, cluster_df, on=["country", "adm1", "adm2"])


# INITIALIZATION ##############


# 1. loop over cluster sets
for cluster_set in ["adm"]:

    # collect dictionary of best models: {"cluster_hold-out-year": {...}, ...}
    best_model_dict = {}
    # as well as a dictionary of best model parameters: {"cluster_hold-out-year": {...}, ...}
    best_model_param_dict = {}

    # columns to be filled with predictions
    yield_df["y_pred"] = np.nan
    yield_df["best_model"] = np.nan

    # 2. loop over cluster
    for cluster_name, cluster_yield_df in yield_df.groupby(cluster_set):
        # define target and year
        y = cluster_yield_df["yield_anomaly"]
        years = cluster_yield_df.harv_year

        # prepare predictors # [cluster_yield_df.harv_year] +
        predictors_list = [cluster_yield_df.harv_year] + [df.loc[cluster_yield_df.index] for df in processed_feature_df_dict.values()]

        # add dummies for regions
        if cluster_yield_df.adm.nunique() > 1:
            predictors_list.append(make_dummies(cluster_yield_df))
            predictors_list.append(soil_df.loc[cluster_yield_df.index])

        # form the regressor-matrix X
        X, predictor_names = make_X(df_ls=predictors_list, standardize=True)

        # LOYOCV - leave one year out cross validation
        for year_out in np.unique(years):
            year_out_bool = (years == year_out)

            # splitt the data
            X_train, y_train = X[~year_out_bool], y[~year_out_bool]
            X_test, y_test = X[year_out_bool], y[year_out_bool]

            # make feature-, model- and hyperparameter-selection using optuna
            sampler = optuna.samplers.TPESampler(n_startup_trials=200, multivariate=True, warn_independent_sampling=False)
            opti = OptunaOptimizer(X=X, y=y, years=years, predictor_names=predictor_names, sampler=sampler,
                                   model_types=["lasso"],
                                   feature_set_selection=True, feature_len_shrinking=True, num_folds=20, seed=42)

            mse, best_params = opti.optimize(n_trials=5000, timeout=6, show_progress_bar=False, print_result=False)

            # train best model
            X_train_trans, trained_model = opti.train_best_model(X=X_train, y=y_train, predictor_names=predictor_names)

            # prepare test-data
            X_test_trans = opti.transform_X(X=X_test, predictor_names=predictor_names)

            # predict test-data
            y_pred = trained_model.predict(X_test_trans)

            # write the predictions into the result df
            yield_df.loc[cluster_yield_df.index[year_out_bool], "y_pred"] = y_pred

            # save trained model and best params
            best_model_dict[f"{cluster_name}_{year_out}"] = trained_model
            best_model_param_dict[f"{cluster_name}_{year_out}"] = best_params

        best_feature_set_mtx.append(best_feature_set_ls)

    # save results!
    yield_df.to_csv(RESULTS_DATA_DIR / f"yield_predictions/{cluster_set}_classic_modeling.csv", index=False)

    # save feature set selection
    feature_set_selection_df = pd.DataFrame(np.array(best_feature_set_mtx), columns=yield_df[cluster_set].unique(), index=list(model_ls))
    feature_set_selection_df.to_csv(RESULTS_DATA_DIR / f"feature_selection/{cluster_set}_selected_feature_sets_{length}.csv")

    # save the models
    with open(RESULTS_DATA_DIR / f"yield_predictions/models/{cluster_set}_classic_models.pickle", 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(best_model_dict_of_dicts, f, pickle.HIGHEST_PROTOCOL)

    #np.mean((yield_df["yield"] - yield_df["rf_pred"]) ** 2)