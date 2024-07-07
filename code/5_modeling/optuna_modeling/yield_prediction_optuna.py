import os

from optuna_modeling.run import list_of_runs

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

from config import PROCESSED_DATA_DIR, RESULTS_DATA_DIR
from data_loader import load_yield_data, load_my_cc, load_cluster_data, load_soil_data
from optuna_modeling.feature_sets_for_optuna import feature_location_dict
from data_assembly import process_list_of_feature_df, make_adm_column, make_X, make_dummies

from optuna_modeling.optuna_optimizer import OptunaOptimizer
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

# LOAD & PREPROCESS ####################################################################################################

# load yield data and benchmark
yield_df = load_yield_data()
yield_df = make_adm_column(yield_df)

# load crop calendar (CC)
cc_df = load_my_cc()

# load and process features
length = 10
processed_feature_df_dict = process_list_of_feature_df(yield_df=yield_df, cc_df=cc_df,
                                                       feature_dict=feature_location_dict,
                                                       length=length,
                                                       start_before_sos=30, end_before_eos=60)

# load soil characteristics
soil_df = load_soil_data()
soil_df = make_adm_column(soil_df)
soil_df = pd.merge(yield_df["adm"], soil_df, on=["adm"], how="left")
soil_df = soil_df[['clay', 'elevation', 'nitrogen', 'phh2o', 'sand', 'silt', 'soc']]
# scale each column (soil property) individually
soil_df.iloc[:, :] = StandardScaler().fit_transform(soil_df.values)

# load clusters
cluster_df = load_cluster_data()
yield_df = pd.merge(yield_df, cluster_df, on=["country", "adm1", "adm2"])


# INITIALIZATION #######################################################################################################

# choose data split for single models by choosing 'country', 'adm' or a cluster from cluster_df
cluster_set = "adm"
assert cluster_set in yield_df.columns, f"The chosen cluster-set '{cluster_set}' is not occuring in the yield_df."

# choose model or set of models that are used
model_types = ["lasso"]
# choose duration (sec) of optimization using optuna
opti_duration = 120
# choose number of optuna startup trails (random parameter search before sampler gets activated)
n_startup_trials = 500

# let's specify tun run (see run.py) using prefix (recommended: MMDD_) and parameters from above
run_name = f"0707_{cluster_set}_{'-'.join(model_types)}_{opti_duration}_{n_startup_trials}"
# load or create that run
if run_name in list_of_runs():
    pass
else:
    run = Run(name=run_name,
              cluster_set=cluster_set,
              model_types=model_types,
              opti_duration=opti_duration,
              n_startup_trials=n_startup_trials)


# collect dictionary of best models: {"cluster_hold-out-year": {...}, ...}
best_model_dict = {}
# as well as a dictionary of best model parameters: {"cluster_hold-out-year": {...}, ...}
best_model_param_dict = {}

# columns to be filled with predictions
yield_df["y_pred"] = np.nan
yield_df["best_model"] = np.nan


for cluster_name, cluster_yield_df in yield_df.groupby(cluster_set):
    # define target and year
    y = cluster_yield_df["yield_anomaly"]
    years = cluster_yield_df.harv_year

    # prepare predictors
    predictors_list = [cluster_yield_df.harv_year] + [df.loc[cluster_yield_df.index] for df in processed_feature_df_dict.values()]

    # add dummies for regions
    if cluster_yield_df.adm.nunique() > 1:
        predictors_list.append(soil_df.loc[cluster_yield_df.index])
        predictors_list.append(make_dummies(cluster_yield_df))
    # form the regressor-matrix X
    X, predictor_names = make_X(df_ls=predictors_list, standardize=True)

    # LOYOCV - leave one year out cross validation
    for year_out in np.unique(years):
        year_out_bool = (years == year_out)

        # splitt the data
        X_train, y_train = X[~year_out_bool], y[~year_out_bool]
        X_test, y_test = X[year_out_bool], y[year_out_bool]

        # make feature-, model- and hyperparameter-selection using optuna
        sampler = optuna.samplers.TPESampler(n_startup_trials=400, multivariate=True, warn_independent_sampling=False)
        opti = OptunaOptimizer(X=X, y=y, years=years, predictor_names=predictor_names, sampler=sampler,
                               model_types=["lasso"],
                               feature_set_selection=True, feature_len_shrinking=True, num_folds=20, seed=42)

        mse, best_params = opti.optimize(n_trials=5000, timeout=120, show_progress_bar=False, print_result=False)

        # check if features are left
        X_trans = self.transform_X(X=X, predictor_names=predictor_names, params=self.best_params)
        if X_trans.shape[1] == 0:
            # in this case just output 0 as the best estimator for yield anomaly in that scenario
            yield_df.loc[cluster_yield_df.index[year_out_bool], "y_pred"] = 0
            best_model_dict[f"{cluster_name}_{year_out}"] = "zero-predictor"
            best_model_param_dict[f"{cluster_name}_{year_out}"] = best_params

        # train best model
        X_train_trans, trained_model = opti.train_best_model(X=X_train, y=y_train, predictor_names=predictor_names)

        # prepare test-data
        X_test_trans = opti.transform_X(X=X_test, predictor_names=predictor_names, params=opti.best_params)

        # predict test-data
        y_pred = trained_model.predict(X_test_trans)

        # write the predictions into the result df
        yield_df.loc[cluster_yield_df.index[year_out_bool], "y_pred"] = y_pred

        # save trained model and best params
        best_model_dict[f"{cluster_name}_{year_out}"] = trained_model
        best_model_param_dict[f"{cluster_name}_{year_out}"] = best_params

    break
    np.nanmean((yield_df["y_pred"] - yield_df["yield_anomaly"]) ** 2)

# save results!
yield_df.to_csv(RESULTS_DATA_DIR / f"yield_predictions/{cluster_set}_classic_modeling.csv", index=False)

# save feature set selection
feature_set_selection_df = pd.DataFrame(np.array(best_feature_set_mtx), columns=yield_df[cluster_set].unique(), index=list(model_ls))
feature_set_selection_df.to_csv(RESULTS_DATA_DIR / f"feature_selection/{cluster_set}_selected_feature_sets_{length}.csv")

# save the models
with open(RESULTS_DATA_DIR / f"yield_predictions/models/{cluster_set}_classic_models.pickle", 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(best_model_dict_of_dicts, f, pickle.HIGHEST_PROTOCOL)

# np.nanmean((yield_df["yield"] - yield_df["rf_pred"]) ** 2)
