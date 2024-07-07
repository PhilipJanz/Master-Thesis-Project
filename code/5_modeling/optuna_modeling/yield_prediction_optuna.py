import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from data_loader import load_yield_data, load_my_cc, load_cluster_data, load_soil_data
from optuna_modeling.feature_sets_for_optuna import feature_location_dict
from data_assembly import process_list_of_feature_df, make_adm_column, make_X, make_dummies
from optuna_modeling.run import list_of_runs, Run, open_run
from optuna_modeling.optuna_optimizer import OptunaOptimizer
import optuna

# silence the message after each trial
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
cluster_set = "country"
assert cluster_set in yield_df.columns, f"The chosen cluster-set '{cluster_set}' is not occuring in the yield_df."

# choose model or set of models that are used
model_types = ["lasso"]
# choose duration (sec) of optimization using optuna
opti_duration = 1200
# choose number of optuna startup trails (random parameter search before sampler gets activated)
n_startup_trials = 200

# let's specify tun run (see run.py) using prefix (recommended: MMDD_) and parameters from above
run_name = f"0707_{cluster_set}_{'-'.join(model_types)}_{opti_duration}_{n_startup_trials}"

# load or create that run
if run_name in list_of_runs():
    run = open_run(run_name=run_name)

    yield_df = run.load_prediction()
else:
    run = Run(name=run_name,
              cluster_set=cluster_set,
              model_types=model_types,
              opti_duration=opti_duration,
              n_startup_trials=n_startup_trials)

    # columns to be filled with predictions
    yield_df["train_mse"] = np.nan
    yield_df["y_pred"] = np.nan
    yield_df["best_model"] = np.nan
    yield_df["n_opt_trials"] = np.nan

# INFERENCE ############################################################################################################

for cluster_name, cluster_yield_df in yield_df.groupby(cluster_set):
    # in case you loaded an existing run, you can skip the clusters already predicted
    if np.all(~cluster_yield_df["y_pred"].isna()):
        continue

    # define target and year
    y = cluster_yield_df["yield_anomaly"]
    years = cluster_yield_df.harv_year

    # prepare predictors
    predictors_list = [cluster_yield_df.harv_year] + [df.loc[cluster_yield_df.index] for df in
                                                      processed_feature_df_dict.values()]

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
        X_train, y_train, years_train = X[~year_out_bool], y[~year_out_bool], years[~year_out_bool]
        X_test, y_test = X[year_out_bool], y[year_out_bool]

        # make feature-, model- and hyperparameter-selection using optuna
        sampler = optuna.samplers.TPESampler(n_startup_trials=run.n_startup_trials,
                                             multivariate=True, warn_independent_sampling=False)
        opti = OptunaOptimizer(X=X_train, y=y_train, years=years_train, predictor_names=predictor_names,
                               sampler=sampler,
                               model_types=run.model_types,
                               feature_set_selection=True, feature_len_shrinking=True, num_folds=5, seed=42)

        mse, best_params = opti.optimize(n_trials=5000, timeout=run.opti_duration,
                                         show_progress_bar=False, print_result=False)

        # check if features are left
        X_trans, _ = opti.transform_X(X=X, predictor_names=predictor_names, params=opti.best_params)
        if X_trans.shape[1] == 0:
            # in this case just output 0 as the best estimator for yield anomaly in that scenario
            yield_df.loc[cluster_yield_df.index[year_out_bool], "y_pred"] = 0
            yield_df.loc[cluster_yield_df.index[year_out_bool], "best_model"] = "zero-predictor"
            yield_df.loc[cluster_yield_df.index[year_out_bool], "train_mse"] = np.var(y_train)
            yield_df.loc[cluster_yield_df.index[year_out_bool], "n_opt_trials"] = opti.study.best_trial.number
            continue

        # train best model
        X_train_trans, new_predictor_names, trained_model = opti.train_best_model(X=X_train, y=y_train,
                                                                                  predictor_names=predictor_names)

        # prepare test-data
        X_test_trans, _ = opti.transform_X(X=X_test, predictor_names=predictor_names, params=opti.best_params)

        # predict train- & test-data
        y_pred_train = trained_model.predict(X_train_trans)
        y_pred = trained_model.predict(X_test_trans)

        # write the predictions into the result df
        yield_df.loc[cluster_yield_df.index[year_out_bool], "train_mse"] = np.mean((y_pred_train - y_train) ** 2)
        yield_df.loc[cluster_yield_df.index[year_out_bool], "y_pred"] = y_pred
        yield_df.loc[cluster_yield_df.index[year_out_bool], "best_model"] = opti.best_params['model_type']
        yield_df.loc[cluster_yield_df.index[year_out_bool], "n_opt_trials"] = opti.study.best_trial.number

        # save trained model and best params
        trained_model.predictor_names = new_predictor_names
        run.save_model_and_params(name=f"{cluster_name}_{year_out}", model=trained_model, params=opti.best_params)

    # break
    # np.nanmean((yield_df.loc[cluster_yield_df.index]["y_pred"] - yield_df.loc[cluster_yield_df.index]["yield_anomaly"]) ** 2)
    # 1 - np.nanmean((yield_df.loc[cluster_yield_df.index]["y_pred"] - yield_df.loc[cluster_yield_df.index]["yield_anomaly"]) ** 2) / np.var(y)
    # plt.scatter(yield_df.loc[cluster_yield_df.index]["yield_anomaly"], yield_df.loc[cluster_yield_df.index]["y_pred"])
    # plt.plot([-0.5, 0.5], [-0.5, 0.5], color="red")
    # plt.xlabel("True yield anomalies")
    # plt.ylabel("Predicted yield anomalies")
    # plt.show()

    # save predictions and performance
    run.save_predictions(prediction_df=yield_df)
    run.save_performance(prediction_df=yield_df[~yield_df["y_pred"].isna()], cluster_set=cluster_set)

    # save run
    run.save()

# VISUALIZATION ########################################################################################################
