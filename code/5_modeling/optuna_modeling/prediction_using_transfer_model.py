import time

from metrics import calc_r2

wait_minutes = 240
for i in range(wait_minutes):
    print(f"{wait_minutes - i} minutes until the script starts.........", end="\r")
    #time.sleep(60)

import os

from config import SEED, BASE_DIR

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd

from data_loader import load_cluster_data, load_soil_pca_data, load_yield_data
from data_assembly import make_X, make_dummies
from run import list_of_runs, Run, open_run
from optuna_modeling.optuna_optimizer import OptunaOptimizer
from optuna.pruners import ThresholdPruner
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


# INITIALIZATION #######################################################################################################

# date of first execution of that run
date = "2810"

# define objective (target)
objective = "yield"

# choose data split for single models by choosing 'country', 'adm' or a cluster from cluster_df
data_split = "adm"

# source model
source_run_name = "2510_yield_all_cnn_32_60_60_optuna_7200_100_50_5"
source_name = "all"

# predictive model
model_type = "lasso"

# optuna hyperparameter optimization params
# choose timeout
timeout = 600
# choose duration (sec) of optimization using optuna
n_trials = 100
# choose number of optuna startup trails (random parameter search before sampler gets activated)
n_startup_trials = 50
# choose a upper limit on loss (mse) for pruning an optuna trial (early stopping due to weak performance)
pruner_upper_limit = 1
# folds of optuna hyperparameter search
num_folds = 5


# LOAD & PREPROCESS ####################################################################################################

# load yield data and benchmark
yield_df = load_yield_data(benchmark_column=True)

# load soil characteristics
soil_df = load_soil_pca_data(pc_number=2)

# load source run
source_run = open_run(source_run_name)

# let's specify tun run (see run.py) using prefix (recommended: MMDD_) and parameters from above
run_name = (f"{date}_{objective}_{data_split}_transfer_features_from_{source_name}_{model_type}_{timeout}_{n_trials}_{n_startup_trials}_{num_folds}")

# load or create that run
if run_name in list_of_runs():
    run = open_run(run_name=run_name)

    yield_df = run.load_prediction()
    yield_df = yield_df.replace(r'^\s*$', np.nan, regex=True)
    yield_df["train_mse"] = pd.to_numeric(yield_df["train_mse"])
    yield_df["y_pred"] = pd.to_numeric(yield_df["y_pred"])
    yield_df["n_opt_trials"] = pd.to_numeric(yield_df["n_opt_trials"])
else:
    run = Run(name=run_name,
              cluster_set=data_split,
              model_type=model_type,
              objective=objective,
              timeout=timeout,
              n_trials=n_trials,
              n_startup_trials=n_startup_trials,
              python_file=BASE_DIR / "code/5_modeling/optuna_modeling/prediction_using_transfer_model.py") #os.path.abspath(__file__))

    # columns to be filled with predictions
    yield_df["train_mse"] = np.nan
    yield_df["y_pred"] = np.nan
    yield_df["best_model"] = np.nan
    yield_df["n_opt_trials"] = np.nan
    run.save_predictions(prediction_df=yield_df)

# INFERENCE ############################################################################################################

for split_name, split_yield_df in yield_df.groupby(data_split):
    #break
    #if "Malawi" in cluster_yield_df["country"].values:
    #    continue

    # in case you loaded an existing run, you can skip the clusters already predicted
    if np.all(~split_yield_df["y_pred"].isna()):
        continue

    # define target and year
    y = split_yield_df[objective]
    years = split_yield_df.harv_year

    # prepare predictors # [cluster_yield_df.harv_year] +
    predictors_list = [split_yield_df.harv_year]

    # add dummies for regions
    if split_yield_df.adm.nunique() > 1:
        predictors_list.append(soil_df.loc[split_yield_df.index])
        predictors_list.append(make_dummies(split_yield_df))
    # form the regressor-matrix X
    X, feature_names = make_X(df_ls=predictors_list, standardize=True)

    # LOYOCV - leave one year out cross validation
    for year_out in np.unique(years):
        year_out_bool = (years == year_out)

        # in case you loaded an existing run, you can skip the year already predicted
        if np.all(~split_yield_df[year_out_bool]["y_pred"].isna()):
            continue

        # load transfer features
        if f"{source_name}_{year_out}.csv" not in os.listdir(source_run.trans_dir):
            # in this case just output 0 as the best estimator for yield anomaly in that scenario
            yield_df.loc[split_yield_df.index[year_out_bool], "y_pred"] = 0.0
            yield_df.loc[split_yield_df.index[year_out_bool], "best_model"] = "no-source-model"
            continue
        transfer_feature_df = pd.read_csv(source_run.trans_dir / f"{source_name}_{year_out}.csv")
        feature_loc = ["transfer" in col for col in transfer_feature_df.columns]
        transfer_feature_names = transfer_feature_df.columns[feature_loc]
        transfer_feature_mtx = transfer_feature_df.loc[split_yield_df.index, transfer_feature_names].values

        X_ = np.concatenate([X, transfer_feature_mtx], 1)
        feature_names_ = np.concatenate([feature_names, transfer_feature_names])
        indicator_loc = ["transfer" in col for col in feature_names_]

        X_train, y_train, years_train = X_[~year_out_bool], y[~year_out_bool], years[~year_out_bool]
        X_test, y_test, years_test = X_[year_out_bool], y[year_out_bool], years[year_out_bool]

        # make feature-, model- and hyperparameter-selection using optuna
        sampler = optuna.samplers.TPESampler(n_startup_trials=run.n_startup_trials, multivariate=True,
                                             warn_independent_sampling=False, seed=SEED)
        pruner = ThresholdPruner(upper=pruner_upper_limit)
        opti = OptunaOptimizer(study_name=f"{split_name}_{year_out}",
                               X=X_train, y=y_train, years=years_train,
                               sampler=sampler,
                               pruner=pruner,
                               model_type=run.model_type,
                               num_folds=num_folds)

        mse, best_params = opti.optimize(n_trials=run.n_trials, timeout=run.timeout,
                                         show_progress_bar=True, print_result=False)
        run.save_optuna_study(study=opti.study)

        # if best_params is None it means that optuna pruned every single trail, which means the model performs exceptionally bad
        if best_params:
            # train best model
            trained_model = opti.train_best_model()

            # predict train- & test-data
            if model_type == "gp":
                y_pred_train = trained_model.predict(X_train, years_train)
                y_pred_test = trained_model.predict(X_test, years_test)
            else:
                y_pred_train = trained_model.predict(X_train)
                y_pred_test = trained_model.predict(X_test)
        else:
            y_pred_train = split_yield_df["y_bench"][~year_out_bool].values
            y_pred_test = split_yield_df["y_bench"][year_out_bool].values

        # write the predictions into the result df
        train_mse = np.mean((y_pred_train - y_train) ** 2)
        train_nse = calc_r2(y_true=y_train, y_pred=y_pred_train)
        test_mse = np.mean((y_pred_test - y_test) ** 2)
        yield_df.loc[split_yield_df.index[year_out_bool], "train_mse"] = train_mse
        yield_df.loc[split_yield_df.index[year_out_bool], "train_nse"] = train_nse
        yield_df.loc[split_yield_df.index[year_out_bool], "test_mse"] = test_mse
        if len(y_test) > 3:
            test_nse = calc_r2(y_true=y_test, y_pred=y_pred_test)
        else:
            test_nse = None
        yield_df.loc[split_yield_df.index[year_out_bool], "test_nse"] = test_nse
        yield_df.loc[split_yield_df.index[year_out_bool], "y_pred"] = y_pred_test
        yield_df.loc[split_yield_df.index[year_out_bool], "n_opt_trials"] = len(opti.study.trials)
        if test_nse:
            print(f"{split_name} - {year_out} finished with train-mse: {round(train_mse, 3)} ({round(train_nse, 2)}) | test-mse: {round(test_mse, 3)} ({round(test_nse, 2)})")
        else:
            print(f"{split_name} - {year_out} finished with train-mse: {round(train_mse, 3)} ({round(train_nse, 2)}) | test-mse: {round(test_mse, 3)}")

        if best_params:
            # save trained model and best params
            trained_model.feature_names = feature_names
            opti.best_params["feature_names"] = feature_names
            run.save_model_and_params(name=f"{split_name}_{year_out}", model=trained_model, params=opti.best_params)

        # save predictions
        run.save_predictions(prediction_df=yield_df)

    # break
    # np.nanmean((yield_df.loc[cluster_yield_df.index]["y_pred"] - yield_df.loc[cluster_yield_df.index]["yield_anomaly"]) ** 2)
    #
    # plt.scatter(yield_df.loc[cluster_yield_df.index]["yield_anomaly"], yield_df.loc[cluster_yield_df.index]["y_pred"])
    # plt.plot([-0.5, 0.5], [-0.5, 0.5], color="red")
    # plt.xlabel("True yield anomalies")
    # plt.ylabel("Predicted yield anomalies")
    # plt.show()

    # save predictions and performance
    run.save_predictions(prediction_df=yield_df)
    run.save_performance(prediction_df=yield_df[~(yield_df["y_pred"].isna())], cluster_set=data_split)

    # save run
    run.save()

# VISUALIZATION ########################################################################################################
