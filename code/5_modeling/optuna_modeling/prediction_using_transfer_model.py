import os

from config import SEED

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_loader import load_cluster_data, load_soil_data
from data_assembly import make_adm_column, make_X, make_dummies
from run import list_of_runs, Run, open_run
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

# source model
source_run_name = "0829_yield_anomaly_all_transferable-lstm_10_10800_250_100_5"
source_name = "all"

# Sleep for 5 hours (5 hours * 60 minutes/hour * 60 seconds/minute)
#time.sleep(5 * 60 * 60)

source_run = open_run(source_run_name)
yield_df = source_run.load_prediction()

# load soil characteristics
soil_df = load_soil_data()
soil_df = make_adm_column(soil_df)
pca = PCA(n_components=2)
soil_df[["soil_1", "soil_2"]] = pca.fit_transform(soil_df[['clay', 'elevation', 'nitrogen', 'phh2o', 'sand', 'silt', 'soc']])
print("explained_variance_ratio_ of PCA on soil:", pca.explained_variance_ratio_)
soil_df = pd.merge(yield_df["adm"], soil_df, on=["adm"], how="left")
soil_df = soil_df[["soil_1", "soil_2"]] # soil_df[['clay', 'elevation', 'nitrogen', 'phh2o', 'sand', 'silt', 'soc']]
# scale each column (soil property) individually
soil_df.iloc[:, :] = StandardScaler().fit_transform(soil_df.values)

# load clusters
cluster_df = load_cluster_data()
yield_df = pd.merge(yield_df, cluster_df, how="left")  # , on=["country", "adm1", "adm2"]


# INITIALIZATION #######################################################################################################

# choose data split for single models by choosing 'country', 'adm' or a cluster from cluster_df
yield_df["adm1_"] = yield_df["country"] + "_" + yield_df["adm1"]
yield_df.loc[yield_df["country"] == "Tanzania", "adm1_"] = "Tanzania"
cluster_set = "adm"
assert cluster_set in yield_df.columns, f"The chosen cluster-set '{cluster_set}' is not occuring in the yield_df."

# define objective (target)
objective = "yield_anomaly"

# choose model or set of models that are used
model_type = "lasso"
# choose timeout
timeout = 300
# choose duration (sec) of optimization using optuna
n_trials = 100
# choose number of optuna startup trails (random parameter search before sampler gets activated)
n_startup_trials = 20
# folds of optuna hyperparameter search
num_folds = 10

# let's specify tun run (see run.py) using prefix (recommended: MMDD_) and parameters from above
run_name = (f"0830_{objective}_{cluster_set}_transfer_features_from_{source_name}_{model_type}_{timeout}_{n_trials}_{n_startup_trials}_{num_folds}")

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
              cluster_set=cluster_set,
              model_types=model_type,
              timeout=timeout,
              n_trials=n_trials,
              n_startup_trials=n_startup_trials,
              python_file=os.path.abspath(__file__))

    # columns to be filled with predictions
    yield_df["train_mse"] = np.nan
    yield_df["y_pred"] = np.nan
    yield_df["best_model"] = np.nan
    yield_df["n_opt_trials"] = np.nan
    run.save_predictions(prediction_df=yield_df)

# INFERENCE ############################################################################################################

for cluster_name, cluster_yield_df in yield_df.groupby(cluster_set):
    #break
    #if "Tanzania" in cluster_yield_df["country"].values:
    #    continue

    # in case you loaded an existing run, you can skip the clusters already predicted
    if np.all(~cluster_yield_df["y_pred"].isna()):
        continue

    # define target and year
    y = cluster_yield_df[objective]
    years = cluster_yield_df.harv_year

    # prepare predictors # [cluster_yield_df.harv_year] +
    predictors_list = [cluster_yield_df.harv_year]

    # add dummies for regions
    if cluster_yield_df.adm.nunique() > 1:
        predictors_list.append(soil_df.loc[cluster_yield_df.index])
        predictors_list.append(make_dummies(cluster_yield_df))
    # form the regressor-matrix X
    X, feature_names = make_X(df_ls=predictors_list, standardize=True)

    # LOYOCV - leave one year out cross validation
    for year_out in np.unique(years):
        year_out_bool = (years == year_out)

        # in case you loaded an existing run, you can skip the year already predicted
        if np.all(~cluster_yield_df[year_out_bool]["y_pred"].isna()):
            continue

        # load transfer features
        if f"{source_name}_{year_out}.csv" not in os.listdir(source_run.trans_dir):
            # in this case just output 0 as the best estimator for yield anomaly in that scenario
            yield_df.loc[cluster_yield_df.index[year_out_bool], "y_pred"] = 0.0
            yield_df.loc[cluster_yield_df.index[year_out_bool], "best_model"] = "no-source-model"
            continue
        transfer_feature_df = pd.read_csv(source_run.trans_dir / f"{source_name}_{year_out}.csv")
        feature_loc = ["transfer" in col for col in transfer_feature_df.columns]
        transfer_feature_names = transfer_feature_df.columns[feature_loc]
        transfer_feature_mtx = transfer_feature_df.loc[cluster_yield_df.index, transfer_feature_names].values

        X_ = np.concatenate([X, transfer_feature_mtx], 1)
        feature_names_ = np.concatenate([feature_names, transfer_feature_names])
        indicator_loc = ["transfer" in col for col in feature_names_]

        X_train, y_train, years_train = X_[~year_out_bool], y[~year_out_bool], years[~year_out_bool]
        X_test, y_test = X_[year_out_bool], y[year_out_bool]

        if objective != "yield_anomaly":
            print(cluster_name, year_out, "\nmse(y_train) =", np.round(np.mean((y_train - np.mean(y_train)) ** 2), 3))
        else:
            print(cluster_name, year_out, f"\nmse(y_train) = {round(np.mean(y_train ** 2), 3)}")

        # make feature-, model- and hyperparameter-selection using optuna
        sampler = optuna.samplers.TPESampler(n_startup_trials=run.n_startup_trials, multivariate=True,
                                             warn_independent_sampling=False, seed=SEED)
        opti = OptunaOptimizer(study_name=f"{cluster_name}_{year_out}",
                               X=X_train, y=y_train, years=years_train,
                               predictor_names=feature_names_,
                               sampler=sampler,
                               model_types=run.model_types,
                               feature_set_selection=False, feature_len_shrinking=False,
                               num_folds=num_folds)

        mse, best_params = opti.optimize(n_trials=run.n_trials, timeout=run.timeout,
                                         show_progress_bar=True, print_result=False)
        run.save_optuna_study(study=opti.study)

        # train best model
        trained_model = opti.train_best_model(X=X_train, y=y_train)

        if model_type == "xgb":
            if np.all(trained_model.feature_importances_[indicator_loc] < 1e-3):
                # in this case just output 0 as the best estimator for yield anomaly in that scenario
                yield_df.loc[cluster_yield_df.index[year_out_bool], "y_pred"] = 0.0
                yield_df.loc[cluster_yield_df.index[year_out_bool], "best_model"] = "zero-predictor"
                yield_df.loc[cluster_yield_df.index[year_out_bool], "train_mse"] = np.mean(y_train ** 2)
                yield_df.loc[cluster_yield_df.index[year_out_bool], "n_opt_trials"] = len(opti.study.trials)
                continue
        if model_type == "lasso":
            if np.all(trained_model.coef_[indicator_loc] == 0):
                # in this case just output 0 as the best estimator for yield anomaly in that scenario
                yield_df.loc[cluster_yield_df.index[year_out_bool], "y_pred"] = 0.0
                yield_df.loc[cluster_yield_df.index[year_out_bool], "best_model"] = "zero-predictor"
                yield_df.loc[cluster_yield_df.index[year_out_bool], "train_mse"] = np.mean(y_train ** 2)
                yield_df.loc[cluster_yield_df.index[year_out_bool], "n_opt_trials"] = len(opti.study.trials)
                continue

        # predict train- & test-data
        y_pred_train = trained_model.predict(X_train)
        y_pred_test = trained_model.predict(X_test)

        # write the predictions into the result df
        train_mse = np.mean((y_pred_train - y_train) ** 2)
        if objective == "yield_anomaly":
            train_nse = 1 - train_mse / np.mean(y_train ** 2)
        else:
            train_nse = 1 - train_mse / np.mean((y_train - np.mean(y_train)) ** 2)
        test_mse = np.mean((y_pred_test - y_test) ** 2)
        yield_df.loc[cluster_yield_df.index[year_out_bool], "train_mse"] = train_mse
        yield_df.loc[cluster_yield_df.index[year_out_bool], "train_nse"] = train_nse
        yield_df.loc[cluster_yield_df.index[year_out_bool], "test_mse"] = test_mse
        if len(y_test) > 5:
            if objective == "yield_anomaly":
                test_nse = 1 - test_mse / np.mean(y_test ** 2)
            else:
                test_nse = 1 - test_mse / np.mean((y_test - np.mean(y_test)) ** 2)
        else:
            test_nse = ""
        yield_df.loc[cluster_yield_df.index[year_out_bool], "test_nse"] = test_nse
        yield_df.loc[cluster_yield_df.index[year_out_bool], "y_pred"] = y_pred_test
        yield_df.loc[cluster_yield_df.index[year_out_bool], "n_opt_trials"] = len(opti.study.trials)
        if test_nse:
            print(f"{cluster_name} - {year_out} finished with train-mse: {round(train_mse, 3)} ({round(train_nse, 2)}) | test-mse: {round(test_mse, 3)} ({round(test_nse, 2)})")
        else:
            print(f"{cluster_name} - {year_out} finished with train-mse: {round(train_mse, 3)} ({round(train_nse, 2)}) | test-mse: {round(test_mse, 3)}")

        # save trained model and best params
        trained_model.feature_names = feature_names
        opti.best_params["feature_names"] = feature_names
        run.save_model_and_params(name=f"{cluster_name}_{year_out}", model=trained_model, params=opti.best_params, model_type=model_type)

        # save predictions
        run.save_predictions(prediction_df=yield_df)

    preds = yield_df.loc[cluster_yield_df.index]["y_pred"]
    y_ = y[~preds.isna()]
    preds_ = preds[~preds.isna()]

    if objective == "yield_anomaly":
        nse = 1 - np.mean((preds_ - y_) ** 2) / np.mean(y_ ** 2)
    else:
        nse = 1 - np.mean((preds_ - y_) ** 2) / np.mean((y_ - np.mean(y_)) ** 2)
    print(f"{cluster_name} finished with: NSE = {np.round(nse, 2)}")

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
    run.save_performance(prediction_df=yield_df[~(yield_df["y_pred"].isna())], cluster_set=cluster_set)

    # save run
    run.save()

# VISUALIZATION ########################################################################################################
