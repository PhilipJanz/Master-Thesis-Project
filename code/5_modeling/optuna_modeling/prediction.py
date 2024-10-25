import time

import shap
from optuna.pruners import ThresholdPruner
import xgboost as xgb
from metrics import calc_r2

wait_minutes = 180
#for i in range(wait_minutes):
    #print(f"{wait_minutes - i} minutes until the script starts.........", end="\r")
    #time.sleep(60)

import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from config import BASE_DIR
import numpy as np
import pandas as pd

from data_loader import load_yield_data, load_processed_features, \
    load_soil_pca_data, load_feature_selection
from data_assembly import make_X, make_dummies
from run import list_of_runs, Run, open_run
from optuna_modeling.optuna_optimizer import OptunaOptimizer
import optuna

# silence the message after each trial
optuna.logging.set_verbosity(optuna.logging.WARNING)

#plt.switch_backend('agg')

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
date = "2110"

# define objective (target)
objective = "yield"

# choose data split for single models by choosing 'country', 'adm' or a cluster from cluster_df
data_split = "adm"

# processed feature code (feature len _ days before sos _ days before eos)
feature_file = "3_60_60"

# feature selection file
vif_threshold = 5
fs_file = f"feature_selection_dir_{data_split}_vif{vif_threshold}_{feature_file}"

# choose model or set of models that are used
model_type = "xgb"

# number of principal components that are used to make soil-features (using PCA)
soil_pc_number = 2

# optuna hyperparameter optimization params
# choose timeout
timeout = 600
# choose duration (sec) of optimization using optuna
n_trials = 250
# choose number of optuna startup trails (random parameter search before sampler gets activated)
n_startup_trials = 100
# choose a upper limit on loss (mse) for pruning an optuna trial (early stopping due to weak performance)
pruner_upper_limit = .5
# folds of optuna hyperparameter search
num_folds = 5


# LOAD & PREPROCESS ####################################################################################################

# load yield data and benchmark
yield_df = load_yield_data(benchmark_column=True)

# load and process features
processed_feature_df = load_processed_features(feature_file)
# test if datasets have same row order
assert np.all(processed_feature_df[["adm", "harv_year"]] == yield_df[["adm", "harv_year"]])
processed_feature_df = processed_feature_df.drop(['country', 'adm1', 'adm2', 'adm', 'harv_year'], axis=1)

# load feature selection
if fs_file:
    feature_selection_dict = load_feature_selection(feature_selection_file=fs_file)

# load soil characteristics
soil_df = load_soil_pca_data(pc_number=2)

# let's specify tun run (see run.py) using prefix (recommended: MMDD_) and parameters from above
run_name = f"{date}_{objective}_{data_split}_{model_type}_{feature_file}_optuna_{timeout}_{n_trials}_{n_startup_trials}_{num_folds}"
if vif_threshold:
    run_name += f"_vif{vif_threshold}"

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
              objective=objective,
              model_type=model_type,
              timeout=timeout,
              n_trials=n_trials,
              n_startup_trials=n_startup_trials,
              python_file=BASE_DIR / "code/5_modeling/optuna_modeling/prediction.py") #os.path.abspath(__file__))

    # columns to be filled with predictions
    yield_df["train_mse"] = np.nan
    yield_df["y_pred"] = np.nan
    yield_df["n_opt_trials"] = np.nan
    run.save_predictions(prediction_df=yield_df)


# INFERENCE ############################################################################################################

for split_name, split_yield_df in yield_df.groupby(data_split):
    #break
    #if "Tanzania" in cluster_yield_df["country"].values:
    #    continue

    # in case you loaded an existing run, you can skip the clusters already predicted
    if np.all(~split_yield_df["y_pred"].isna()):
        continue

    # define target and year
    y = split_yield_df[objective]
    years = split_yield_df.harv_year

    # prepare predictors # [cluster_yield_df.harv_year] +
    predictors_list = [split_yield_df.harv_year] + [processed_feature_df.loc[split_yield_df.index]]

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

        if vif_threshold:
            selected_features = feature_selection_dict[split_name][year_out]
            drop_features = [feature for feature in processed_feature_df.columns if feature not in selected_features]
            remaining_features = [feature for feature in feature_names if feature not in drop_features]
            X_sel = X[:, [feature not in drop_features for feature in feature_names]]
        else:
            remaining_features = feature_names.tolist()
            X_sel = X.copy()

        X_train, y_train, years_train = X_sel[~year_out_bool], y[~year_out_bool], years[~year_out_bool]
        X_test, y_test = X_sel[year_out_bool], y[year_out_bool]

        # make feature-, model- and hyperparameter-selection using optuna
        sampler = optuna.samplers.TPESampler(n_startup_trials=run.n_startup_trials, multivariate=True, warn_independent_sampling=False)
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

            # estimate shap values
            if model_type == "xgb":
                explainer = shap.TreeExplainer(trained_model, X_train)
            else:
                explainer = shap.Explainer(trained_model, X_train)
            shap_data = explainer(X_test)
            shap_data.feature_names = remaining_features

            # save shapley values
            run.save_shap(name=f"{split_name}_{year_out}", explainer=explainer, shap_data=shap_data)

            # save feature importance
            if model_type == "xgb":
                feature_importance_dict = {}
                for importance, feature_name in zip(trained_model.feature_importances_, remaining_features):
                    feature_importance_dict[feature_name] = importance
                pd.DataFrame(feature_importance_dict, index=[0]).to_csv(run.run_dir / f"feature_importance/{split_name}_{year_out}.csv")

            # predict train- & test-data
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

        # save predictions and performance
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
