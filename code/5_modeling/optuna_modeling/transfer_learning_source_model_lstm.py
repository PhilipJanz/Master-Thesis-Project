import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from sklearn.decomposition import PCA

from config import SEED, BASE_DIR, PROCESSED_DATA_DIR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model

from data_loader import load_yield_data, load_my_cc, load_cluster_data, load_soil_data, load_processed_features, \
    load_soil_pca_data
from optuna_modeling.feature_sets_for_optuna import feature_location_dict
from data_assembly import process_list_of_feature_df, make_adm_column, make_X, make_dummies
from run import list_of_runs, Run, open_run
from optuna_modeling.optuna_optimizer import OptunaOptimizer
import optuna
import tensorflow as tf
#tf.keras.config.disable_interactive_logging()
tf.random.set_seed(SEED)

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
date = "0901"

# define objective (target)
objective = "yield"

# choose data split for single models by choosing 'country', 'adm' or a cluster from cluster_df
data_split = "all"

# length of timeseries of remotes sensing and meteorological features
ts_length = 10

# number of principal components that are used to make soil-features (using PCA)
soil_pc_number = 2

# optuna hyperparameter optimization params
# choose timeout
timeout = 60 * 60 * 1
# choose duration (sec) of optimization using optuna
n_trials = 250
# choose number of optuna startup trails (random parameter search before sampler gets activated)
n_startup_trials = 100
# folds of optuna hyperparameter search
num_folds = 5


# LOAD & PREPROCESS ####################################################################################################

# load yield data and benchmark
yield_df = load_yield_data()

# load and process features
processed_feature_df_dict = load_processed_features(ts_length)

# load soil characteristics
soil_df = load_soil_pca_data(pc_number=2)

# load clusters
cluster_df = load_cluster_data()
yield_df = pd.merge(yield_df, cluster_df, how="left")  # , on=["country", "adm1", "adm2"]

# let's specify tun run (see run.py) using prefix (recommended: MMDD_) and parameters from above
run_name = f"{date}_{objective}_{data_split}_transferable-lstm_{ts_length}_{timeout}_{n_trials}_{n_startup_trials}_{num_folds}"

# load or create that run
if run_name in list_of_runs():
    run = open_run(run_name=run_name)

    yield_df = run.load_prediction()
    yield_df = yield_df.replace(r'^\s*$', np.nan, regex=True)
    yield_df["train_mse"] = pd.to_numeric(yield_df["train_mse"])
    yield_df["y_pred"] = pd.to_numeric(yield_df["y_pred"])
    yield_df["n_opt_trials"] = pd.to_numeric(yield_df["n_opt_trials"])
    run.trans_dir = run.run_dir / "transfer_features"
else:
    run = Run(name=run_name,
              cluster_set=data_split,
              model_types="lstm",
              timeout=timeout,
              n_trials=n_trials,
              n_startup_trials=n_startup_trials,
              python_file=BASE_DIR / "code/5_modeling/optuna_modeling/transfer_learning_source_model_lstm.py") #os.path.abspath(__file__))
    run.trans_dir = run.run_dir / "transfer_features"
    os.mkdir(run.trans_dir)

    # columns to be filled with predictions
    yield_df["train_mse"] = np.nan
    yield_df["y_pred"] = np.nan
    yield_df["n_opt_trials"] = np.nan
    run.save_predictions(prediction_df=yield_df)

# define target and year
y = yield_df[objective].values.astype("float32")
years = yield_df.harv_year.values

# prepare predictors # [cluster_yield_df.harv_year] +
predictors_list = [yield_df.harv_year]
# add dummies for regions
predictors_list.append(soil_df)
predictors_list.append(make_dummies(yield_df))
# form the static regressor-matrix X_static
X_static, static_feature_names = make_X(df_ls=predictors_list, standardize=True)
# time sensitive features with shape (n x t x f) (samples x timeseries-lenght x features)
X_time = np.array([df.values.T for df in processed_feature_df_dict.values()], dtype='f').transpose()
# unite
X = (X_static.astype("float32"), X_time.astype("float32"))

# INFERENCE ############################################################################################################

#for source_name, source_yield_df in yield_df.groupby(data_split):
source_name = data_split
source_yield_df = yield_df
for _ in [0]:
    #break
    # in case you loaded an existing run, you can skip the clusters already predicted
    if np.all(~source_yield_df["y_pred"].isna()):
        continue

    # extract source data
    source_ix = source_yield_df.index
    X_static_source = X_static[source_ix]
    X_time_source = X_time[source_ix]
    y_source = y[source_ix]
    years_source = years[source_ix]

    # further filter the dummy features for non-source regions
    non_source_adm = yield_df.loc[~yield_df.index.isin(source_ix), "adm"].unique()
    non_source_ix = np.array([feature in non_source_adm for feature in static_feature_names])
    X_static_source = X_static_source[:, ~non_source_ix]
    X_static_ = X_static[:, ~non_source_ix]
    static_feature_names_ = static_feature_names[~non_source_ix]
    # unite
    X_ = (X_static_, X_time)

    # hyperparameter-selection using optuna
    sampler = optuna.samplers.TPESampler(n_startup_trials=run.n_startup_trials, seed=SEED, multivariate=True, warn_independent_sampling=False) #
    opti = OptunaOptimizer(study_name=f"{source_name}", #_{year_out}",
                           X=X_, y=y, years=years,
                           sampler=sampler,
                           model_types=run.model_types,
                           feature_set_selection=False, feature_len_shrinking=False,
                           num_folds=num_folds)

    mse, best_params = opti.optimize(n_trials=run.n_trials, timeout=run.timeout,
                                     show_progress_bar=True, print_result=False, n_jobs=-1)
    run.save_optuna_study(study=opti.study)

    # LOYOCV - leave one year out cross validation
    for year_out in np.unique(years_source):
        # year_out=2016
        year_out_bool = (years_source == year_out)

        # in case you loaded an existing run, you can skip the year already predicted
        if np.all(~source_yield_df[year_out_bool]["y_pred"].isna()):
            continue

        # split the data
        X_train, y_train, years_train = (X_static_source[~year_out_bool], X_time_source[~year_out_bool]), y_source[~year_out_bool], years_source[~year_out_bool]
        X_test, y_test = (X_static_source[year_out_bool], X_time_source[year_out_bool]), y_source[year_out_bool]
        if objective == "yield_anomaly":
            print(source_name, year_out, f"\nmse(train)={round(np.mean(y_train ** 2), 3)}")
        else:
            print(source_name, year_out, f"\nmse(train)={round(np.mean((y_train - np.mean(y_train)) ** 2), 3)}")

        # train best model
        trained_model = opti.train_best_model(X=X_train, y=y_train)

        # predict train- & test-data
        y_pred_train = trained_model.predict(X_train).T[0]
        y_pred_test = trained_model.predict(X_test).T[0]

        # write the predictions into the result df
        train_mse = np.mean((y_pred_train - y_train) ** 2)
        if objective == "yield_anomaly":
            train_nse = 1 - train_mse / np.mean(y_train ** 2)
        else:
            train_nse = 1 - train_mse / np.mean((y_train - np.mean(y_train)) ** 2)
        test_mse = np.mean((y_pred_test - y_test) ** 2)
        yield_df.loc[source_yield_df.index[year_out_bool], "train_mse"] = train_mse
        yield_df.loc[source_yield_df.index[year_out_bool], "train_nse"] = train_nse
        yield_df.loc[source_yield_df.index[year_out_bool], "test_mse"] = test_mse
        if len(y_test) > 3:
            if objective == "yield_anomaly":
                test_nse = 1 - test_mse / np.mean(y_test ** 2)
            else:
                test_nse = 1 - test_mse / np.mean((y_test - np.mean(y_test)) ** 2)
        else:
            test_nse = None
        yield_df.loc[source_yield_df.index[year_out_bool], "test_nse"] = test_nse
        yield_df.loc[source_yield_df.index[year_out_bool], "y_pred"] = y_pred_test
        yield_df.loc[source_yield_df.index[year_out_bool], "n_opt_trials"] = len(opti.study.trials)
        if test_nse:
            print(f"{source_name} - {year_out} finished with train-mse: {round(train_mse, 3)} ({round(train_nse, 2)}) | test-mse: {round(test_mse, 3)} ({round(test_nse, 2)})")
        else:
            print(f"{source_name} - {year_out} finished with train-mse: {round(train_mse, 3)} ({round(train_nse, 2)}) | test-mse: {round(test_mse, 3)}")

        # save trained model and best params
        trained_model.feature_names = static_feature_names_
        opti.best_params["feature_names"] = static_feature_names_
        run.save_model_and_params(name=f"{source_name}_{year_out}", model=trained_model, params=opti.best_params, model_type="lstm")

        # extract learned features for all data by taking results of last hidden layer
        transfer_feature_mtx = []
        # get last weights before network output
        last_weights = trained_model.layers[-1].get_weights()
        # set all weights to zero
        last_weights[0] *= 0
        last_weights[1] *= 0
        # for each unit set weight to 1 so the networks output is the output of that hidden unit
        for i in range(best_params["hidden_units"]):
            last_weights[0] *= 0
            last_weights[0][i] = 1
            trained_model.layers[-1].set_weights(last_weights)
            transfer_feature_mtx.append(trained_model.predict(X_, verbose=0))

        transfer_feature_df = pd.DataFrame(np.hstack(transfer_feature_mtx), columns=[f"transfer_feature_{i + 1}" for i in range(best_params["hidden_units"])])
        transfer_feature_df = pd.concat([yield_df[["country", "adm1", "adm2", "adm", "harv_year"]], transfer_feature_df], axis=1)
        transfer_feature_df.to_csv(run.trans_dir / f"{source_name}_{year_out}.csv", index=False)

        # After each model is done
        tf.keras.backend.clear_session()
        del trained_model

        # save predictions and performance
        run.save_predictions(prediction_df=yield_df)
        run.save_performance(prediction_df=yield_df[~(yield_df["y_pred"].isna())], cluster_set=data_split)

        # save run
        run.save()

    preds = yield_df.loc[source_yield_df.index]["y_pred"]
    y_ = y_source[~preds.isna()]
    preds_ = preds[~preds.isna()]

    if objective == "yield_anomaly":
        nse = 1 - np.mean((preds_ - y_) ** 2) / np.mean(y_ ** 2)
    else:
        nse = 1 - np.mean((preds_ - y_) ** 2) / np.mean((y_ - np.mean(y_)) ** 2)
    print(f"{source_name} finished with: NSE = {np.round(nse, 2)}")


# VISUALIZATION ########################################################################################################
