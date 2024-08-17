import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from scipy.stats import kendalltau
from sklearn.decomposition import PCA

from config import PROCESSED_DATA_DIR
from feature_selection import FeatureLearner

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model

from data_loader import load_yield_data, load_my_cc, load_cluster_data, load_soil_data
from optuna_modeling.feature_sets_for_optuna import feature_location_dict
from data_assembly import process_list_of_feature_df, make_adm_column, make_X, make_dummies
from optuna_modeling.run import list_of_runs, Run, open_run
from optuna_modeling.optuna_optimizer import OptunaOptimizer
import optuna
from tensorflow.keras import backend as K

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

# LOAD & PREPROCESS ####################################################################################################

# load yield data and benchmark
yield_df = load_yield_data()
yield_df = yield_df[(yield_df.harv_year > 2001) & (yield_df.harv_year < 2023)].reset_index(drop=True)
yield_df = make_adm_column(yield_df)

# load crop calendar (CC)
cc_df = load_my_cc()

# load and process features
length = 5
processed_feature_df_dict = process_list_of_feature_df(yield_df=yield_df, cc_df=cc_df,
                                                       feature_dict=feature_location_dict,
                                                       length=length,
                                                       start_before_sos=30, end_before_eos=60)


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
yield_df = pd.merge(yield_df, cluster_df, how="left") # , on=["country", "adm1", "adm2"]

# INITIALIZATION #######################################################################################################

# choose data split for single models by choosing 'country', 'adm' or a cluster from cluster_df
yield_df["adm1_"] = yield_df["country"] + "_" + yield_df["adm1"]
data_split = "country"
assert data_split in yield_df.columns, f"The chosen cluster-set '{data_split}' is not occuring in the yield_df."

# define objective (target)
objective = "yield_anomaly"

# choose timeout
timeout = 3600
# choose duration (sec) of optimization using optuna
n_trials = 1
# choose number of optuna startup trails (random parameter search before sampler gets activated)
n_startup_trials = 50
# folds of optuna hyperparameter search
num_folds = 2

# let's specify tun run (see run.py) using prefix (recommended: MMDD_) and parameters from above
run_name = f"0816_{objective}_{data_split}_transferable-nn_{length}_{timeout}_{n_trials}_{n_startup_trials}_{num_folds}"

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
              model_types=["nn"],
              timeout=timeout,
              n_trials=n_trials,
              n_startup_trials=n_startup_trials,
              python_file="prediction_optuna_ho_transfer_learning_fs")
    run.trans_dir = run.run_dir / "transfer_features"
    os.mkdir(run.trans_dir)

    # columns to be filled with predictions
    yield_df["train_mse"] = np.nan
    yield_df["y_pred"] = np.nan
    yield_df["n_opt_trials"] = np.nan

# define target and year
y = yield_df[objective]
years = yield_df.harv_year

# prepare predictors # [cluster_yield_df.harv_year] +
predictors_list = [yield_df.harv_year] + [df.loc[yield_df.index] for df in processed_feature_df_dict.values()]
# add dummies for regions
predictors_list.append(soil_df.loc[yield_df.index])
predictors_list.append(make_dummies(yield_df))

# form the regressor-matrix X
X, feature_names = make_X(df_ls=predictors_list, standardize=True)

# INFERENCE ############################################################################################################

for source_name, source_yield_df in yield_df.groupby(data_split):

    # in case you loaded an existing run, you can skip the clusters already predicted
    if np.all(~source_yield_df["y_pred"].isna()):
        continue

    # extract source data
    source_ix = source_yield_df.index
    X_source = X[source_ix]
    y_source = y[source_ix]
    years_source = years[source_ix]

    # further filter the dummy features for non-source regions
    non_source_adm = yield_df.loc[~yield_df.index.isin(source_ix), "adm"].unique()
    non_source_ix = np.array([feature in non_source_adm for feature in feature_names])
    X_source = X_source[:, ~non_source_ix]
    X_ = X[:, ~non_source_ix]
    feature_names_ = feature_names[~non_source_ix]

    # LOYOCV - leave one year out cross validation
    for year_out in np.unique(years_source):
        #if year_out==2016:
        #    break
        year_out_bool = (years_source == year_out)

        # split the data
        X_train, y_train, years_train = X_source[~year_out_bool], y_source[~year_out_bool], years_source[~year_out_bool]
        X_test, y_test = X_source[year_out_bool], y_source[year_out_bool]
        print(source_name, year_out, f"\nmse(train)={round(np.mean(y_train ** 2), 3)}")

        # hyperparameter-selection using optuna
        sampler = optuna.samplers.TPESampler(n_startup_trials=run.n_startup_trials, multivariate=True,
                                             warn_independent_sampling=False)
        opti = OptunaOptimizer(X=X_train, y=y_train, years=years_train, predictor_names=feature_names_,
                               sampler=sampler,
                               model_types=run.model_types,
                               feature_set_selection=False, feature_len_shrinking=False,
                               num_folds=num_folds, seed=42)

        mse, best_params = opti.optimize(n_trials=run.n_trials, timeout=run.timeout,
                                         show_progress_bar=True, print_result=False)

        # train best model
        _, _, trained_model = opti.train_best_model(X=X_train, y=y_train, predictor_names=feature_names_)

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
        if len(y_test) > 5:
            if objective == "yield_anomaly":
                test_nse = 1 - test_mse / np.mean(y_test ** 2)
            else:
                test_nse = 1 - test_mse / np.mean((y_test - np.mean(y_test)) ** 2)
        else:
            test_nse = ""
        yield_df.loc[source_yield_df.index[year_out_bool], "test_nse"] = test_nse
        yield_df.loc[source_yield_df.index[year_out_bool], "y_pred"] = y_pred_test
        yield_df.loc[source_yield_df.index[year_out_bool], "n_opt_trials"] = len(opti.study.trials)
        print(f"{source_name} - {year_out} finished with train-mse: {round(train_mse, 3)} ({round(train_nse, 2)}) | test-mse: {round(test_mse, 3)} ({round(test_nse, 2)})")

        # save trained model and best params
        trained_model.feature_names = feature_names_
        opti.best_params["feature_names"] = feature_names_
        run.save_model_and_params(name=f"{source_name}_{year_out}", model=trained_model, params=opti.best_params)

        # extract learned features for all data by taking results of last hidden layer
        transfer_model = Model(inputs=trained_model.layers[0].input, outputs=trained_model.layers[-3].output)
        # Use the encoder model to transform the data
        transfer_features = transfer_model.predict(X_)
        transfer_feature_df = pd.DataFrame(transfer_features, columns=[f"transfer_feature_{i + 1}" for i in range(transfer_features.shape[1])])
        transfer_feature_df = pd.concat([yield_df[["country", "adm1", "adm2", "adm", "harv_year"]], transfer_feature_df], axis=1)
        transfer_feature_df.to_csv(run.trans_dir / f"{source_name}_{year_out}.csv", index=False)

        # After each model is done
        K.clear_session()

    preds = yield_df.loc[source_yield_df.index]["y_pred"]
    y_ = y_source[~preds.isna()]
    preds_ = preds[~preds.isna()]

    if objective == "yield_anomaly":
        nse = 1 - np.mean((preds_ - y_) ** 2) / np.mean(y_ ** 2)
    else:
        nse = 1 - np.mean((preds_ - y_) ** 2) / np.mean((y_ - np.mean(y_)) ** 2)
    print(f"{source_name} finished with: NSE = {np.round(nse, 2)}")

    # save predictions and performance
    run.save_predictions(prediction_df=yield_df)
    run.save_performance(prediction_df=yield_df[~(yield_df["y_pred"].isna())], cluster_set=data_split)

    # save run
    run.save()

# VISUALIZATION ########################################################################################################
