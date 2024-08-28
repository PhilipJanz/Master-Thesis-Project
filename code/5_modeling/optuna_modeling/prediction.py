import os

from config import BASE_DIR

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from sklearn.decomposition import PCA

from feature_selection import feature_selection_vif, feature_selection_corr_test

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_loader import load_yield_data, load_my_cc, load_cluster_data, load_soil_data
from optuna_modeling.feature_sets_for_optuna import feature_location_dict
from data_assembly import process_list_of_feature_df, make_adm_column, make_X, make_dummies
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

# LOAD & PREPROCESS ####################################################################################################

# load yield data and benchmark
yield_df = load_yield_data()
yield_df = yield_df[(yield_df.harv_year > 2001) & (yield_df.harv_year < 2023)].reset_index(drop=True)
yield_df = make_adm_column(yield_df)
# data unit test
assert len(yield_df) == 1775

# load crop calendar (CC)
cc_df = load_my_cc()

# load and process features
length = 3
processed_feature_df_dict = process_list_of_feature_df(yield_df=yield_df, cc_df=cc_df,
                                                       feature_dict=feature_location_dict,
                                                       length=length,
                                                       start_before_sos=30, end_before_eos=60)
"""
corr_mtx = []
adm_ls = []
for cluster_name, cluster_yield_df in yield_df.groupby("adm"):
    corr_ls = []
    feature_ls = []
    for feature, processed_feature_df in processed_feature_df_dict.items():
        for feature_num in processed_feature_df.columns:
            feature_values = processed_feature_df.loc[cluster_yield_df.index, feature_num].values
            #corr = np.corrcoef(np.vstack([feature_values, cluster_yield_df["yield_anomaly"].values]))[0, 1]
            corr, p_value = kendalltau(feature_values, cluster_yield_df["yield_anomaly"].values)
            if p_value > .1:
                corr = np.nan
            corr_ls.append(corr)
            feature_ls.append(feature_num)
    corr_mtx.append(corr_ls)
    adm_ls.append(cluster_name)
corr_df = pd.DataFrame(corr_mtx, columns=feature_ls, index=adm_ls)
# filter out unplausible data
#unplausible_adm = corr_df[(corr_df.ndvi_1 < 0) & (corr_df["preci-cdd_1"] > 0)].index
"""

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
yield_df.loc[yield_df["country"] == "Tanzania", "adm1_"] = "Tanzania"
cluster_set = "adm"
assert cluster_set in yield_df.columns, f"The chosen cluster-set '{cluster_set}' is not occuring in the yield_df."

# define objective (target)
objective = "yield_anomaly"

# corr test alpha for selecting important features
alpha = None

# vif threshold
vif_threshold = 5

# choose model or set of models that are used
model_type = "lasso"
# choose max feature len for feature shrinking
#max_feature_len = 1
# choose timeout
timeout = 30
# choose duration (sec) of optimization using optuna
n_trials = 100
# choose number of optuna startup trails (random parameter search before sampler gets activated)
n_startup_trials = 50
# folds of optuna hyperparameter search
num_folds = 5

# let's specify tun run (see run.py) using prefix (recommended: MMDD_) and parameters from above
run_name = f"0828_{objective}_{cluster_set}_{model_type}_{length}_{timeout}_{n_trials}_{n_startup_trials}_{num_folds}"
if alpha:
    run_name += f"_corrtest{alpha}"
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
              cluster_set=cluster_set,
              model_types=model_type,
              timeout=timeout,
              n_trials=n_trials,
              n_startup_trials=n_startup_trials,
              python_file=BASE_DIR / "code/5_modeling/optuna_modeling/prediction.py") #os.path.abspath(__file__))

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
    predictors_list = [cluster_yield_df.harv_year] + [df.loc[cluster_yield_df.index] for df in
                                                      processed_feature_df_dict.values()]

    # add dummies for regions
    if cluster_yield_df.adm.nunique() > 1:
        predictors_list.append(soil_df.loc[cluster_yield_df.index])
        predictors_list.append(make_dummies(cluster_yield_df))
    # form the regressor-matrix X
    X, feature_names = make_X(df_ls=predictors_list, standardize=True)
    indicator_ix = np.array(
        [np.any([feature_name in feature for feature_name in list(processed_feature_df_dict.keys())]) for feature in
         feature_names])
    X_indicator = X[:, indicator_ix]
    indicator_names = feature_names[indicator_ix]

    if vif_threshold:
        X, sel_feature_names = feature_selection_vif(X=X,
                                                     feature_names=feature_names,
                                                     indicator_names=indicator_names,
                                                     threshold=vif_threshold)
    else:
        sel_feature_names = feature_names.copy()

    # LOYOCV - leave one year out cross validation
    for year_out in np.unique(years):
        year_out_bool = (years == year_out)

        # in case you loaded an existing run, you can skip the year already predicted
        if np.all(~cluster_yield_df[year_out_bool]["y_pred"].isna()):
            continue

        X_train, y_train, years_train = X[~year_out_bool], y[~year_out_bool], years[~year_out_bool]
        X_test, y_test = X[year_out_bool], y[year_out_bool]

        if alpha:
            X_train, X_test, sel_feature_names_ = feature_selection_corr_test(X_train=X_train,
                                                                              X_test=X_test,
                                                                              y_train=y_train,
                                                                              feature_names=sel_feature_names,
                                                                              indicator_names=indicator_names,
                                                                              alpha=alpha)
        else:
            sel_feature_names_ = sel_feature_names.copy()
        indicator_loc = [feature in indicator_names for feature in sel_feature_names_]

        if not any(indicator_loc):
            # in this case just output 0 as the best estimator for yield anomaly in that scenario
            yield_df.loc[cluster_yield_df.index[year_out_bool], "y_pred"] = 0.0
            yield_df.loc[cluster_yield_df.index[year_out_bool], "best_model"] = "zero-predictor"
            yield_df.loc[cluster_yield_df.index[year_out_bool], "train_mse"] = np.mean(y_train ** 2)
            yield_df.loc[cluster_yield_df.index[year_out_bool], "n_opt_trials"] = 0
            continue

        print(cluster_name, year_out, "selected features: ", sel_feature_names_, "\nmse(y_train)=", np.mean((y_train - np.mean(y_train)) ** 2))

        # make feature-, model- and hyperparameter-selection using optuna
        sampler = optuna.samplers.TPESampler(n_startup_trials=run.n_startup_trials, multivariate=True, warn_independent_sampling=False)
        opti = OptunaOptimizer(X=X_train, y=y_train, years=years_train, predictor_names=sel_feature_names_,
                               sampler=sampler,
                               model_types=run.model_types,
                               feature_set_selection=False, feature_len_shrinking=False,
                               num_folds=num_folds)

        mse, best_params = opti.optimize(n_trials=run.n_trials, timeout=run.timeout,
                                         show_progress_bar=True, print_result=False)


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
                yield_df.loc[cluster_yield_df.index[year_out_bool], "train_mse"] = np.var(y_train)
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
        if len(y_test) > 3:
            if objective == "yield_anomaly":
                test_nse = 1 - test_mse / np.mean(y_test ** 2)
            else:
                test_nse = 1 - test_mse / np.mean((y_test - np.mean(y_test)) ** 2)
        else:
            test_nse = None
        yield_df.loc[cluster_yield_df.index[year_out_bool], "test_nse"] = test_nse
        yield_df.loc[cluster_yield_df.index[year_out_bool], "y_pred"] = y_pred_test
        yield_df.loc[cluster_yield_df.index[year_out_bool], "n_opt_trials"] = len(opti.study.trials)
        if test_nse:
            print(f"{cluster_name} - {year_out} finished with train-mse: {round(train_mse, 3)} ({round(train_nse, 2)}) | test-mse: {round(test_mse, 3)} ({round(test_nse, 2)})")
        else:
            print(f"{cluster_name} - {year_out} finished with train-mse: {round(train_mse, 3)} ({round(train_nse, 2)}) | test-mse: {round(test_mse, 3)}")

        # save predictions and performance
        run.save_predictions(prediction_df=yield_df)


    preds = yield_df.loc[cluster_yield_df.index]["y_pred"]
    y_ = y[~preds.isna()]
    preds_ = preds[~preds.isna()]
    nse = 1 - np.nanmean((preds_ - y_) ** 2) / np.mean((y_ - np.mean(y_)) ** 2)
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
