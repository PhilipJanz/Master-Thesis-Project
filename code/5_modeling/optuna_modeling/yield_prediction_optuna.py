import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from mlxtend.feature_selection import SequentialFeatureSelector
from scipy.stats import kendalltau
from sklearn.decomposition import PCA

from config import PROCESSED_DATA_DIR
from feature_selection import feature_selection_vif, backwards_feature_selection


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from data_loader import load_yield_data, load_my_cc, load_cluster_data, load_soil_data
from optuna_modeling.feature_sets_for_optuna import feature_location_dict
from data_assembly import process_list_of_feature_df, make_adm_column, make_X, make_dummies
from optuna_modeling.run import list_of_runs, Run, open_run
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

# load crop calendar (CC)
cc_df = load_my_cc()

# load and process features
length = 10
processed_feature_df_dict = process_list_of_feature_df(yield_df=yield_df, cc_df=cc_df,
                                                       feature_dict=feature_location_dict,
                                                       length=length,
                                                       start_before_sos=30, end_before_eos=60)

#x = pd.concat(processed_feature_df_dict.values(), axis=1)
#x.to_csv(PROCESSED_DATA_DIR / "xy/x.csv", index=False)
#yield_df.to_csv(PROCESSED_DATA_DIR / "xy/y.csv", index=False)

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
unplausible_adm = corr_df[(corr_df.ndvi_1 < 0) & (corr_df["preci-cdd_1"] > 0)].index
#yield_df = yield_df[~yield_df.adm.isin(unplausible_adm)]


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
cluster_set = "country"
assert cluster_set in yield_df.columns, f"The chosen cluster-set '{cluster_set}' is not occuring in the yield_df."

# define objective (target)
objective = "yield_anomaly"

# vif thresgold
vif_threshold = 2

# choose model or set of models that are used
model_types = ["nn"]
# choose max feature len for feature shrinking
max_feature_len = 1
# choose duration (sec) of optimization using optuna
opti_duration = 180
# choose number of optuna startup trails (random parameter search before sampler gets activated)
n_startup_trials = 50
# folds of optuna hyperparameter search
num_folds = 5

# let's specify tun run (see run.py) using prefix (recommended: MMDD_) and parameters from above
run_name = f"0804_{objective}_{cluster_set}_corrtest005_vif{vif_threshold}_{'-'.join(model_types)}_{length}_{opti_duration}_{n_startup_trials}_{num_folds}" #

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
    #break
    if "Tanzania" in cluster_yield_df["country"].values:
        continue

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

    # filter features of X
    #selected_features = ['harv_year', 'si-ndvi_1'] # , 'preci-max_1'
    #selected_features_loc = [feat in selected_features for feat in predictor_names]
    #X = X[:, selected_features_loc]
    #predictor_names = predictor_names[selected_features_loc]

    # LOYOCV - leave one year out cross validation
    for year_out in np.unique(years):
        #if year_out==2015:
        #    break
        year_out_bool = (years == year_out)

        # split the data
        X_train, y_train, years_train = X[~year_out_bool], y[~year_out_bool], years[~year_out_bool]
        X_test, y_test = X[year_out_bool], y[year_out_bool]

        selected_feature_ix = np.repeat(True, len(feature_names))
        indicator_feature_ls = []
        for i, feature in enumerate(feature_names):
            if not np.any([feature_name in feature for feature_name in list(processed_feature_df_dict.keys())]):
                continue
            corr, p_value = kendalltau(X_train[:, i], y_train)
            if p_value > 0.05:
                selected_feature_ix[i] = False
            else:
                selected_feature_ix[i] = True
                indicator_feature_ls.append(feature)

        #"""
        if not indicator_feature_ls:
            # in this case just output 0 as the best estimator for yield anomaly in that scenario
            yield_df.loc[cluster_yield_df.index[year_out_bool], "y_pred"] = 0.0
            yield_df.loc[cluster_yield_df.index[year_out_bool], "best_model"] = "zero-predictor"
            yield_df.loc[cluster_yield_df.index[year_out_bool], "train_mse"] = np.mean(y_train ** 2)
            yield_df.loc[cluster_yield_df.index[year_out_bool], "n_opt_trials"] = 0
            continue

        corr_selected_feature_names = feature_names[selected_feature_ix]
        X_train = X_train[:, selected_feature_ix]
        X_test = X_test[:, selected_feature_ix]
        #"""
        #corr_selected_feature_names = feature_names

        # backward selection feature selection
        #X_train, selected_feature_names, fs = backwards_feature_selection(X=X_train, y=y_train,
        #                                                                  feature_names=selected_feature_names)
        #X_test = fs.transform(X_test)

        # filter by correlation with yield
        #corrs = np.corrcoef(np.vstack([X_train.T, y_train]))[:-1, -1]
        #interesting_features = predictor_names[np.abs(corrs) >= 0.3]
        #X_train = X_train[:, [name in interesting_features for name in predictor_names]]
        #X_test = X_test[:, [name in interesting_features for name in predictor_names]]

        # check if features are left
        #if len(interesting_features) == 0:
        #    # in this case just output 0 as the best estimator for yield anomaly in that scenario
        #    yield_df.loc[cluster_yield_df.index[year_out_bool], "y_pred"] = 0.0
        #    yield_df.loc[cluster_yield_df.index[year_out_bool], "best_model"] = "zero-predictor"
        #    yield_df.loc[cluster_yield_df.index[year_out_bool], "train_mse"] = np.var(y_train)
        #    yield_df.loc[cluster_yield_df.index[year_out_bool], "n_opt_trials"] = len(opti.study.trials)
        #    continue

        # vif
        #indicator_feature_ix = np.array([feature in indicator_feature_ls for feature in corr_selected_feature_names])
        #X_vif, vif_selected_feature_names = feature_selection_vif(X_train[:, indicator_feature_ix],
        #                                                          corr_selected_feature_names[indicator_feature_ix],
        #                                                          threshold=2)
        X_train, vif_selected_feature_names = feature_selection_vif(X=X_train,
                                                                    feature_names=corr_selected_feature_names,
                                                                    indicator_features=indicator_feature_ls,
                                                                    threshold=vif_threshold)
        #X_train = np.hstack([X_vif, X_train[:, ~np.array(indicator_feature_ix)]])
        #selected_feature_names = np.array(vif_selected_feature_names + list(corr_selected_feature_names[~indicator_feature_ix]))
        selected_feature_names = np.array(vif_selected_feature_names)
        X_test = X_test[:, [name in selected_feature_names for name in corr_selected_feature_names]]
        #indicator_feature_ix = np.concatenate([np.repeat(True, len(vif_selected_feature_names)),
        #                                       np.repeat(False, sum(~indicator_feature_ix))])
        print(cluster_name, year_out, "selected features: ", selected_feature_names, "\nmse(y_train)=", np.mean(y_train ** 2))

        # make feature-, model- and hyperparameter-selection using optuna
        sampler = optuna.samplers.TPESampler(n_startup_trials=run.n_startup_trials, multivariate=True, warn_independent_sampling=False)
        opti = OptunaOptimizer(X=X_train, y=y_train, years=years_train, predictor_names=selected_feature_names,
                               sampler=sampler,
                               model_types=run.model_types,
                               feature_set_selection=False, feature_len_shrinking=False,
                               max_feature_len=max_feature_len, num_folds=num_folds, seed=42)

        mse, best_params = opti.optimize(n_trials=100000, timeout=run.opti_duration,
                                         show_progress_bar=True, print_result=False)


        # train best model
        #X_train_trans, new_predictor_names = opti.transform_X(X=X_train, predictor_names=selected_feature_names, params=opti.best_params)

        # train best model
        X_train_trans, optuna_selected_feature_names, trained_model = opti.train_best_model(X=X_train, y=y_train,
                                                                                  predictor_names=selected_feature_names)

        if opti.best_params["model_type"] == "xgb":

            indicator_feature_ls = [sel_feature for sel_feature in optuna_selected_feature_names if np.any([feature in sel_feature for feature in list(processed_feature_df_dict.keys())])]
            if not indicator_feature_ls:
                # in this case just output 0 as the best estimator for yield anomaly in that scenario
                yield_df.loc[cluster_yield_df.index[year_out_bool], "y_pred"] = 0.0
                yield_df.loc[cluster_yield_df.index[year_out_bool], "best_model"] = "zero-predictor"
                yield_df.loc[cluster_yield_df.index[year_out_bool], "train_mse"] = np.var(y_train)
                yield_df.loc[cluster_yield_df.index[year_out_bool], "n_opt_trials"] = len(opti.study.trials)
                continue
        if opti.best_params["model_type"] == "lasso":
            if np.all(trained_model.coef_[optuna_selected_feature_names != "harv_year"] == 0):
                # in this case just output 0 as the best estimator for yield anomaly in that scenario
                yield_df.loc[cluster_yield_df.index[year_out_bool], "y_pred"] = 0.0
                yield_df.loc[cluster_yield_df.index[year_out_bool], "best_model"] = "zero-predictor"
                yield_df.loc[cluster_yield_df.index[year_out_bool], "train_mse"] = np.var(y_train)
                yield_df.loc[cluster_yield_df.index[year_out_bool], "n_opt_trials"] = len(opti.study.trials)
                continue

        # prepare test-data
        X_test_trans, _ = opti.transform_X(X=X_test, predictor_names=selected_feature_names, params=opti.best_params)

        # predict train- & test-data
        y_pred_train = trained_model.predict(X_train_trans)
        y_pred = trained_model.predict(X_test_trans)

        # write the predictions into the result df
        yield_df.loc[cluster_yield_df.index[year_out_bool], "train_mse"] = np.mean((y_pred_train - y_train) ** 2)
        yield_df.loc[cluster_yield_df.index[year_out_bool], "y_pred"] = y_pred
        yield_df.loc[cluster_yield_df.index[year_out_bool], "best_model"] = opti.best_params['model_type']
        yield_df.loc[cluster_yield_df.index[year_out_bool], "n_opt_trials"] = len(opti.study.trials)

        # save trained model and best params
        trained_model.feature_names = optuna_selected_feature_names
        opti.best_params["feature_names"] = optuna_selected_feature_names
        run.save_model_and_params(name=f"{cluster_name}_{year_out}", model=trained_model, params=opti.best_params)

        """
        # Plotting:
        optuna_alpha = best_params["alpha"]
        train_mse = []
        test_mse = []
        y_preds = []
        alphas = np.exp(np.arange(-20.8, 0, 0.2))
        for alpha in alphas:
            opti.best_params["alpha"] = alpha
            X_train_trans, new_predictor_names, trained_model = opti.train_best_model(X=X_train, y=y_train,
                                                                                      predictor_names=predictor_names)

            # prepare test-data
            X_test_trans, _ = opti.transform_X(X=X_test, predictor_names=predictor_names, params=opti.best_params)

            # predict train- & test-data
            y_pred_train = trained_model.predict(X_train_trans)
            y_pred = trained_model.predict(X_test_trans)
            y_preds.append(y_pred)

            # write the predictions into the result df
            train_mse.append(np.mean((y_pred_train - y_train) ** 2))
            test_mse.append(np.mean((y_pred - y_test) ** 2))


        plt.plot(alphas, test_mse, label=f"test mse (min at {alphas[np.argmin(test_mse)]})")
        plt.plot(alphas, train_mse, label=f"train mse (min at {alphas[np.argmin(train_mse)]})")
        plt.hlines(y_test ** 2, xmin=alphas[0], xmax=alphas[-1], color="red", linestyles="dotted",
                   label="zero-predictor", alpha=0.8)
        plt.vlines(optuna_alpha, ymin=np.min([test_mse, train_mse]), ymax=np.max([test_mse, train_mse]),
                   color="tab:green", linestyles="dashed", label=f"optuna best parameter: {optuna_alpha}", alpha=0.8)
        plt.xlabel("alpha")
        plt.ylabel("mse")
        plt.title(f"LOYOCV test on {cluster_name} hold-out: {year_out}")
        # plt.yscale("log")
        plt.xscale("log")
        plt.legend()
        plt.savefig(run.run_dir / f"plots/regional/alpha_choice_{cluster_name}_{year_out}.png")
        plt.close()
        """

    preds = yield_df.loc[cluster_yield_df.index]["y_pred"]
    y_ = y[~preds.isna()]
    preds_ = preds[~preds.isna()]
    nse = 1 - np.nanmean((preds_ - y_) ** 2) / np.mean(y_ ** 2)
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
