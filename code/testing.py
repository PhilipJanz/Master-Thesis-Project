import matplotlib.pyplot as plt
import numpy as np

"""
This is a dirty testing script for whatever is in production 
"""

opti_alpha = np.exp(np.mean(np.log(best_alphas))) # np.median(best_alphas) # opti.best_params["alpha"]
alphas = np.exp(np.arange(-20.8, 0, 0.2))

train_mse = []
test_mse = []
for alpha in alphas:
    opti.best_params["alpha"] = alpha
    X_train_trans, new_predictor_names, trained_model = opti.train_best_model(X=X_train, y=y_train,
                                                                          predictor_names=predictor_names)

    # prepare test-data
    X_test_trans, _ = opti.transform_X(X=X_test, predictor_names=predictor_names, params=opti.best_params)

    # predict train- & test-data
    y_pred_train = trained_model.predict(X_train_trans)
    y_pred = trained_model.predict(X_test_trans)

    # write the predictions into the result df
    train_mse.append(np.mean((y_pred_train - y_train) ** 2))
    test_mse.append(np.mean((y_pred - y_test) ** 2))

opti_alpha = alphas[np.argmin(np.abs((np.max(train_mse) / 2 + np.min(train_mse) / 2) - train_mse))]

plt.plot(alphas, test_mse, label=f"test mse (min at {alphas[np.argmin(test_mse)]})")
plt.plot(alphas, train_mse, label=f"train mse (min at {alphas[np.argmin(train_mse)]})")
plt.hlines(y_test ** 2, xmin=alphas[0], xmax=alphas[-1], color="red", linestyles="dotted", label="zero-predictor",  alpha=0.8)
plt.vlines(opti_alpha, ymin=np.min([test_mse, train_mse]), ymax=np.max([test_mse, train_mse]), color="tab:green", linestyles="dashed", label=f"optuna best parameter: {opti_alpha}",  alpha=0.8)
plt.xlabel("alpha")
plt.ylabel("mse")
plt.title(f"LOYOCV test on {cluster_name} hold-out: {year_out}")
#plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()




opti_alpha = opti.best_params["alpha"]
alphas = np.exp(np.arange(-20.8, 0, 0.1))

train_mse = []
test_mse = []
for alpha in alphas:
    opti.best_params["alpha"] = alpha
    X_train_trans, new_predictor_names, trained_model = opti.train_best_model(X=X_train,
                                                                              y=y_train,
                                                                              predictor_names=predictor_names)

    # prepare test-data
    X_test_trans, _ = opti.transform_X(X=X_test, predictor_names=predictor_names, params=opti.best_params)

    # predict train- & test-data
    y_pred_train = trained_model.predict(X_train_trans)
    y_pred = trained_model.predict(X_test_trans)

    # write the predictions into the result df
    train_mse.append(np.mean((y_pred_train - y_train) ** 2))
    test_mse.append(np.mean((y_pred - y_test) ** 2))

plt.plot(alphas, test_mse, label=f"test mse (min at {alphas[np.argmin(test_mse)]})")
plt.plot(alphas, train_mse, label=f"train mse (min at {alphas[np.argmin(train_mse)]})")
plt.hlines(y_test ** 2, xmin=alphas[0], xmax=alphas[-1], color="red", linestyles="dotted", label="zero-predictor",  alpha=0.8)
plt.vlines(opti_alpha, ymin=np.min([test_mse, train_mse]), ymax=np.max([test_mse, train_mse]), color="tab:green", linestyles="dashed", label=f"optuna best parameter: {opti_alpha}",  alpha=0.8)
plt.xlabel("alpha")
plt.ylabel("mse")
plt.title(f"LOYOCV test on {cluster_name} hold-out: {year_out}")
#plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()



best_alphas = []
# LOYOCV - leave one year out cross validation
for year_out_alpha in np.unique(years_train):
    # if year_out==2011:
    #    break
    year_out_bool = (years_train == year_out_alpha)

    # splitt the data
    X_train_alpha, y_train_alpha = X_train[~year_out_bool], y_train[~year_out_bool]
    X_test_alpha, y_test_alpha = X_train[year_out_bool], y_train[year_out_bool]

    test_mse = []
    for alpha in alphas:
        opti.best_params["alpha"] = alpha
        X_train_trans, new_predictor_names, trained_model = opti.train_best_model(X=X_train_alpha,
                                                                                  y=y_train_alpha,
                                                                                  predictor_names=predictor_names)

        # prepare test-data
        X_test_trans, _ = opti.transform_X(X=X_test_alpha, predictor_names=predictor_names, params=opti.best_params)

        # predict train- & test-data
        y_pred = trained_model.predict(X_test_trans)

        test_mse.append(np.mean((y_pred - y_test_alpha) ** 2))

    best_alphas.append(alphas[np.argmin(test_mse)])

print(np.mean(best_alphas))
print(np.median(best_alphas))


cluster_set = "adm"

corr_mtx = []
adm_ls = []
for cluster_name, cluster_yield_df in cluster_yield_df.groupby("adm"):
    corr_ls = []
    feature_ls = []
    for feature, processed_feature_df in processed_feature_df_dict.items():
        for feature_num in processed_feature_df.columns:
            feature_values = processed_feature_df.loc[cluster_yield_df.index, feature_num].values
            corr = np.corrcoef(np.vstack([feature_values, cluster_yield_df["yield_anomaly"].values]))[0, 1]
            corr_ls.append(corr)
            feature_ls.append(feature_num)
    corr_mtx.append(corr_ls)
    adm_ls.append(cluster_name)
corr_df = pd.DataFrame(corr_mtx, columns=feature_ls, index=adm_ls)

corr_df.mean()
xy = pd.DataFrame({"rolling new": corr_df.mean(), "rolling": roll_corr_df.mean(), "quadratic": quad_corr_df.mean(), "linear": old_corr_df.mean()}, index=corr_df.columns).transpose()



kmean_elbow(data_mtx=corr_df_imputed.iloc[:, :10], max_k=41)
labels, _ = kmean_cluster(data_mtx=corr_df_imputed.iloc[:, :10], n_clusters=20)
yield_df = pd.merge(yield_df, pd.DataFrame({"adm": corr_df.index, "corr_cluster_20": labels}), on="adm")



nn = create_nn(input_shape=X.shape[1], params={'n_layers': 3, 'units_layer1': 128, 'units_layer2': 128, 'units_layer3': 8, 'dropout_layer1': 0.1, 'dropout_layer2': 0.1, 'dropout_layer3': 0.1, "learning_rate": 0.01})
history = nn.fit(X_train, y_train, epochs=500, batch_size=100, shuffle=True, verbose=2, validation_data=(X_test, y_test))

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history.history["loss"], label='Training Loss')
plt.plot(history.history["val_loss"], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')  # Set the y-axis to a logarithmic scale
plt.legend()
plt.show()

y_pred = nn.predict(X_test).T[0]
nse = 1 - np.nanmean((y_pred - y_test) ** 2) / np.mean(y_test ** 2)
print(nse)

# LOYOCV - leave one year out cross validation
for year_out in np.unique(years):
    #if year_out == 2014:
    #    break
    year_out_bool = (years == year_out)

    # split the data
    X_train, y_train, years_train = X[~year_out_bool], y[~year_out_bool], years[~year_out_bool]
    X_test, y_test = X[year_out_bool], y[year_out_bool]

    nn = create_nn(input_shape=X.shape[1],
                   params={'n_layers': 3, 'units_layer1': 128, 'units_layer2': 32, 'units_layer3': 8,
                           'dropout_layer1': 0.1, 'dropout_layer2': 0.1, 'dropout_layer3': 0.1, "learning_rate": 0.001})
    history = nn.fit(X_train, y_train, epochs=500, batch_size=100, shuffle=True, verbose=1,
                     validation_data=(X_test, y_test))

    # predict train- & test-data
    y_pred_train = nn.predict(X_train).T[0]
    y_pred = nn.predict(X_test).T[0]

    # write the predictions into the result df
    yield_df.loc[cluster_yield_df.index[year_out_bool], "train_mse"] = np.mean((y_pred_train - y_train) ** 2)
    yield_df.loc[cluster_yield_df.index[year_out_bool], "y_pred"] = y_pred

preds = yield_df.loc[cluster_yield_df.index]["y_pred"]
y_ = y[~preds.isna()]
preds_ = preds[~preds.isna()]
nse = 1 - np.nanmean((preds_ - y_) ** 2) / np.mean(y_ ** 2)
print(f"{cluster_name} finished with: NSE = {np.round(nse, 2)}")


##################################################################

import os

import numpy as np
import pandas as pd

from metrics import calc_rmse, calc_r2, calc_brr2
from config import RESULTS_DATA_DIR

"""
This script unfolds the yield estimation performance by giving metrics for each cluster set and model.
Additionally it plots the performance as map (for each admin) and charts.
"""


# LOAD RESULTS ######

# find data
pred_result_dir = RESULTS_DATA_DIR / "yield_predictions/"
print(os.listdir(pred_result_dir))

run_name_ls = []
rmse_ls = []
avg_r2_ls = []
n_adm_ls = []
num_r2_above0_ls = []
num_r2_above03_ls = []
for run_name in os.listdir(pred_result_dir):
    if run_name in ["old", "yield"]:
        continue

    if "anomaly" in run_name:
        objective = "yield_anomaly"
    else:
        objective = "yield"

    #model_dir, params_df, feature_ls_ls = run.load_model_and_params()
    #for i, (name, model) in enumerate(model_dir.items()):
    #    importance = model.feature_importances_
    #    print(name, feature_ls_ls[i][np.argmax(importance)], np.max(importance))

    yield_df = pd.read_csv(pred_result_dir / f"{run_name}/prediction.csv", keep_default_na=False)

    result_df = yield_df[yield_df["y_pred"] != ""]
    result_df["y_pred"] = pd.to_numeric(result_df["y_pred"])

    performance_dict = {"adm": [], "rmse": [], "r2": []}
    for adm, adm_results_df in result_df.groupby("adm"):
        y_true = adm_results_df[objective]

        rmse = calc_rmse(y_true=y_true, y_pred=adm_results_df["y_pred"])

        r2 = calc_r2(y_true=y_true, y_pred=adm_results_df["y_pred"])
        #brr2 = calc_brr2(y_true=y_true, y_pred=adm_results_df["y_pred"], y_benchmark=)

        # fill dict
        performance_dict["adm"].append(adm)
        performance_dict["rmse"].append(rmse)
        performance_dict["r2"].append(r2)
        #performance_dict["brr2"].append(brr2)

    performance_df = pd.DataFrame(performance_dict)

    run_name_ls.append(run_name)
    rmse_ls.append(np.sqrt(np.mean((result_df["y_pred"] - result_df[objective]) ** 2)))
    avg_r2_ls.append(np.mean(performance_df["r2"]))
    n_adm_ls.append(len(performance_df))
    num_r2_above0_ls.append(np.sum(performance_df["r2"] > 0))
    num_r2_above03_ls.append(np.sum(performance_df["r2"] > 0.3))

pd.set_option('display.max_columns', None)
overview_df = pd.DataFrame({"run": run_name_ls, "rmse": rmse_ls, "avg_r2": avg_r2_ls,
                            "n_adm": n_adm_ls,
                            "num_r2_above0": num_r2_above0_ls,
                            "num_r2_above03": num_r2_above03_ls})

##################################################################

xy = split_yield_df[year_out_bool]
np.where(xy.adm1 == "Shinyanga")[-1][-1]
shap.plots.waterfall(explainer(X_test_)[20], max_display=10)
X_test_ = pd.DataFrame(X_test, columns=remaining_features)
xyc = yield_df[yield_df.adm1 == "Shinyanga"]

# Create the plot
fig, ax = plt.subplots()

# Hide the top and right spines (outline)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.plot(xyc["harv_year"], xyc["yield"], color="grey", label="True Yield")
plt.plot(xyc["harv_year"], xyc["y_pred"], color="tab:orange", label="Yield Prediction")
plt.legend()
plt.ylabel("Yield (T/ha)")
ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)  # Enable the grid

#plt.savefig("example_predicion.svg")
plt.show()



#################################################################
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data_loader import load_yield_data, load_timeseries_features
feature_file = "32_60_60"
timeseries_feature_ndarray, feature_names, data_id_df = load_timeseries_features(feature_file)
yield_df = load_yield_data()

adm_yield_df = yield_df.iloc[9:23]

fig, ax = plt.subplots(figsize=(5, 5))
plt.plot(adm_yield_df["harv_year"], adm_yield_df["yield"], color="grey")
plt.ylabel("Yield (T/ha)")
# Remove right and upper boundaries
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig("graphics/Dedza_yield_ts.svg", format="svg")
plt.show()

good_idx = 19 # good year 2021
flood_idx = 18 # flood year 2019
drought_idx = 15 # drought year 2016

fig, ax = plt.subplots(3, 1, figsize=(6, 5))

for ax_i, idx in zip(ax, [good_idx, flood_idx, drought_idx]):
    ex = timeseries_feature_ndarray[idx]
    feature_ls = [arr[0][:-2] for arr in feature_names]

    sns.heatmap(ex.T, ax=ax_i, cmap="vlag", vmin=-2.5, vmax=2.5)

    # Set the y-axis (vertical) ticks using feature_names
    ax_i.set_yticks([i + 0.5 for i in range(len(feature_names))])
    ax_i.set_yticklabels(feature_ls, rotation=0, va='center')

    # Remove x-axis ticks
    ax_i.set_xticks([])

    # Customize the color bar to show only "Low" and "High"
    cbar = ax_i.collections[0].colorbar
    cbar.set_ticks([cbar.vmin, cbar.vmax])
    cbar.set_ticklabels(['Low', 'High'])

    # Add a thin border around the heatmap
    for spine in ax_i.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)  # Set the border width to 0.5
plt.tight_layout()
plt.savefig("graphics/good_flood_droght_feature_matricies.svg", format="svg")
plt.show()
