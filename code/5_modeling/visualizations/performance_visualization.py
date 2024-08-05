import os

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from config import RESULTS_DATA_DIR
from optuna_modeling.run import open_run
from visualizations.visualization_functions import plot_performance_map

"""
This script unfolds the yield estimation performance by giving metrics for each cluster set and model.
Additionally it plots the performance as map (for each admin) and charts.
"""


# LOAD RESULTS ######

# find data
pred_result_dir = RESULTS_DATA_DIR / "yield_predictions/"
print(os.listdir(pred_result_dir))

run_name = '0801_yield_anomaly_adm1__corrtest01_vif2_xgb_6_180_50_30'
run = open_run(run_name=run_name)
model_dir, params_df, feature_ls_ls = run.load_model_and_params()
for i, (name, model) in enumerate(model_dir.items()):
    importance = model.feature_importances_
    print(name, feature_ls_ls[i][np.argmax(importance)], np.max(importance))

yield_df = run.load_prediction()
result_df = yield_df[yield_df["y_pred"] != ""]
result_df["y_pred"] = pd.to_numeric(result_df["y_pred"])

performance_dict = {"adm": [], "adm1_": [], "mse": [], "nse": []}
for adm, adm_results_df in result_df.groupby("adm"):

    mse = np.mean((adm_results_df["y_pred"] - adm_results_df["yield_anomaly"]) ** 2)

    #best_model = pred_columns[np.argmin(mse_ls)][:-5]
    best_nse = 1 - mse / np.mean(adm_results_df["yield_anomaly"] ** 2)

    # fill dict
    performance_dict["adm"].append(adm)
    performance_dict["adm1_"].append(adm_results_df["adm1_"].iloc[0])
    #performance_dict["best_model"].append(best_model)
    performance_dict["mse"].append(mse)
    performance_dict["nse"].append(best_nse)

performance_df = pd.DataFrame(performance_dict)

# print performance
print(run_name, "avg NSE:", np.round(np.mean(performance_dict["nse"]), 2))

# plot results as a map
plot_performance_map(performance_data=performance_df, performance_column="nse", result_filename=run_name)
