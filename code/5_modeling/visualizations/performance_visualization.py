import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_DATA_DIR, COUNTRY_COLORS
from run import open_run
from visualizations.visualization_functions import plot_performance_map, plot_map

"""
This script unfolds the yield estimation performance by giving metrics for each cluster set and model.
Additionally it plots the performance as map (for each admin) and charts.
"""


# LOAD RESULTS ######

# find data
pred_result_dir = RESULTS_DATA_DIR / "yield_predictions/"
print(os.listdir(pred_result_dir))

run_name = '1010_yield_country_xgb_3_60_60_optuna_600_250_100_5'
#run = open_run(run_name=run_name)
#model_dir, params_df, feature_ls_ls = run.load_model_and_params()
#for i, (name, model) in enumerate(model_dir.items()):
#    importance = model.feature_importances_
#    print(name, feature_ls_ls[i][np.argmax(importance)], np.max(importance))

yield_df = pd.read_csv(pred_result_dir / f"{run_name}/prediction.csv") # run.load_prediction()
result_df = yield_df[(yield_df["y_pred"] != "") & (~yield_df["y_pred"].isna())]
result_df["y_pred"] = pd.to_numeric(result_df["y_pred"])

if "anomaly" in run_name:
    objective = "yield_anomaly"
else:
    objective = "yield"

performance_dict = {"country": [], "adm": [], "n": [], "mse": [], "nse": []}
for adm, adm_results_df in result_df.groupby("adm"):
    y_true = adm_results_df[objective]

    mse = np.mean((adm_results_df["y_pred"] - y_true) ** 2)

    if objective == "yield_anomaly":
        nse = 1 - mse / np.mean(y_true ** 2)
    else:
        nse = 1 - mse / np.mean((y_true - np.mean(y_true)) ** 2)
    # fill dict
    performance_dict["country"].append(adm_results_df["country"].values[0])
    performance_dict["adm"].append(adm)
    performance_dict["n"].append(len(adm_results_df))
    performance_dict["mse"].append(mse)
    performance_dict["nse"].append(nse)
    """
    if "Malawi" in adm:
        fig, ax = plt.subplots()
        ax.plot(adm_results_df["harv_year"], y_true)
        ax.plot(adm_results_df["harv_year"], adm_results_df["y_pred"])
        plt.title(f"{adm} {nse}")
        plt.show()
    """
performance_df = pd.DataFrame(performance_dict)

# print performance
print(run_name, "avg NSE:", np.round(np.mean(performance_dict["nse"]), 2))

# plot results as a map
plot_performance_map(performance_data=performance_df.drop("country", axis=1), performance_column="nse", result_filename=run_name)


for c, country_df in performance_df.groupby("country"):
    plt.scatter(country_df["n"], country_df["nse"], color="grey", alpha=0.6, marker='s')
plt.xlabel("Number of data points in region")
plt.ylabel("NSE")
plt.show()
