import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_DATA_DIR, COUNTRY_COLORS, COUNTRY_MARKERS
from data_loader import load_yield_data
from metrics import calc_rmse, calc_r2, calc_brr2
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

run_name = '1810_yield_country_transfer_features_from_all_xgb_600_250_100_10'
#run = open_run(run_name=run_name)
#model_dir, params_df, feature_ls_ls = run.load_model_and_params()
#for i, (name, model) in enumerate(model_dir.items()):
#    importance = model.feature_importances_
#    print(name, feature_ls_ls[i][np.argmax(importance)], np.max(importance))

prediction_df = pd.read_csv(pred_result_dir / f"{run_name}/prediction.csv") # run.load_prediction()
prediction_df = prediction_df[(prediction_df["y_pred"] != "") & (~prediction_df["y_pred"].isna())]
prediction_df["y_pred"] = pd.to_numeric(prediction_df["y_pred"])

if "y_bench" not in prediction_df:
    benchmark_df = load_yield_data(benchmark_column=True)
    # test if datasets have same row order
    assert np.all(prediction_df[["adm", "harv_year"]] == benchmark_df[["adm", "harv_year"]])

    prediction_df["y_bench"] = benchmark_df["y_bench"]

if "anomaly" in run_name:
    objective = "yield_anomaly"
else:
    objective = "yield"

rmse = calc_rmse(y_pred=prediction_df["y_pred"], y_true=prediction_df[objective])
brr2 = calc_brr2(y_pred=prediction_df["y_pred"], y_true=prediction_df[objective], y_benchmark=prediction_df["y_bench"])
print(f"Run '{run_name}' achieved RMSE (BR-R2): {np.round(rmse, 3)} ({np.round(brr2, 3)})")


performance_dict = {"country": [], "adm": [], "n": [], "rmse": [], "r2": [], "brr2": []}
for adm, adm_results_df in prediction_df.groupby("adm"):
    if len(adm_results_df) < 3:
        continue
    y_true = adm_results_df[objective]

    rmse = calc_rmse(y_true=y_true, y_pred=adm_results_df["y_pred"])

    r2 = calc_r2(y_true=y_true, y_pred=adm_results_df["y_pred"])
    brr2 = calc_brr2(y_pred=adm_results_df["y_pred"], y_true=y_true, y_benchmark=adm_results_df["y_bench"])

    # fill dict
    performance_dict["country"].append(adm_results_df["country"].values[0])
    performance_dict["adm"].append(adm)
    performance_dict["n"].append(len(adm_results_df))
    performance_dict["rmse"].append(rmse)
    performance_dict["r2"].append(r2)
    performance_dict["brr2"].append(brr2)
    if "Chipata" in adm: # Chipata pwani
        fig, ax = plt.subplots()
        ax.plot(adm_results_df["harv_year"], y_true, label="True Yield")
        ax.plot(adm_results_df["harv_year"], adm_results_df["y_bench"], label="Benchmark Prediction")
        ax.plot(adm_results_df["harv_year"], adm_results_df["y_pred"], label="Model Prediction")
        ax.text(0.9, 0.05, f"R2: {np.round(r2, 2)}\nBR-R2: {np.round(brr2, 2)}", transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left')
        plt.legend()
        plt.show()


fig, ax = plt.subplots(1, 2, figsize=(12, 3))
ax1, ax2 = ax
adm_results_df = prediction_df[prediction_df.adm2 == "Dedza"]
y_true = adm_results_df[objective]
rmse = calc_rmse(y_true=y_true, y_pred=adm_results_df["y_pred"])
r2 = calc_r2(y_true=y_true, y_pred=adm_results_df["y_pred"])
brr2 = calc_brr2(y_pred=adm_results_df["y_pred"], y_true=y_true, y_benchmark=adm_results_df["y_bench"])

ax1.plot(adm_results_df["harv_year"], adm_results_df["yield"], label="True Yield", color="grey")
ax1.plot(adm_results_df["harv_year"], adm_results_df["y_bench"], label="Benchmark Prediction", alpha=.8, linestyle="--")
ax1.plot(adm_results_df["harv_year"], adm_results_df["y_pred"], label="Model Prediction")
ax1.text(0.7, 0.05, f"R2: {np.round(r2, 2)}\nBR-R2: {np.round(brr2, 2)}", transform=ax1.transAxes, verticalalignment='bottom', horizontalalignment='left')

ax1.set_ylabel("Yield (T/ha)")

adm_results_df = prediction_df[prediction_df.adm1 == "Singida"]
y_true = adm_results_df[objective]
rmse = calc_rmse(y_true=y_true, y_pred=adm_results_df["y_pred"])
r2 = calc_r2(y_true=y_true, y_pred=adm_results_df["y_pred"])
brr2 = calc_brr2(y_pred=adm_results_df["y_pred"], y_true=y_true, y_benchmark=adm_results_df["y_bench"])

ax2.plot(adm_results_df["harv_year"], adm_results_df["yield"], label="True Yield", color="grey")
ax2.plot(adm_results_df["harv_year"], adm_results_df["y_bench"], label="Benchmark Prediction", alpha=.8, linestyle="--")
ax2.plot(adm_results_df["harv_year"], adm_results_df["y_pred"], label="Model Prediction")
ax2.text(0.7, 0.05, f"R2: {np.round(r2, 2)}\nBR-R2: {np.round(brr2, 2)}", transform=ax2.transAxes, verticalalignment='bottom', horizontalalignment='left')

plt.legend()
plt.savefig("br_r2_motivation.pdf", format="pdf") # TODO path
plt.show()

performance_df = pd.DataFrame(performance_dict)

# plot results as a map
plot_performance_map(performance_data=performance_df.drop("country", axis=1), performance_column="r2", result_filename=run_name)
plot_performance_map(performance_data=performance_df.drop("country", axis=1), performance_column="brr2", result_filename=run_name, cmap="RdYlGn")


for c, country_df in performance_df.groupby("country"):
    plt.scatter(country_df["n"], country_df["nse"], color="grey", alpha=0.7, marker=COUNTRY_MARKERS[c], label=c)
plt.xlabel("Number of data points in region")
plt.ylabel("NSE")
plt.legend()
plt.show()
