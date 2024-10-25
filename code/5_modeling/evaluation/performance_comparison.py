
import os

import numpy as np
import pandas as pd

from data_loader import load_yield_data, yield_datapoints_after_filtering
from evaluation.optuna_inspection import calculate_total_runtime_and_average_trials
from metrics import calc_rmse, calc_r2, calc_brr2, mean_estimator, calc_mse
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
trials_ls = []
runtime_ls = []
r2_ls = []
brr2_ls = []
for run_name in os.listdir(pred_result_dir):
    if run_name in ["old", "yield"]:
        continue

    prediction_df = pd.read_csv(pred_result_dir / f"{run_name}/prediction.csv", keep_default_na=False)

    prediction_df = prediction_df[prediction_df["y_pred"] != ""]
    prediction_df["y_pred"] = pd.to_numeric(prediction_df["y_pred"])


    # only take completed runs
    if len(prediction_df) != yield_datapoints_after_filtering:
        continue

    if "y_bench" not in prediction_df:
        benchmark_df = load_yield_data(benchmark_column=True)
        # test if datasets have same row order
        assert np.all(prediction_df[["adm", "harv_year"]] == benchmark_df[["adm", "harv_year"]])

        prediction_df["y_bench"] = benchmark_df["y_bench"]

    #

    prediction_df["y_mean"] = np.nan
    for adm, adm_results_df in prediction_df.groupby("adm"):
        y_true = adm_results_df["yield"]
        y_mean = mean_estimator(y_true)
        prediction_df.loc[adm_results_df.index, "y_mean"] = y_mean

    # get runtime and avg number of trials
    runtime, avg_n_trials = calculate_total_runtime_and_average_trials(pred_result_dir / f"{run_name}/optuna_studies/")

    run_name_ls.append(run_name)
    runtime_ls.append(np.round(runtime, 1))
    trials_ls.append(int(avg_n_trials))
    rmse_ls.append(calc_rmse(y_true=prediction_df["yield"], y_pred=prediction_df["y_pred"]))
    r2_ls.append(calc_brr2(y_true=prediction_df["yield"], y_pred=prediction_df["y_pred"], y_benchmark=prediction_df["y_mean"]))
    brr2_ls.append(calc_brr2(y_true=prediction_df["yield"], y_pred=prediction_df["y_pred"], y_benchmark=prediction_df["y_bench"]))

pd.set_option('display.max_columns', None)
overview_df = pd.DataFrame({"run": run_name_ls, "runtime": runtime_ls, "trials": trials_ls,
                            "rmse": rmse_ls, "r2": r2_ls, "brr2": brr2_ls})
