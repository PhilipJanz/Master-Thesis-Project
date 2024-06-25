import os

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from config import RESULTS_DATA_DIR
from visualizations.visualization_functions import plot_performance_map

"""
This script unfolds the yield estimation performance by giving metrics for each cluster set and model.
Additionally it plots the performance as map (for each admin) and charts.
"""


# LOAD RESULTS ######

# find data
pred_result_dir = RESULTS_DATA_DIR / "yield_predictions/"
pred_result_files = [x for x in os.listdir(pred_result_dir) if (".csv" in x)]

# read data and print performance for each df and model
pred_result_df_ls = []
performance_df_ls = []
for pred_result_file in pred_result_files:
    #print(pred_result_file[:-4])

    # read data
    pred_result_df = pd.read_csv(pred_result_dir / pred_result_file, keep_default_na=False)

    # calc MSE for each model
    pred_columns = [col for col in pred_result_df.columns if col[-4:] == "pred"]
    for col in pred_columns:
        mse = np.mean((pred_result_df[col] - pred_result_df["yield"]) ** 2)
        #print(f"{col[:-5]} MSE: {np.round(mse, 3)} \n")

    pred_result_df_ls.append(pred_result_df)

    performance_dict = {"adm": [], "best_model": [], "bench_mse": [], "mse": [], "nse": []}
    for adm, adm_results_df in pred_result_df.groupby("adm"):
        mse_ls = []
        for col in pred_columns:
            mse = np.mean((adm_results_df[col] - adm_results_df["yield"]) ** 2)
            mse_ls.append(mse)

        # calc metrics
        bench_mse = np.mean((adm_results_df["bench_lin_reg"] - adm_results_df["yield"]) ** 2)
        best_model = pred_columns[np.argmin(mse_ls)][:-5]
        best_mse = np.min(mse_ls)
        best_nse = 1 - best_mse / np.var(adm_results_df["yield"])

        # fill dict
        performance_dict["adm"].append(adm)
        performance_dict["best_model"].append(best_model)
        performance_dict["bench_mse"].append(bench_mse)
        performance_dict["mse"].append(best_mse)
        performance_dict["nse"].append(best_nse)

    performance_df = pd.DataFrame(performance_dict)
    performance_df_ls.append(performance_df)

    # print performance
    print(pred_result_file[:-4], "avg NSE:", np.round(np.mean(performance_dict["nse"]), 2))

    # plot results as a map
    plot_performance_map(performance_data=performance_df, performance_column="nse", result_filename=pred_result_file[:-4])
