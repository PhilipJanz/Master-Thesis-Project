
import os
import seaborn as sns

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from config import RESULTS_DATA_DIR

"""

"""


# 1. Plot: Evaluating Feature Selection ######

# find data
pred_result_dir = RESULTS_DATA_DIR / "yield_predictions/"
print(os.listdir(pred_result_dir))

# interesting trials (only the non-feature-selection ones)
run_name_ls = ['2110_yield_adm_lasso_1_60_60_optuna_600_100_50_5',
               '2110_yield_adm_lasso_3_60_60_optuna_600_100_50_5',
               '2110_yield_adm_lasso_6_60_60_optuna_600_100_50_5',
               '2110_yield_country_lasso_1_60_60_optuna_600_100_50_5',
               '2110_yield_country_lasso_3_60_60_optuna_600_100_50_5',
               '2110_yield_country_lasso_6_60_60_optuna_600_100_50_5',
               '2110_yield_country_xgb_1_60_60_optuna_1200_250_100_5',
               '2110_yield_country_xgb_3_60_60_optuna_1200_250_100_5',
               '2110_yield_country_xgb_6_60_60_optuna_1200_250_100_5']
# store the data for bar plots as a list
bar_plot_data_ls = []
for run_name in run_name_ls:
    prediction_df = pd.read_csv(pred_result_dir / f"{run_name}/prediction.csv", keep_default_na=False)
    se_ls = (prediction_df["yield"] - prediction_df["y_pred"]) ** 2

    # feature selection version:
    # ajust run name
    run_name = run_name + "_vif5"
    if "xgb" in run_name:
        # xgb has different timeouts for fs and without
        run_name = run_name.replace("1200", "600")
    prediction_df = pd.read_csv(pred_result_dir / f"{run_name}/prediction.csv", keep_default_na=False)
    fs_se_ls = (prediction_df["yield"] - prediction_df["y_pred"]) ** 2

    # fill into list
    bar_plot_data_ls.append({"no-FS": se_ls.values, "MI-VIF": fs_se_ls.values, "Deep Transfer": fs_se_ls.values})

model_names = ["LASSO\n(admin-level)", "LASSO\n(country-level)", "XGBoost\n(country-level)"]

fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(6, 6), sharey=True, width_ratios=[2, 2, 2, 1])

# Flatten the color list if each plot has 3 boxes
colors = ["#EFCFE3", "#EA9AB2"]
dl_color =  "#B3DEE2"

meanprops = {"linewidth": 2, "color": "black"}  # Adjust linewidth as needed
for i, model in enumerate(model_names):
    for j, t in enumerate([1, 3, 6, 32]):
        if j == 3:
        sns.boxplot(
            data=pd.DataFrame(bar_plot_data_ls[j + 3 * i]),
            ax=axs[i, j],
            palette=dl_color,
            showmeans=True,
            meanprops=meanprops
        )
        else:
            sns.boxplot(
                data=pd.DataFrame(bar_plot_data_ls[j + 3 * i]),
                ax=axs[i, j],
                palette=colors,
                showmeans=True,
                meanprops=meanprops
            )

        axs[i, j].set_ylim([0.00, 0.5])
        axs[i, j].set_xticklabels([])

        if (i == 0) & (j != 3):
            axs[i, j].set_title(f't={t}')
    axs[i, 0].set_ylabel(model, rotation=0, labelpad=50, ha='center', va='center')

# Custom legend below the plot
legend_patches = [mpatches.Patch(color=colors[0], label="no FS"),
                  mpatches.Patch(color=colors[1], label="MI-VIF"),
                  mpatches.Patch(color=colors[2], label="Deep Transfer")]
fig.legend(handles=legend_patches, loc="lower center", ncol=3)

plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust layout to make space for the legend
plt.show()
