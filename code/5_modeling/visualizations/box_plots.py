
import os
import seaborn as sns

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from config import RESULTS_DATA_DIR, GRAPHICS_DIR

"""
Box-Plot for comparing squared-error for multiple models.
"""


# 1. Plot: Evaluating Feature Selection ######

# find data
pred_result_dir = RESULTS_DATA_DIR / "yield_predictions/"
print(os.listdir(pred_result_dir))

# interesting trials (only the non-feature-selection ones)
run_name_ls = ['2110_yield_country_lasso_1_60_60_optuna_600_100_50_5',
               '2110_yield_country_lasso_3_60_60_optuna_600_100_50_5',
               '2110_yield_country_lasso_6_60_60_optuna_600_100_50_5',
               "3110_yield_country_external_transfer_features_from_all_lasso_1200_100_50_5",
               '2110_yield_country_xgb_1_60_60_optuna_1200_250_100_5',
               '2110_yield_country_xgb_3_60_60_optuna_1200_250_100_5',
               '2110_yield_country_xgb_6_60_60_optuna_1200_250_100_5',
               "3110_yield_country_external_transfer_features_from_all_xgb_1200_250_100_5"] # TODO renew
# store the data for bar plots as a list
bar_plot_data_ls = []
for run_name in run_name_ls:
    prediction_df = pd.read_csv(pred_result_dir / f"{run_name}/prediction.csv", keep_default_na=False)
    se_ls = (prediction_df["yield"] - prediction_df["y_pred"]) ** 2

    if "transfer" in run_name:
        internal_name = run_name.replace("ex", "in")
        prediction_df = pd.read_csv(pred_result_dir / f"{internal_name}/prediction.csv", keep_default_na=False)
        internal_se_ls = (prediction_df["yield"] - prediction_df["y_pred"]) ** 2

        bar_plot_data_ls.append({"External Transfer": se_ls.values, "Internal Transfer": internal_se_ls.values})
        continue

    # feature selection version:
    # ajust run name
    run_name = run_name + "_vif5"
    if "xgb" in run_name:
        # xgb has different timeouts for fs and without
        run_name = run_name.replace("1200", "600")
    prediction_df = pd.read_csv(pred_result_dir / f"{run_name}/prediction.csv", keep_default_na=False)
    fs_se_ls = (prediction_df["yield"] - prediction_df["y_pred"]) ** 2

    # fill into list
    bar_plot_data_ls.append({"no-FS": se_ls.values, "MI-VIF": fs_se_ls.values})

model_names = ["LASSO", "XGBoost"]

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(6, 6), sharey=True, width_ratios=[2, 2, 2, 2])

# Flatten the color list if each plot has 3 boxes
colors = ["#e7e7e7", "#bebebe"]
dl_color = ["#979797", "#6e6e6e"]

medianprops = {"linewidth": 1.5, "color": "black"}  # Adjust linewidth as needed
for i, model in enumerate(model_names):
    mse_ls = []
    for j, t in enumerate([1, 3, 6, 32]):
        mse_ls.append(np.min([np.mean(values) for _, values in bar_plot_data_ls[j + 4 * i].items()]))
        if j == 3:
            sns.boxplot(
                data=pd.DataFrame(bar_plot_data_ls[j + 4 * i]),
                ax=axs[i, j],
                palette=dl_color,
                showmeans=True,
                medianprops=medianprops
            )
        else:
            sns.boxplot(
                data=pd.DataFrame(bar_plot_data_ls[j + 4 * i]),
                ax=axs[i, j],
                palette=colors,
                showmeans=True,
                medianprops=medianprops
            )

        axs[i, j].set_ylim([-0.02, 0.45])
        axs[i, j].set_xticklabels([])
        axs[i, j].get_xaxis().set_visible(False)
        if j == 0:
            axs[i, j].spines[['right', "top", "bottom"]].set_visible(False)
        #elif j == 3:
        #    axs[i, j].get_yaxis().set_visible(False)
         #   axs[i, j].spines[['left', "top", "bottom"]].set_visible(False)
        else:
            axs[i, j].get_yaxis().set_visible(False)
            axs[i, j].spines[['right', 'left', "top", "bottom"]].set_visible(False)

        if (i == 0) & (j != 3):
            axs[i, j].set_title(f't={t}')

    # Add dashed horizontal line at best mse
    for j in range(4):
        axs[i, j].axhline(np.min(mse_ls), color='tab:green', linestyle='--', linewidth=1)

    axs[i, 0].set_ylabel(model, rotation=0, labelpad=50, ha='center', va='center')

# Custom legend below the plot
legend_patches = [mpatches.Patch(color=colors[0], label="no FS"),
                  mpatches.Patch(color=colors[1], label="MI-VIF"),
                  mpatches.Patch(color=dl_color[0], label="External Transfer"),
                  mpatches.Patch(color=dl_color[1], label="Internal Transfer")]
fig.legend(handles=legend_patches, loc="lower center", ncol=4, bbox_to_anchor=(.5, -0.05))

plt.tight_layout() # rect=[0, 0.1, 1, 1])  # Adjust layout to make space for the legend

# Show and save the plot
save_path = GRAPHICS_DIR / "boxplot_feature_select_vs_deep_transfer.pdf"
plt.savefig(save_path, format="pdf", dpi=150, bbox_inches='tight')
plt.show()
