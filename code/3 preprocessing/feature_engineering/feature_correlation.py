import time

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR
from crop_calendar.crop_calendar_functions import load_my_cc
from data_assembly import process_feature_df
from yield_.yield_functions import load_yield_data

# load yield data and benchmark
yield_df = load_yield_data()
yield_df = yield_df[~yield_df.country.isin(["Ethiopia", "Kenya"])]
yield_df = yield_df[yield_df.harv_year > 2001]
yield_df.sort_values(["country", "adm1", "adm2", "harv_year"], inplace=True, ignore_index=True)

# load crop calendar (CC)
cc_df = load_my_cc()

# load features and make feature dataframe for yield data with respect to the CC
feature_ls = ["remote sensing/smooth_ndvi_regional_matrix.csv", "remote sensing/si_smooth_ndvi_regional_matrix.csv",
              "remote sensing/smooth_evi_regional_matrix.csv", "remote sensing/si_smooth_evi_regional_matrix.csv",
              "climate/pr_sum_regional_matrix.csv", "climate/si_pr_sum_regional_matrix.csv",
              "climate/pr_max_regional_matrix.csv", "climate/pr_cddMax_regional_matrix.csv",
              "climate/pr_belowP01_regional_matrix.csv", "climate/pr_aboveP99_regional_matrix.csv",
              "climate/tas_median_regional_matrix.csv", "climate/si_tas_median_regional_matrix.csv",
              "climate/tasmin_median_regional_matrix.csv", "climate/tasmax_median_regional_matrix.csv",
              "climate/tasmin_belowP01_regional_matrix.csv", "climate/tasmax_aboveP99_regional_matrix.csv"] #
feature_name_ls = ["ndvi", "si-ndvi", "evi", "si-evi",
                   "preci-sum", "si-preci-sum", "pr-max", "pr-cdd", "pr-belowP01", "pr-aboveP99",
                   "temp-median", "si-temp-median", "min-temp-median", "max-temp-median",
                   "min-temp-belowP01", "max-temp-aboveP99"]  # ,

processed_feature_df_ls = []
for feature, feature_name in zip(feature_ls, feature_name_ls):
    feature_df = pd.read_csv(PROCESSED_DATA_DIR / feature, keep_default_na=False)

    # apply CC for each yield datapoint
    processed_feature_df = process_feature_df(yield_df=yield_df,
                                              cc_df=cc_df,
                                              feature_df=feature_df,
                                              feature_name=feature_name,
                                              length=20,
                                              start_before_sos=0,
                                              end_before_eos=60)
    processed_feature_df_ls.append(processed_feature_df)


# 1. PLOT: CORRELATION OVER YEARLY AVERAGE #####

yearly_avg_processed_feature_mtx = np.array([np.mean(df.values, 1) for df in processed_feature_df_ls])
yearly_avg_processed_feature_df = pd.DataFrame(yearly_avg_processed_feature_mtx.T, columns=feature_name_ls)
corr = yearly_avg_processed_feature_df.corr()

plt.figure(figsize=(13, 12))  # Set the size of the plot
sns.set(style='white')  # Set the style to white

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Create the heatmap
heatmap = sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f",
                      vmax=1.0, vmin=-1.0)

# Add titles and labels
plt.title('Correlation matrix for features (yearly average)', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Show and save the plot
plt.savefig(PROCESSED_DATA_DIR / "feature engineering/plots/over_year_correlation_mtx.png", dpi=600)
plt.show()


# 2. PLOT: CORRELATION WITHIN YEAR #####

corr = np.zeros((len(feature_ls), len(feature_ls)))
for i, df1 in enumerate(processed_feature_df_ls):
    for j, df2 in enumerate(processed_feature_df_ls):
        if j >= i:
            continue
        corr[i, j] = np.nanmean([np.corrcoef(df1.loc[ix], df2.loc[ix])[0, 1] for ix in df1.index])
corr = pd.DataFrame(corr, index=feature_name_ls, columns=feature_name_ls)


plt.figure(figsize=(13, 12))  # Set the size of the plot
sns.set(style='white')  # Set the style to white

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Create the heatmap
heatmap = sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f",
                      vmax=1.0, vmin=-1.0)

# Add titles and labels
plt.title('Correlation matrix for features (within one season)', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Show and save the plot
plt.savefig(PROCESSED_DATA_DIR / "feature engineering/plots/within_year_correlation_mtx.png", dpi=600)
plt.show()
