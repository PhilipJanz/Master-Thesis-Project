import pickle
import time

from scipy.stats import kendalltau

import os

from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import PROCESSED_DATA_DIR
from data_loader import load_yield_data, load_my_cc, load_soil_data
from optuna_modeling.feature_sets_for_optuna import feature_location_dict
from data_assembly import process_list_of_feature_df, make_adm_column

"""

"""

# INITIALIZATION  ####################################################################################################

# length of timeseries of remotes sensing and meteorological features
ts_length = 1

# number of principal components that are used to make soil-features (using PCA)
soil_pc_number = 2


# YIELD & CROP CALENDAR ################################################################################################

# load yield data and benchmark
yield_df = load_yield_data()

# load crop calendar (CC)
cc_df = load_my_cc()


# FEATURES  ####################################################################################################

# load and process features
processed_feature_df_dict = process_list_of_feature_df(yield_df=yield_df, cc_df=cc_df,
                                                       feature_dict=feature_location_dict,
                                                       length=ts_length,
                                                       start_before_sos=30, end_before_eos=60)

# save params dict
with open(PROCESSED_DATA_DIR / f"features/processed_feature_df_dict_{ts_length}.pkl", 'wb') as f:
    # Pickle using the highest protocol available.
    pickle.dump(processed_feature_df_dict, f, pickle.HIGHEST_PROTOCOL)


# SOIL ##########################

# load soil characteristics
soil_df = load_soil_data()
soil_df = make_adm_column(soil_df)
pca = PCA(n_components=soil_pc_number)
soil_df[["soil_1", "soil_2"]] = pca.fit_transform(soil_df[['clay', 'nitrogen', 'phh2o', 'sand', 'silt', 'soc']])
print("explained_variance_ratio_ of PCA on soil:", pca.explained_variance_ratio_)
soil_df = pd.merge(yield_df["adm"], soil_df, on=["adm"], how="left")
soil_df = soil_df[["soil_1", "soil_2"]] # soil_df[['clay', 'elevation', 'nitrogen', 'phh2o', 'sand', 'silt', 'soc']]
# scale each column (soil property) individually
soil_df.iloc[:, :] = StandardScaler().fit_transform(soil_df.values)

# save
soil_df.to_csv(PROCESSED_DATA_DIR / f"features/soil_pca_{soil_pc_number}.csv", index=False)


# VISUALIZATION ##########################

"""
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
#unplausible_adm = corr_df[(corr_df.ndvi_1 < 0) & (corr_df["preci-cdd_1"] > 0)].index
"""

