import pickle

import numpy as np

import pandas as pd

from config import PROCESSED_DATA_DIR
from data_loader import load_yield_data, load_my_cc
from feature_engineering.feature_engineering_functions import max_cdd, process_features_dict

"""
This script processes features based on feature design.
"""

# INITIALIZATION  ####################################################################################################

# length of timeseries of remotes sensing and meteorological features
ts_length = 6

# move the time-window for inference based on the crop calendar (in days)
start_before_sos = 60
end_before_eos = 60


# YIELD & CROP CALENDAR ################################################################################################

# load yield data and benchmark
yield_df = load_yield_data()

# load crop calendar (CC)
cc_df = load_my_cc()


# FEATURES  ####################################################################################################

# location of features
feature_location_dict = {"ndvi": "remote sensing/cleaned_ndvi_regional_matrix.csv",
                         "svi": "remote sensing/svi_regional_matrix.csv",
                         "preci": "climate/preci_regional_matrix.csv",
                         "spi1": "climate/spi1_regional_matrix.csv",
                         "spi6": "climate/spi6_regional_matrix.csv",
                         "temp": "climate/temp_regional_matrix.csv",
                         "sti": "climate/sti_regional_matrix.csv"}

# coose a set of function to perform on the features or set None for only mean
feature_engineering_dict = {"ndvi": {"mean": np.mean, "max": np.max, "min": np.min},
                            "svi": None,
                            "preci": {"max": np.max, "sum": np.sum, "max-cdd": max_cdd},
                            "spi1": None,
                            "spi6": None,
                            "temp": {"mean": np.mean, "max": np.max, "min": np.min},
                            "sti": None
                            }

# load and process features
processed_features_dict = process_features_dict(yield_df=yield_df, cc_df=cc_df,
                                                feature_location_dict=feature_location_dict,
                                                feature_engineering_dict=feature_engineering_dict,
                                                length=ts_length,
                                                start_before_sos=start_before_sos, end_before_eos=end_before_eos)

# form Dataframe
processed_features_df = pd.DataFrame(processed_features_dict)
id_columns = ["country", "adm1", "adm2", "adm", "harv_year"]
processed_features_df[id_columns] = yield_df[id_columns]

# save Dataframe
save_path = PROCESSED_DATA_DIR / f"features/processed_designed_features_df_{ts_length}_{start_before_sos}_{end_before_eos}.csv"
processed_features_df.to_csv(save_path)

#with open(PROCESSED_DATA_DIR / f"features/processed_designed_features_df_{ts_length}_{start_before_sos}_{end_before_eos}.pkl", 'wb') as f:
    # Pickle using the highest protocol available.
#    pickle.dump(processed_features_df, f, pickle.HIGHEST_PROTOCOL)

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

