import pickle

import numpy as np

import pandas as pd

from config import PROCESSED_DATA_DIR
from data_loader import load_yield_data, load_my_cc
from feature_engineering.feature_engineering_functions import process_feature_dict

"""
This script processes harmonizes multiple time series into a feature matrix customized for CNNs
"""

# INITIALIZATION  ####################################################################################################

# length of timeseries of remotes sensing and meteorological features
ts_length = 32

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


processed_features_ls = []
feature_name_ls = []
for feature_name, feature_path in feature_location_dict.items():
    print(feature_name, end="\r")
    feature_df = pd.read_csv(PROCESSED_DATA_DIR / feature_path, keep_default_na=False)

    # apply CC for each yield datapoint
    processed_feature_dict = process_feature_dict(yield_df=yield_df,
                                                  cc_df=cc_df,
                                                  feature_df=feature_df,
                                                  feature_name=feature_name,
                                                  length=ts_length,
                                                  start_before_sos=start_before_sos,
                                                  end_before_eos=end_before_eos)
    feature_matrix = pd.DataFrame(processed_feature_dict).values
    # standardize
    if feature_name != "preci":
        feature_matrix = feature_matrix - np.mean(feature_matrix)
    feature_matrix = feature_matrix / np.std(feature_matrix)

    # append to final array
    feature_name_ls.append(np.array(list(processed_feature_dict.keys())))
    processed_features_ls.append(feature_matrix)

# form save file
processed_timeserie_file = {"data": np.dstack(processed_features_ls),
                            "feature_name": np.stack(feature_name_ls)
                            }
id_columns = ["country", "adm1", "adm2", "adm", "harv_year"]
processed_timeserie_file["data_id"] = yield_df[id_columns]

with open(PROCESSED_DATA_DIR / f"features/processed_timeseries_features_{ts_length}_{start_before_sos}_{end_before_eos}.pkl", 'wb') as f:
    # Pickle using the highest protocol available.
    pickle.dump(processed_timeserie_file, f, pickle.HIGHEST_PROTOCOL)

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

