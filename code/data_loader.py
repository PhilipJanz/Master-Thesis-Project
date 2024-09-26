import pickle

import pandas as pd

from config import PROCESSED_DATA_DIR
from data_assembly import make_adm_column

# variables for filtering data
harv_year_range = (2001, 2022)

# variables for unit test (please alter if changes in the yield database occure)
# set 'None' to deactivate a test
yield_datapoints_after_filtering = 1902


def load_yield_data():
    yield_df = pd.read_csv(PROCESSED_DATA_DIR / "yield/processed_comb_yield.csv", keep_default_na=False)
    yield_df = yield_df[(yield_df.harv_year >= harv_year_range[0]) & (yield_df.harv_year <= harv_year_range[1])].reset_index(drop=True)
    yield_df = make_adm_column(yield_df)
    yield_df["adm1_"] = yield_df["country"] + "_" + yield_df["adm1"]
    # data unit test
    if yield_datapoints_after_filtering:
        assert len(yield_df) == yield_datapoints_after_filtering, f"After loading and filtering there are {len(yield_df)} yield datapoint instead of the expected {yield_datapoints_after_filtering}"
    return yield_df


def load_my_cc():
    return pd.read_csv(PROCESSED_DATA_DIR / f"crop calendar/my_crop_calendar.csv", keep_default_na=False)


def load_cluster_data():
    return pd.read_csv(PROCESSED_DATA_DIR / f"data clustering/cluster_data.csv", keep_default_na=False)


def load_soil_data():
    return pd.read_csv(PROCESSED_DATA_DIR / "soil/soil_property_region_level.csv", keep_default_na=False)


def load_soil_pca_data(pc_number):
    return pd.read_csv(PROCESSED_DATA_DIR / f"features/soil_pca_{pc_number}.csv", keep_default_na=False)


def load_processed_features(ts_length):
    with open(PROCESSED_DATA_DIR / f"features/processed_feature_df_dict_{ts_length}.pkl", 'rb') as file:
        processed_feature_df_dict = pickle.load(file)
    return processed_feature_df_dict
