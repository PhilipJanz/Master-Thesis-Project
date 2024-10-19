import pickle

import pandas as pd

from config import PROCESSED_DATA_DIR, FEATURE_SELECTION_DIR

# variables for filtering data
harv_year_range = (2001, 2023)

# variables for unit test (please alter if changes in the yield database occure)
# set 'None' to deactivate a test
yield_datapoints_after_filtering = 1927


def make_adm_column(df):
    """
    Create common 'adm' column for faster matching of dataframes
    It units country, adm1 and adm2
    """
    df["adm"] = [str(x).replace("'", "").replace(" None", "") for x in df[["country", "adm1", "adm2"]].values]
    return df


def load_yield_data(benchmark_column=False):
    if benchmark_column:
        yield_df = pd.read_csv(PROCESSED_DATA_DIR / "yield/processed_comb_yield_and_benchmark.csv", keep_default_na=False)
    else:
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


def load_processed_features(feature_code):
    return pd.read_csv(PROCESSED_DATA_DIR / f"features/processed_designed_features_df_{feature_code}.csv", keep_default_na=False)


def load_timeseries_features(feature_code=None):
    with open(PROCESSED_DATA_DIR / f"features/processed_timeseries_features_{feature_code}.pkl", 'rb') as file:
        timeseries_feature_data = pickle.load(file)
        timeseries_feature_ndarray = timeseries_feature_data["data"]
        feature_names = timeseries_feature_data["feature_name"]
        data_id_df = timeseries_feature_data["data_id"]
    return timeseries_feature_ndarray, feature_names, data_id_df


def load_feature_selection(feature_selection_file=None):
    with open(FEATURE_SELECTION_DIR / f'{feature_selection_file}.pkl', 'rb') as file:
        feature_selection_dict = pickle.load(file)
    return feature_selection_dict
