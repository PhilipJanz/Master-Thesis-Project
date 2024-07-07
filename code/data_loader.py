import pandas as pd

from config import PROCESSED_DATA_DIR


def load_yield_data():
    return pd.read_csv(PROCESSED_DATA_DIR / "yield/processed_comb_yield.csv", keep_default_na=False)


def load_my_cc():
    return pd.read_csv(PROCESSED_DATA_DIR / f"crop calendar/my_crop_calendar.csv", keep_default_na=False)


def load_cluster_data():
    return pd.read_csv(PROCESSED_DATA_DIR / f"data clustering/cluster_data.csv", keep_default_na=False)


def load_soil_data():
    return pd.read_csv(PROCESSED_DATA_DIR / "soil/soil_property_region_level.csv", keep_default_na=False)
