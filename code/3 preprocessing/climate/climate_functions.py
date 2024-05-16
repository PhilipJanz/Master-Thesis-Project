import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xarray as xr

from config import SOURCE_DATA_DIR

"""
This script is a small cosy libray to load and allocate the climate data provided by PIK with following structure;
Under the 'climate' folder you find folders for each country, eg. MWI (Malawi)
The next level of folders specify the dekade and origin of the data, eg. ERA5_Malawi_MWI_04_21_04_30 (ERA5 data and 21.-30. April)
Inside there are files where each contains maps of one feature for all possible years, eg. tas_mean_ERA5_historical_1981-2022_MWI_04_21_04_30_Malawi.nc (mean temperature for the years 1981 until 2022)
Data-dimensions: {'time': 42, 'bnds': 2, 'longitude': 15, 'latitude': 35}
"""

file_path = SOURCE_DATA_DIR / "climate/MWI/CHIRPS-0.05_Malawi_MWI_12_21_12_30/pr_sum_CHIRPS-0.05_historical_1981-2022_MWI_12_21_12_30_Malawi.nc"
file_path = SOURCE_DATA_DIR / "climate/MWI/ERA5_Malawi_MWI_12_21_12_30/tas_mean_ERA5_historical_1981-2022_MWI_12_21_12_30_Malawi.nc"

def load_one_file(file_path):
    data = xr.open_dataset(file_path)
    data = data.rio.write_crs("EPSG:4326")
