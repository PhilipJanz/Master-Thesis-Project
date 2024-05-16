import matplotlib.pyplot as plt
import numpy as np
import rasterio
import geopandas as gpd

from config import SOURCE_DATA_DIR, POCESSED_DATA_DIR
from crop_mask.crop_mask_functions import load_geoglam_crop_mask

"""
This script prepares the climate data from PIK consisting of precipitation (CHIRPS) and temperature data (ERA5)
We are processing selected features for each region of interest and apply the GEOGLAM crop mask
"""

# load administrative boundaries (AOI) with geographic information
adm_map = gpd.read_file(POCESSED_DATA_DIR / "admin map/comb_map.shp")

# load crop mask (cropped on AOI). Make sure the geographic boundaries fit your soil-data
crop_mask, target_transform, target_crs = load_geoglam_crop_mask(lon_min=22, lon_max=48, lat_min=-18, lat_max=15)

# path to MODIS remote sensing data
data_path = SOURCE_DATA_DIR / "climate"
