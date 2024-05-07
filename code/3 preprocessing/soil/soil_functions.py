import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import geopandas as gpd
import shapely
import rasterio
from rasterio.plot import show
import rasterio.features
from rasterio.enums import Resampling

from config import *


def load_soil_data(target_crs, target_width, target_height):
    """
    Loads SoilGrid data and reshapes it in the right format. This function will source any file in the soilgrids folder.
    Make sure they are in the fitting format.
    :param target_crs: used to make sure the red data is having the right crs (should be epsg4326)
    :param target_width: width  the soil data should have
    :param target_height: height the soil data should have
    :return: list of soil maps | list of feature of the soil maps
    """
    soilgrids_path = SOURCE_DATA_DIR / "soil/SoilGrids"
    soilgrids_filenames = os.listdir(soilgrids_path)

    soil_nutrients = []
    soil_nutrient_maps = []
    for soilgrids_filename in soilgrids_filenames:
        # extract soil nutrient
        soil_nutrients.append(soilgrids_filename.split("_")[0])
        # loas data and reshape
        with rasterio.open(soilgrids_path / soilgrids_filename) as soil_src:
            soil_nutrient_map = soil_src.read(
                1,
                out_shape=(target_height, target_width),
                resampling=Resampling.bilinear  # You can choose other resampling methods
            )
            assert soil_src.crs == target_crs, f"The soil data {soilgrids_filename} does not fulfill the target crs: {target_crs}. This is required."
            soil_nutrient_maps.append(soil_nutrient_map)

    return soil_nutrient_maps, soil_nutrients
