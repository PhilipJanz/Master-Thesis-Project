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

def load_soil_data(target_transform, target_crs, target_width, target_height):
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
            soil_meta = soil_src.meta

            # Update the metadata for the resampled soil-nutrient map
            soil_meta.update({
                "driver": "GTiff",
                "height": target_height,
                "width": target_width,
                "transform": target_transform,
                "crs": target_crs
            })
            soil_nutrient_maps.append(soil_nutrient_map)

    return soil_nutrient_maps, soil_nutrients