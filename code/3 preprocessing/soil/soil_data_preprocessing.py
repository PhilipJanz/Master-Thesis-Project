import geopandas as gpd
import numpy as np
import rasterio

from config import *
from crop_mask.crop_mask_functions import load_geoglam_crop_mask
from soil.soil_functions import load_soil_data

# load administrative boundaries with geographic information
adm_map = gpd.read_file(POCESSED_DATA_DIR / "admin map/comb_map.shp")

# load crop mask (cropped on AOI). Make sure the geographic boundaries fit your soil-data
crop_mask, target_transform, target_crs = load_geoglam_crop_mask(lon_min=22, lon_max=48, lat_min=-18, lat_max=15)

# load soil data with same transformation like the crop mask.
# ALERT!: Soil-data must be provided with the exact same geographic boundaries like given above
soil_nutrient_maps, soil_nutrients = load_soil_data(target_transform, target_crs,
                                                    target_width=crop_mask.shape[1],
                                                    target_height=crop_mask.shape[0])

geometry = adm_map.geometry[0]
multipolygon_mask = rasterio.features.rasterize(
    (geometry, 1),
    out_shape=(crop_mask.shape[0], crop_mask.shape[1]),
    transform=target_transform,
    fill=0,  # Areas outside the polygons will be filled with 0
    dtype=np.uint8,
    all_touched=True  # Consider all pixels touched by geometries
)