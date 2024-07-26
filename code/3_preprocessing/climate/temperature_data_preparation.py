import os
import time

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import geopandas as gpd

from climate.climate_functions import *
from config import SOURCE_DATA_DIR, PROCESSED_DATA_DIR
from crop_mask.crop_mask_functions import load_geoglam_crop_mask


"""
This script prepares temperature data (ERA5)
We are processing selected features for each region of interest and apply the GEOGLAM crop mask
"""

# load administrative boundaries (AOI) with geographic information
adm_map = gpd.read_file(PROCESSED_DATA_DIR / "admin map/comb_map.shp")

# path to temperature data
data_path = SOURCE_DATA_DIR / "climate/ERA5"
country_codes = os.listdir(data_path)
country_name_dict = {"MWI": "Malawi", "TZA": "Tanzania", "ZMB": "Zambia"}

# Initialization ########

# create a list of selected features for
selected_features = ["tasmin_median", "tas_median", "tasmax_median", "tasmin_belowP05", "tasmin_belowP01", "tasmax_aboveP95", "tasmax_aboveP99"]
# create a data-matrix for each of the features to be filled with values in the loops ahead
data_matrix = make_dekadal_data_matrix(adm_df=adm_map.drop("geometry", axis=1), first_year=1981, last_year=2022)
data_matrix_dict = {feature: data_matrix.copy() for feature in selected_features}


# Execution #########

# since the data was generated for each country with individual geographic boundaries, this is the first level loop
for country_code in country_codes:
    country = country_name_dict[country_code]
    country_data_path = data_path / country_code

    # load subfolders representing a dekade each
    dekade_folder_names = os.listdir(country_data_path)

    # run through the dekades
    for dekade_folder_name in dekade_folder_names:
        print(dekade_folder_name)
        dekade_country_data_path = country_data_path / dekade_folder_name
        dekade_end = str(dekade_country_data_path)[-5:]

        # load file names representing a feature each
        file_names = os.listdir(dekade_country_data_path)
        feature_names = ["_".join(name.split("_")[:2]) for name in file_names]

        # run through the single files and collect the features
        for file_name, feature_name in zip(file_names, feature_names):
            if feature_name not in selected_features:
                continue
            feature_file_path = dekade_country_data_path / file_name

            # load the file and get transform information for merging with crop masks...
            feature_data, target_transform = load_one_file(feature_file_path, get_transform=True)
            dims = feature_data.dims
            feature_data = feature_data[feature_name]

            # get the geographic boundaries
            lon_min, lon_max, lat_min, lat_max = calculate_geo_boundaries(dims=dims, transform=target_transform, round=4)

            # load crop mask (cropped on country ).
            crop_mask, target_transform_, _ = load_geoglam_crop_mask(lon_min=lon_min,
                                                                     lon_max=lon_max,
                                                                     lat_min=lat_min,
                                                                     lat_max=lat_max,
                                                                     resolution=0.25)

            # iterate over regions and generate average feature values for each year
            for ix, row in adm_map[adm_map.country == country].iterrows():
                # rasterize the region polygon to harmonize it with crop mask and soil map
                region_mask = rasterio.features.rasterize(
                    [row.geometry],
                    out_shape=crop_mask.shape,
                    transform=target_transform_,
                    fill=0,  # Areas outside the polygons will be filled with 0
                    dtype=np.uint8,
                    all_touched=True  # Consider all pixels touched by geometries
                )

                # This will set pixels outside the MultiPolygon to np.nan (assuming your data is in a float dtype)
                regional_crop_mask = np.where(region_mask == 1, crop_mask, np.nan)
                # normalize the crop mask to make it easy to calculate a weighted average based on percentage cropland per pixel
                assert np.nansum(regional_crop_mask) > 0
                normalized_regional_crop_mask = regional_crop_mask / np.nansum(regional_crop_mask)

                weighted_averages = calculate_weighted_average_on_xarray_images(images=feature_data,
                                                                                mask=normalized_regional_crop_mask)
                # write into data matix
                dates = [f"{str(t)[:4]}_{dekade_end}" for t in weighted_averages.time.values]
                data_matrix_dict[feature_name].loc[ix, dates] = weighted_averages.values

# save feature matrices
save_data_matrices(data_matrix_dict)

# take vacation, it was a long day!
