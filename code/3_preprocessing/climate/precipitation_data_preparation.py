import os
import pickle

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from matplotlib import pyplot as plt

from config import SOURCE_DATA_DIR, PROCESSED_DATA_DIR
from crop_mask.crop_mask_functions import weighted_avg_over_crop_mask, load_worldcereal_crop_mask
from maps.map_functions import load_aoi_map

"""
This script prepares the CHIRPS precipitation data
We are process the temperature for each region of interest and apply the WorldCereal crop mask
"""

# INITIALIZATION #######################

histogram_range = (0, 250)
bins = 32

"""
max_ls = []
min_ls = []
for i in range(365):
    # load the first file to get target-information to preprocess crop- and regional mask 
    with rasterio.open(data_path / image_filename_list[i]) as src:
        example_image = src.read(1)
        # plt.imshow(rs_image)
        # plt.show()
        target_transform = src.transform
        target_crs = src.crs
        min_ls.append(np.nanmin(example_image))
        max_ls.append(np.nanmax(example_image))

np.max(max_ls)
Out[24]: 307.1617711385091
np.min(min_ls)
Out[25]: 277.5318075815837
"""

# load administrative boundaries (AOI) with geographic information
adm_map = load_aoi_map()

# path to data
data_path = SOURCE_DATA_DIR / "climate/CHIRPS"

# list images
image_filename_list = os.listdir(data_path)
date_list = ["_".join([file_name[22:26], file_name[26: 28], file_name[28: 30]]) for file_name in image_filename_list]

# create preci datasets for average values
preci_data = adm_map.copy()
# make date columns to fill in a loop in the next step
for date in date_list:
    preci_data[date] = np.nan

# make histogram file based on a dictionary to fill in the loops ahead
historgram_dict = {}
for adm in adm_map.adm:
    # list to be filled with histograms
    historgram_dict[adm] = []

# load the first file to get target-information to preprocess crop- and regional mask
with rasterio.open(data_path / image_filename_list[0]) as src:
    example_image = src.read(1)
    # plt.imshow(rs_image)
    # plt.show()
    target_transform = src.transform
    target_crs = src.crs

# load crop mask (cropped on AOI).
crop_mask, cm_transform, cm_crs = load_worldcereal_crop_mask(target_transform=target_transform,
                                                             target_shape=example_image.shape,
                                                             binary=False)

# iterate over regions to preprocess regional-crop-masks
regional_crop_mask_dict = {}
for ix, row in adm_map.iterrows():
    # print(ix, row.drop("geometry").values)
    # rasterize the region polygon to harmonize it with crop mask and soil map
    region_mask = rasterio.features.rasterize(
        [row.geometry],
        out_shape=crop_mask.shape,
        transform=target_transform,
        all_touched=True  # Consider all pixels touched by geometries
    )

    # This will set pixels outside the MultiPolygon to np.nan (assuming your data is in a float dtype)
    regional_crop_mask = (region_mask == 1) * (crop_mask >= 1)
    regional_crop_mask_dict[row["adm"]] = regional_crop_mask

# iterate over each NDVI file (single image)
for date, image_filename in zip(date_list, image_filename_list):
    print(f"Processing: {date}", end="\r")

    # Load the remote sensing data
    with rasterio.open(data_path / image_filename) as src:
        image = src.read(1)
        # plt.imshow(rs_image)
        # plt.show()
        transform = src.transform
        crs = src.crs

    mean_value_ls = []
    for ix, (adm, regional_crop_mask) in enumerate(regional_crop_mask_dict.items()):
        masked_image = regional_crop_mask * image

        # extract list of values given preprocessed crop mask
        values = masked_image[regional_crop_mask]

        # make histogram:
        hist = np.histogram(values, bins=bins, range=histogram_range)[0]
        # normalize histogram and compress to int8
        hist = (hist / np.sum(hist) * 100)
        hist = hist.astype("float16")
        historgram_dict[adm].append(hist)

        # take averge
        mean_value_ls.append(np.nanmean(values))
    preci_data.loc[:, date] = mean_value_ls

# postprocess histogram dict into dict of dataframes given the date as columns
for adm, hist_ls in historgram_dict.items():
    historgram_dict[adm] = pd.DataFrame(np.vstack(hist_ls).T, columns=date_list)

#### SAVE #####

preci_data.drop("geometry", axis=1).to_csv(PROCESSED_DATA_DIR / "climate/preci_regional_matrix.csv", index=False)

with open(PROCESSED_DATA_DIR / "climate/precipitation_histograms.pickle", 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(historgram_dict, f, pickle.HIGHEST_PROTOCOL)
