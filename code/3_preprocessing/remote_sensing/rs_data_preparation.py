import os
import pickle

import numpy as np
import rasterio
import geopandas as gpd
from matplotlib import pyplot as plt

from config import SOURCE_DATA_DIR, PROCESSED_DATA_DIR
from crop_mask.crop_mask_functions import weighted_avg_over_crop_mask, load_worldcereal_crop_mask
from maps.map_functions import load_aoi_map
from remote_sensing.rs_functions import custom_rolling_average, fill_missing_values

"""
This script prepares the remote sensing data from MODIS vegetation indices (VI)
We are process the NDVI for each region of interest and apply the WorldCereal crop mask
Since there are some NaN values and outliers in the data (due to the occcurence of distorted data) 
we also appie an interpolation and smoother to generate smooth data without missing values.
"""

# load administrative boundaries (AOI) with geographic information
adm_map = load_aoi_map()

# load crop mask (cropped on AOI).
crop_mask, target_transform, target_crs = load_worldcereal_crop_mask()

# path to MODIS remote sensing data
rs_path = SOURCE_DATA_DIR / "remote sensing/MODIS_NDVI"

# list NDVI images
ndvi_image_filename_list = os.listdir(rs_path)
date_list = [file_name[18:28] for file_name in ndvi_image_filename_list]

# create ndvi datasets for average values
ndvi_data = adm_map.copy()
quality_data = adm_map.copy()

# make date columns to fill in a loop in the next step
for date in date_list:
    ndvi_data[date] = np.nan
    quality_data[date] = np.nan

# make histogram file based on a dictionary to fill in the loops ahead
historgram_dict = {}
for adm in adm_map.adm:
    # list to be filled with histograms
    historgram_dict[adm] = []

# iterate over regions to preprocess regional-crop-masks
regional_crop_mask_dict = {}
for ix, row in adm_map.iterrows():
    #print(ix, row.drop("geometry").values)
    # rasterize the region polygon to harmonize it with crop mask and soil map
    region_mask = rasterio.features.rasterize(
        [row.geometry],
        out_shape=crop_mask.shape,
        transform=target_transform,
        all_touched=True  # Consider all pixels touched by geometries
    )

    # This will set pixels outside the MultiPolygon to np.nan (assuming your data is in a float dtype)
    regional_crop_mask_dict[row["adm"]] = (region_mask == 1) * crop_mask


# iterate over each NDVI file (single image)
for date, ndvi_image_filename in zip(date_list, ndvi_image_filename_list):
    print(f"Processing: {date}", end="\r")

    # Load the remote sensing data
    with rasterio.open(rs_path / ndvi_image_filename) as rs_src:
        rs_image = rs_src.read(1) / 100
        # plt.imshow(rs_image)
        # plt.show()
        transform = rs_src.transform
        crs = rs_src.crs

    for ix, (adm, regional_crop_mask) in enumerate(regional_crop_mask_dict.items()):
        masked_rs_image = regional_crop_mask * rs_image
        qualitative_pixel = masked_rs_image > 0

        quality_percentage = np.sum(qualitative_pixel) / np.sum(regional_crop_mask)
        quality_data.loc[ix, date] = quality_percentage

        # check the availability of qualitative pixels
        # If less than a certain ratio of the theoretical available pixel we safe None values to be filled later
        if quality_percentage < 0.1:
            historgram_dict[adm].append(None)
            ndvi_data.loc[ix, date] = np.nan
            continue

        # extract list of values from qualitative pixels
        rs_values = masked_rs_image[qualitative_pixel]

        # make histogram:
        hist = np.histogram(rs_values, bins=32, range=(0, 100))[0]
        # normalize histogram and compress to int8
        hist = (hist / np.sum(hist) * 100)
        hist = hist.astype("float16")
        historgram_dict[adm].append(hist)

        # take averge
        ndvi_data.loc[ix, date] = np.mean(rs_values)

### FILLING MISSING VALUES ####

cleaned_ndvi_data = ndvi_data.copy()
for i in range(len(ndvi_data)):
    _, cleaned_ndvi_data.iloc[i, 5:] = fill_missing_values(data=ndvi_data.values[i, 5:], periode_length_guess=365/16, N=4)


plt.plot(ndvi_data.values[i, 5:], 'o-', label="Qualitative NDVI-values")
plt.plot(cleaned_ndvi_data.iloc[i, 5:], linestyle="--", alpha=.85, label="Sine Approximation")
plt.ylabel("NDVI")
plt.legend(loc="upper right")
plt.title("Filling missing values in NDVI curves")
plt.show()

### VISUALIZE DATA QUALITY #####
# example from Tanzania
tanz_quality_data = quality_data.loc[quality_data.country == "Tanzania"].iloc[:, 5:]
zamb_quality_data = quality_data.loc[quality_data.country == "Zambia"].iloc[:, 5:]
mala_quality_data = quality_data.loc[quality_data.country == "Malawi"].iloc[:, 5:]
month_ls = [int(col[5:7]) for col in tanz_quality_data.columns]
tanz_monthly_average_quality = []
zamb_monthly_average_quality = []
mala_monthly_average_quality = []
for i in range(12):
    tanz_monthly_average_quality.append(np.mean(tanz_quality_data.loc[:, (i + 1) == np.array(month_ls)]))
    zamb_monthly_average_quality.append(np.mean(zamb_quality_data.loc[:, (i + 1) == np.array(month_ls)]))
    mala_monthly_average_quality.append(np.mean(mala_quality_data.loc[:, (i + 1) == np.array(month_ls)]))


# Labels for months
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Create a grouped bar plot
width = 0.25  # width of the bars
x = np.arange(len(months))  # the label locations

plt.figure(figsize=(12, 6))
plt.bar(x - width, tanz_monthly_average_quality, width, label='Tanzania')
plt.bar(x, zamb_monthly_average_quality, width, label='Zambia')
plt.bar(x + width, mala_monthly_average_quality, width, label='Malawi')

# Add titles and labels
plt.title('Monthly Average Quality Ratio of MODIS Images', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Quality Ratio', fontsize=12)
plt.xticks(x, months)

# Display grid and legend
plt.grid(axis='y')
plt.legend()

"""
### SMOOTHING ###
# apply rolling average and fill na values
smooth_ndvi_data = ndvi_data.copy()

for ix, row in ndvi_data.iterrows():
    smooth_ndvi_data.iloc[ix, 4:] = custom_rolling_average(row[4:].values, window=5)
"""

#### SAVE #####
ndvi_data.drop("geometry", axis=1).to_csv(PROCESSED_DATA_DIR / "remote sensing/ndvi_regional_matrix.csv", index=False)
cleaned_ndvi_data.drop("geometry", axis=1).to_csv(PROCESSED_DATA_DIR / "remote sensing/cleaned_ndvi_regional_matrix.csv", index=False)
quality_data.drop("geometry", axis=1).to_csv(PROCESSED_DATA_DIR / "remote sensing/quality_regional_matrix.csv", index=False)


with open(PROCESSED_DATA_DIR / "remote sensing/ndvi_histograms.pickle", 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(historgram_dict, f, pickle.HIGHEST_PROTOCOL)


#smooth_ndvi_data.drop("geometry", axis=1).to_csv(PROCESSED_DATA_DIR / "remote sensing/smooth_ndvi_regional_matrix.csv", index=False)
