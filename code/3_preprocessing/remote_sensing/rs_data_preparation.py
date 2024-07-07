import numpy as np
import rasterio
import geopandas as gpd

from config import SOURCE_DATA_DIR, PROCESSED_DATA_DIR
from crop_mask.crop_mask_functions import weighted_avg_over_crop_mask
from maps.map_functions import load_aoi_map, load_grid_data
from remote_sensing.rs_functions import load_rs_data, custom_rolling_average

"""
This script prepares the remote sensing data from MODIS vegetation indices (VI)
We are process the NDVI and EVI for each region of interest and apply the GEOGLAM crop mask
Since there are some NaN values and outliers in the data (due to the occcurence of distorted data) 
we also appie an interpolation and smoother to generate smooth data without missing values.
"""

# load administrative boundaries (AOI) with geographic information
adm_map = load_aoi_map()

# load crop mask (cropped on AOI).
crop_mask, target_transform, target_crs = load_grid_data(path=SOURCE_DATA_DIR / "crop mask/GEOGLAM_Percent_Maize.tif",
                                                         lon_min=22, lon_max=48, lat_min=-18, lat_max=15)
# filter crop mask (10 ~ 0.1%)
crop_mask = np.where(crop_mask < 10, np.nan, crop_mask)

# path to MODIS remote sensing data
rs_path = SOURCE_DATA_DIR / "remote sensing/MODIS"

# load NDVI and EVI images
ndvi_image_list, ndvi_date_list = load_rs_data(rs_path, filter_filename="NDVI")
evi_image_list, evi_date_list = load_rs_data(rs_path, filter_filename="EVI")
assert evi_date_list == ndvi_date_list

# create ndvi and evi datasets
ndvi_data = adm_map.copy()
evi_data = adm_map.copy()

# make date columns to fill in a loop in the next step
for date in evi_date_list:
    ndvi_data[date] = np.nan
    evi_data[date] = np.nan

# iterate over regions and generate average VI values for each date
for ix, row in adm_map.iterrows():
    print(ix, row.drop("geometry").values)
    # rasterize the region polygon to harmonize it with crop mask and soil map
    region_mask = rasterio.features.rasterize(
        [row.geometry],
        out_shape=crop_mask.shape,
        transform=target_transform,
        fill=0,  # Areas outside the polygons will be filled with 0
        dtype=np.uint8,
        all_touched=True  # Consider all pixels touched by geometries
    )

    # This will set pixels outside the MultiPolygon to np.nan (assuming your data is in a float dtype)
    regional_crop_mask = np.where(region_mask == 1, crop_mask, np.nan)

    # iterate over dates and calculate the (weighted) average of NDVI and EVI for this region
    for evi_image, ndvi_image, date in zip(evi_image_list, ndvi_image_list, ndvi_date_list):
        ndvi_data.loc[ix, date] = weighted_avg_over_crop_mask(crop_mask=regional_crop_mask,
                                                              data_image=ndvi_image,
                                                              instance_name="NDVI",
                                                              region_name=str(adm_map.loc[ix, ["country", "adm1", "adm2"]].values).replace("'None' ", ""),
                                                              warn_spread_above=0.1)
        ndvi_data.loc[ix, date] = weighted_avg_over_crop_mask(crop_mask=regional_crop_mask,
                                                              data_image=evi_image,
                                                              instance_name="EVI",
                                                              region_name=str(adm_map.loc[ix, ["country", "adm1", "adm2"]].values).replace("'None' ", ""),
                                                              warn_spread_above=0.1)


### SMOOTHING ###
# apply rolling average and fill na values
smooth_ndvi_data = ndvi_data.copy()
smooth_evi_data = evi_data.copy()

for ix, row in ndvi_data.iterrows():
    smooth_ndvi_data.iloc[ix, 4:] = custom_rolling_average(row[4:].values, window=5)
for ix, row in evi_data.iterrows():
    smooth_evi_data.iloc[ix, 4:] = custom_rolling_average(row[4:].values, window=5)


#### SAVE #####
ndvi_data.drop("geometry", axis=1).to_csv(PROCESSED_DATA_DIR / "remote sensing/ndvi_regional_matrix.csv", index=False)
evi_data.drop("geometry", axis=1).to_csv(PROCESSED_DATA_DIR / "remote sensing/evi_regional_matrix.csv", index=False)

smooth_ndvi_data.drop("geometry", axis=1).to_csv(PROCESSED_DATA_DIR / "remote sensing/smooth_ndvi_regional_matrix.csv", index=False)
smooth_evi_data.drop("geometry", axis=1).to_csv(PROCESSED_DATA_DIR / "remote sensing/smooth_evi_regional_matrix.csv", index=False)
