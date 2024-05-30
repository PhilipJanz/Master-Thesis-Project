import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import rasterio

from config import *
from crop_mask.crop_mask_functions import load_geoglam_crop_mask
from soil.soil_functions import load_soil_data

# load administrative boundaries with geographic information
adm_map = gpd.read_file(PROCESSED_DATA_DIR / "admin map/comb_map.shp")

# load crop mask (cropped on AOI). Make sure the geographic boundaries fit your soil-data
crop_mask, target_transform, target_crs = load_geoglam_crop_mask(lon_min=22, lon_max=48, lat_min=-18, lat_max=15)

# load soil data with same transformation like the crop mask.
# ALERT!: Soil-data must be provided with the exact same geographic boundaries like given above
soil_nutrient_maps, soil_nutrients = load_soil_data(target_crs,
                                                    target_width=crop_mask.shape[1],
                                                    target_height=crop_mask.shape[0])

# make soil columns to fill in a loop in the next step
for soil_nutrient in soil_nutrients:
    adm_map[soil_nutrient] = np.nan
# we want to save the area in ha
# certainly this only applies for areas close to the equator but thats almost given
area_per_pixel = (0.05 / 360 * 400300) ** 2
adm_map["est_cropland_area"] = np.nan

# iterate over regions and generate average soil properties
for ix, row in adm_map.iterrows():
    # rasterize the region polygon to harmonize it with crop mask and soil map
    region_mask = rasterio.features.rasterize(
        [row.geometry],
        out_shape=(crop_mask.shape[0], crop_mask.shape[1]),
        transform=target_transform,
        fill=0,  # Areas outside the polygons will be filled with 0
        dtype=np.uint8,
        all_touched=True  # Consider all pixels touched by geometries
    )

    # This will set pixels outside the MultiPolygon to np.nan (assuming your data is in a float dtype)
    regional_crop_mask = np.where(region_mask == 1, crop_mask, np.nan)
    # estimate cropland area in that region
    adm_map.loc[ix, "est_cropland_area"] = np.nansum(regional_crop_mask) / 10000 * area_per_pixel
    # normalize the crop mask to make it easy to calculate a weighted average based on percentage cropland per pixel
    regional_crop_mask = regional_crop_mask / np.nansum(regional_crop_mask)

    # iterate over soil property and calculate the (weighted) average for this region
    for soil_nutrient_map, soil_nutrient in zip(soil_nutrient_maps, soil_nutrients):
        adm_map.loc[ix, soil_nutrient] = np.nansum(regional_crop_mask * soil_nutrient_map)

assert not np.any(adm_map.isna()), "A nan values was found. That should not occur. Check it out."

#### SAVE #####
adm_map.drop("geometry", axis=1).to_csv(PROCESSED_DATA_DIR / "soil/soil_property_region_level.csv", index=False)

#### PLOT #####
# List of column names you want to plot
columns_to_plot = ['clay', 'sand', 'silt', 'soc', 'nitrogen', 'phh2o']  # example column names
titles = ["Clay (g/Kg)",
          "Sand (g/Kg)",
          "Silt (g/Kg)",
          "Soil organic carbon (dg/Kg)",
          "Nitrogen (cg/Kg)",
          "pH water (pH * 10)"]

# Set up the plotting parameters
fig, axs = plt.subplots(int(len(columns_to_plot) / 2), 2, figsize=(12, 3*len(columns_to_plot)))
for ax, column, title in zip(axs.flatten(), columns_to_plot, titles):
    adm_map.plot(column=column, ax=ax, legend=True)
    ax.set_title(f'{column}')
    ax.set_axis_off()
    ax.set_title(title)
plt.tight_layout()
plt.savefig(PROCESSED_DATA_DIR / "soil/plots/soil_property_region_level.jpg", dpi=600)
#plt.show()

# plot cropland area
fig, ax = plt.subplots(figsize=(8, 8))
adm_map.plot(column='est_cropland_area', cmap="YlGn", legend=True, ax=ax)
plt.title("Estimated maize cropland area (ha)")
plt.savefig(PROCESSED_DATA_DIR / "soil/plots/estimated_cropland_area.jpg", dpi=600)
plt.show()

# plot results
fig, ax = plt.subplots(figsize=(8, 8))
adm_map.plot(column='elevation', cmap="RdYlGn_r", legend=True, ax=ax)
plt.title("Average elevation of cropland area")
plt.savefig(PROCESSED_DATA_DIR / "soil/plots/elevation.jpg", dpi=600)
plt.show()
