import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import rasterio.features

from config import SOURCE_DATA_DIR, PROCESSED_DATA_DIR
from crop_mask.crop_mask_functions import weighted_avg_over_crop_mask
from maps.map_functions import load_aoi_map, load_grid_data, transform_grid_data, fill_nan_with_neighbors

"""
Loads & processes ASAP phenology data.
Not further used in this repository
"""

# load administrative boundaries with geographic information
adm_map = load_aoi_map()

# load crop mask (cropped on AOI).
crop_mask, target_transform, target_crs = load_grid_data(path=SOURCE_DATA_DIR / "crop mask/GEOGLAM_Percent_Maize.tif",
                                                         lon_min=22, lon_max=48, lat_min=-18, lat_max=15)

# select cc
cc_columns = ["sos_1", "eos_1", "sos_2", "eos_2"]
file_names = ["phenos1_v03", "phenoe1_v03", "phenos2_v03", "phenoe2_v03"]
cc_data_images = []

# load crop calendar (cropped on AOI) and reshape them
for file_name in file_names:
    asap_cc, transform, crs = load_grid_data(path=SOURCE_DATA_DIR / f"crop calendar/ASAP/{file_name}.tif",
                                             lon_min=22, lon_max=48, lat_min=-18, lat_max=15)
    # make water bodies into nan so they dont distort the reshaping in the next step
    asap_cc = np.where(asap_cc > 200, np.nan, asap_cc)
    #asap_sos = fill_nan_with_neighbors(asap_sos)
    # reshape based on crop mask to match it
    asap_cc = transform_grid_data(data=asap_cc, transform=transform, crs=crs,
                                  target_transform=target_transform, target_crs=target_crs)
    assert asap_cc.shape == crop_mask.shape, f"Reshape went wrong: The shapes of cc image and crop mask differ: {asap_cc.shape}, {crop_mask.shape}"

    cc_data_images.append(asap_cc)

# make columns to fill in a loop in the next step
for cc_column in cc_columns:
    adm_map[cc_column] = np.nan

# iterate over regions and generate average soil properties
for ix, row in adm_map.iterrows():
    # rasterize the region polygon to harmonize it with crop mask and soil map
    region_mask = rasterio.features.rasterize(
        [row.geometry],
        out_shape=(crop_mask.shape[0], crop_mask.shape[1]),
        transform=target_transform,
        fill=0,  # Areas outside the polygons will be filled with 0
        dtype="int8",
        all_touched=True  # Consider all pixels touched by geometries
    )

    # This will set pixels outside the MultiPolygon to np.nan (assuming your data is in a float dtype)
    regional_crop_mask = np.where(region_mask == 1, crop_mask, np.nan)

    # iterate over soil property and calculate the (weighted) average for this region
    for cc_data_image, cc_column in zip(cc_data_images, cc_columns):
        # calculate the weighted average
        adm_map.loc[ix, cc_column] = weighted_avg_over_crop_mask(crop_mask=regional_crop_mask,
                                                                 data_image=cc_data_image,
                                                                 instance_name=cc_column,
                                                                 region_name=str(adm_map.loc[ix, ["country", "adm1", "adm2"]].values).replace("'None' ", ""),
                                                                 warn_spread_above=5)


# check if all rows got filled
#assert not np.any(adm_map.isna()), "A nan values was found. Check it out."

# POSTPROCESSING
# make crop calendar into day of year (instead of dekades)
adm_map[cc_columns] = np.round(adm_map[cc_columns] * 10)
# calculate growth time
adm_map["growth_time_1"] = adm_map["eos_1"] - adm_map["sos_1"]
adm_map["growth_time_2"] = adm_map["eos_2"] - adm_map["sos_2"]
# shrink cc in 1 - 365
adm_map[cc_columns] = adm_map[cc_columns] % 365 + 365 * (adm_map[cc_columns] % 365 == 0)


#### SAVE #####
adm_map.drop("geometry", axis=1).to_csv(PROCESSED_DATA_DIR / "crop calendar/cc_on_asap_phenology.csv", index=False)


assert False
# TODO

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
