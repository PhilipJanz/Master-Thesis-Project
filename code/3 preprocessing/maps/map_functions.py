import numpy as np
import rasterio
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from shapely.ops import unary_union
import pandas as pd
import geopandas as gpd
from rasterio.enums import Resampling
from rasterio.warp import reproject

from config import SOURCE_DATA_DIR, COUNTRY_COLORS, PROCESSED_DATA_DIR


def load_africa_map():
    return gpd.read_file(SOURCE_DATA_DIR / 'admin map/africa_countries/Africa_Countries.shp')


def load_aoi_map():
    return gpd.read_file(PROCESSED_DATA_DIR / "admin map/comb_map.shp")


def load_grid_data(path, lon_min=None, lat_max=None, lon_max=None, lat_min=None):
    """
    Load grid data from a raster file with optional geographical bounding box.

    Parameters:
    path (str): Path to the raster file.
    lon_min (float, optional): Minimum longitude of the bounding box.
    lat_max (float, optional): Maximum latitude of the bounding box.
    lon_max (float, optional): Maximum longitude of the bounding box.
    lat_min (float, optional): Minimum latitude of the bounding box.

    Returns:
    tuple:
        numpy.ndarray: 2D array of the raster data.
        affine.Affine: Affine transformation matrix for the raster.
        CRS: Coordinate reference system of the raster.
    """
    with rasterio.open(path) as src:
        if lon_min is not None and lat_max is not None and lon_max is not None and lat_min is not None:
            # Compute the window of the raster to be read based on the geographical bounds
            window = src.window(lon_min, lat_min, lon_max, lat_max)
            data = src.read(1, window=window)
            transform = src.window_transform(window)
        else:
            data = src.read(1)
            transform = src.transform

        crs = src.crs

    return data, transform, crs


def transform_grid_data(data, transform, crs, target_transform, target_crs):
    """
    Apply target transformation to the raster data.

    Parameters:
    data (numpy.ndarray): 2D array of the raster data.
    transform (affine.Affine): Affine transformation matrix for the raster.
    crs (CRS): Coordinate reference system of the raster.
    target_transform (Affine): Target affine transformation matrix to resample the raster data.
    target_crs (CRS): Coordinate reference system of the target.

    Returns:
        numpy.ndarray: Transformed 2D array of the raster data.
    """
    # Calculate the new dimensions based on the target transformation
    new_width = int(data.shape[1] * (transform.a / target_transform.a))
    new_height = int(data.shape[0] * (transform.e / target_transform.e))

    # Prepare an array to hold the reprojected data
    reprojected_data = np.empty((new_height, new_width), dtype=data.dtype)

    # Resample the data to the new transformation
    reproject(
        source=data,
        destination=reprojected_data,
        src_transform=transform,
        src_crs=crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear
    )

    return reprojected_data


def fill_nan_with_neighbors(image):
    """
    Fill nan values in the image with the mean of their nearest neighbors.

    Parameters:
    image (numpy.ndarray): 2D array with nan values to be filled.

    Returns:
    numpy.ndarray: 2D array with nan values filled.
    """
    filled_image = image.copy()
    nan_mask = np.isnan(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if nan_mask[i, j]:
                # Extract the 3x3 window around the pixel, handling edge cases
                neighbors = filled_image[max(0, i-1):min(image.shape[0], i+2), max(0, j-1):min(image.shape[1], j+2)]
                valid_neighbors = neighbors[~np.isnan(neighbors)]
                if valid_neighbors.size > 0:
                    filled_image[i, j] = np.nanmean(valid_neighbors)

    return filled_image


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


def merge_regions(gdf, column, regions, new_region_name):
    """
    Merges two regions in a GeoPandas DataFrame and combines their geometries.

    Parameters:
    gdf (GeoDataFrame): The GeoDataFrame containing the regions and geometries.
    column (str): The column to look for regions ('adm1' or 'adm2').
    regions (list): List of the names of the regions to be merged.
    new_region_name (str): The name of the new merged region.

    Returns:
    GeoDataFrame: Updated GeoDataFrame with the two regions merged.
    """
    # Filter rows for the two regions to merge
    mask = gdf[column].isin(regions)

    # Create new geometry by merging the geometries of the two regions
    new_geometry = unary_union(gdf.loc[mask, 'geometry'])

    # Create a new row for the merged region
    new_row = gdf[mask].iloc[-1].copy()
    new_row[column] = new_region_name
    new_row['geometry'] = new_geometry

    # Remove the original rows of the two regions
    gdf = gdf[~mask]

    # Append the new row
    gdf = pd.concat([gdf, pd.DataFrame([new_row])], ignore_index=True)

    return gdf


def plot_region_map(region_map, save_path=None):
    # load country-based map for better understanding for country borders
    africa_map = load_africa_map()

    fig, ax = plt.subplots(figsize=(10, 12))

    # Plot each country in the region_map and comb_map using colors from the dictionary
    for country, color in COUNTRY_COLORS.items():
        africa_map[africa_map.NAME == country].plot(color=color, edgecolor=color, linewidth=0.3, ax=ax, alpha=0.2)
        region_map[region_map.country == country].plot(color=color, edgecolor='white', linewidth=0.3, ax=ax, alpha=0.8)

    # Create a legend
    handles = [Patch(color=color, label=country, alpha=0.8) for country, color in COUNTRY_COLORS.items()]
    ax.legend(handles=handles, loc="upper left")

    # Annotate the map
    for idx, row in region_map.iterrows():
        label = row['adm2'] if row['adm2'] != "None" else row['adm1']
        # calculated center point for comparison and easier plotting
        centroid = row["geometry"].representative_point()
        plt.annotate(text=label, xy=(centroid.x, centroid.y),
                     horizontalalignment='center', fontsize=4, color="black")

    # Save the plot if a save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=800)

    plt.show()
