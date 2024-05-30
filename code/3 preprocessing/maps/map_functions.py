from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from shapely.ops import unary_union
import pandas as pd
import geopandas as gpd

from config import SOURCE_DATA_DIR, COUNTRY_COLORS, PROCESSED_DATA_DIR


def load_africa_map():
    return gpd.read_file(SOURCE_DATA_DIR / 'admin map/africa_countries/Africa_Countries.shp')


def load_aoi_map():
    return gpd.read_file(PROCESSED_DATA_DIR / "admin map/comb_map.shp")


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
