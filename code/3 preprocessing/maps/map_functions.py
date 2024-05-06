import geopandas as gpd
from shapely.ops import unary_union
import pandas as pd

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
