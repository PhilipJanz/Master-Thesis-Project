import pandas as pd
from matplotlib import pyplot as plt

from config import PROCESSED_DATA_DIR, RESULTS_DATA_DIR
from data_assembly import make_adm_column
from maps.map_functions import load_aoi_map, load_africa_map


def plot_performance_map(performance_data, performance_column, result_filename):
    # load country-based map for better understanding for country borders
    africa_map = load_africa_map()
    # load map for areas of interest
    aoi_map = load_aoi_map()
    aoi_map = make_adm_column(aoi_map)

    # fusion with performance data
    geo_performance_data = pd.merge(aoi_map, performance_data, on="adm")

    # start plotting
    fig, ax = plt.subplots(figsize=(10, 12))

    # Plot country maps as background
    africa_map[africa_map.NAME.isin(geo_performance_data.country.unique())].plot(color="#e6e6e6", edgecolor="white", linewidth=2, ax=ax)

    # Plot NSE
    cmap = "RdYlGn"
    geo_performance_data.plot(column=performance_column, cmap=cmap, vmin=-1, vmax=1, edgecolor='white', linewidth=0.3, ax=ax, alpha=0.8)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    sm._A = []  # Dummy array for the ScalarMappable
    cbar = fig.colorbar(sm, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label(performance_column)

    # Annotate the map
    for idx, row in geo_performance_data.iterrows():
        label = row['adm2'] if row['adm2'] != "None" else row['adm1']
        # calculated center point for comparison and easier plotting
        centroid = row["geometry"].representative_point()
        plt.annotate(text=label, xy=(centroid.x, centroid.y),
                     horizontalalignment='center', fontsize=4, color="black")
    plt.title(result_filename)

    # Save the plot
    plt.savefig(RESULTS_DATA_DIR / f"yield_predictions/{result_filename}/plots/overall/nse_map.jpg", dpi=1200)

    plt.show()


def plot_map(df, column, title, cmap="viridis", cmap_range=None, save_path=None):
    # load country-based map for better understanding for country borders
    africa_map = load_africa_map()
    # load map for areas of interest
    aoi_map = load_aoi_map()
    aoi_map = make_adm_column(aoi_map)

    # fusion with performance data
    geo_performance_data = pd.merge(aoi_map, df, on="adm")

    # start plotting
    fig, ax = plt.subplots(figsize=(10, 12))

    # Plot country maps as background
    africa_map[africa_map.NAME.isin(geo_performance_data.country.unique())].plot(color="#e6e6e6", edgecolor="white", linewidth=2, ax=ax)

    # Plot NSE
    if cmap_range:
        geo_performance_data.plot(column=column, cmap=cmap, vmin=cmap_range[0], vmax=cmap_range[1], edgecolor='white', linewidth=0.3, ax=ax, alpha=0.8)
    else:
        geo_performance_data.plot(column=column, cmap=cmap, edgecolor='white', linewidth=0.3, ax=ax, alpha=0.8)

    # Add colorbar
    if cmap_range:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=cmap_range[0], vmax=cmap_range[1]))
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap)
    sm._A = []  # Dummy array for the ScalarMappable
    cbar = fig.colorbar(sm, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label(column)

    # Annotate the map
    for idx, row in geo_performance_data.iterrows():
        label = row['adm2'] if row['adm2'] != "None" else row['adm1']
        # calculated center point for comparison and easier plotting
        centroid = row["geometry"].representative_point()
        plt.annotate(text=label, xy=(centroid.x, centroid.y),
                     horizontalalignment='center', fontsize=4, color="black")
    plt.title(title)

    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=1200)

    plt.show()
