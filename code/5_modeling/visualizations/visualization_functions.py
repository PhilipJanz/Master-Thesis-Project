import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config import PROCESSED_DATA_DIR, RESULTS_DATA_DIR, AOI, COUNTRY_COLORS
from data_assembly import make_adm_column
from maps.map_functions import load_aoi_map, load_africa_map
import seaborn as sns
import matplotlib.patches as mpatches


def plot_performance_map(performance_data, performance_column, result_filename, cmap="RdYlBu"):
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
    africa_map[africa_map.NAME.isin(geo_performance_data["country"].unique())].plot(color="#e6e6e6", edgecolor="white", linewidth=2, ax=ax)

    # Plot
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
    plt.savefig(RESULTS_DATA_DIR / f"yield_predictions/{result_filename}/plots/overall/{performance_column}_map.pdf", format="pdf")

    plt.close()


def plot_map(df, column, title=None, cmap="viridis", cmap_range=None, region_annotation=False, country_annotation=False, save_path=None):
    # load country-based map for better understanding for country borders
    africa_map = load_africa_map()
    # load map for areas of interest
    aoi_map = load_aoi_map()
    aoi_map = make_adm_column(aoi_map)

    # fusion with performance data
    geo_performance_data = pd.merge(aoi_map, df, on=list(aoi_map.columns.intersection(df.columns)))

    # start plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(AOI[0] - 0.5, AOI[2] + 0.5)
    ax.set_ylim(AOI[1] - 0.5, AOI[3] + 0.5)

    # Plot country maps as background
    africa_map.plot(color="lightgrey", linewidth=1, ax=ax, alpha=1)

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
    if region_annotation:
        for idx, row in geo_performance_data.iterrows():
            label = row['adm2'] if row['adm2'] != "None" else row['adm1']
            # calculated center point for comparison and easier plotting
            centroid = row["geometry"].representative_point()
            plt.annotate(text=label, xy=(centroid.x, centroid.y),
                         horizontalalignment='center', fontsize=4, color="black")

    africa_map.boundary.plot(edgecolor="white", linewidth=1, ax=ax, alpha=.8)

    if country_annotation:
        for idx, row in africa_map.iterrows():
            # calculated center point for comparison and easier plotting
            centroid = row["geometry"].representative_point()
            x_pos, y_pos = centroid.x, centroid.y
            # adjust position of text if needed
            if row.NAME == "Zambia":
                x_pos += 2.5
                y_pos -= 1.5
            elif row.NAME == "Congo DRC":
                x_pos += 3
                y_pos -= 3
            elif row.NAME == "Kenya":
                x_pos += 0.5
                y_pos -= 1.5
            elif row.NAME == "Mozambique":
                x_pos += 3
                y_pos += 6.1
            elif row.NAME == "Malawi":
                x_pos += 0
                y_pos += -0.5
            plt.annotate(text=row.NAME, xy=(x_pos, y_pos),
                         horizontalalignment='center', fontsize=20, fontweight='bold', color="white", alpha=1)

    if title:
        plt.title(title)

    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def ax_plot_map(fig, ax, df, column, cmap="viridis", cmap_range=None):
    # load country-based map for better understanding for country borders
    africa_map = load_africa_map()
    # load map for areas of interest
    aoi_map = load_aoi_map()
    aoi_map = make_adm_column(aoi_map)

    # fusion with performance data
    geo_performance_data = pd.merge(aoi_map, df, on=list(aoi_map.columns.intersection(df.columns)))

    ax.set_xlim(AOI[0] - 0.5, AOI[2] + 0.5)
    ax.set_ylim(AOI[1] - 0.5, AOI[3] + 0.5)

    # Plot country maps as background
    africa_map.plot(color="lightgrey", linewidth=1, ax=ax, alpha=1)

    # Plot NSE
    if cmap_range:
        geo_performance_data.plot(column=column, cmap=cmap, vmin=cmap_range[0], vmax=cmap_range[1], edgecolor='white',
                                  linewidth=0.3, ax=ax, alpha=0.8)
    else:
        geo_performance_data.plot(column=column, cmap=cmap, edgecolor='white', linewidth=0.3, ax=ax, alpha=0.8)

    # Add colorbar
    if cmap_range:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=cmap_range[0], vmax=cmap_range[1]))
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap)
    sm._A = []  # Dummy array for the ScalarMappable
    cbar = fig.colorbar(sm, ax=ax, fraction=0.036, pad=0.04)
    #cbar.set_label(column)

    africa_map.boundary.plot(edgecolor="white", linewidth=1, ax=ax, alpha=.8)

    return ax


def plot_performance_box_x_map(performance_df, metrics, save_path=None):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 9), width_ratios=[1, 3])
    (ax1, ax2), (ax3, ax4) = axs

    # Prepare boxplots
    r2_data = {}
    brr2_data = {}
    colors = []
    for country, color in COUNTRY_COLORS.items():
        country_df = performance_df[performance_df["country"] == country]
        r2_data[country] = country_df["r2"].values
        brr2_data[country] = country_df["brr2"].values
        colors.append(color)

    medianprops = {"linewidth": 1.7, "color": "black"}  # Adjust linewidth as needed
    sns.boxplot(
        data=r2_data,
        ax=ax1,
        palette=colors,
        medianprops=medianprops
    )
    sns.boxplot(
        data=brr2_data,
        ax=ax3,
        palette=colors,
        medianprops=medianprops
    )

    # Add dashed horizontal line at y=0
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1.2)
    ax3.axhline(0, color='gray', linestyle='--', linewidth=1.2)

    # Set labels and remove x-tick labels
    ax1.set_xticklabels([])
    ax3.set_xticklabels([])
    ax1.set_ylabel(r"$\mathrm{R}^2$", rotation=0)
    ax3.set_ylabel(r"$\mathrm{BR-R}^2$", rotation=0, labelpad=20) # , labelpad=50, ha='center', va='center'

    # Custom legend below the plot
    legend_patches = []
    for country, color in COUNTRY_COLORS.items():
        legend_patches.append(mpatches.Patch(color=color, label=country))
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, bbox_to_anchor=(.5, -0.02))

    # Plot maps
    ax_plot_map(fig, ax=ax2, df=performance_df, column="r2", cmap="RdYlBu", cmap_range=(-1, 1))
    ax_plot_map(fig, ax=ax4, df=performance_df, column="brr2", cmap="RdYlGn", cmap_range=(-1, 1))
    ax2.set_rasterized(True)
    ax4.set_rasterized(True)
    ax2.text(0.02, 0.98, r"$\mathrm{R}^2$: " + str(np.round(metrics[0], 2)), transform=ax2.transAxes,
         fontsize=12, verticalalignment='top', horizontalalignment='left',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    ax4.text(0.02, 0.98, r"$\mathrm{BR-R}^2$: " + str(np.round(metrics[1], 2)), transform=ax4.transAxes,
         fontsize=12, verticalalignment='top', horizontalalignment='left',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    # Remove coordinates
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax4.set_xticks([])
    ax4.set_yticks([])

    # Add space on the left for annotations
    #plt.subplots_adjust(left=0.1)

    plt.tight_layout()
    # Show and save the plot
    if save_path:
        plt.savefig(save_path, format="pdf", dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_corr_matrix(corr_df, title, save_path=None):
    plt.figure(figsize=(13, 12))  # Set the size of the plot
    sns.set(style='white')  # Set the style to white
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    heatmap = sns.heatmap(corr_df, mask=mask, annot=True, cmap='coolwarm', fmt=".2f",
                          vmax=1.0, vmin=-1.0)
    # Add titles and labels
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    # Show and save the plot
    if save_path:
        plt.savefig(save_path, dpi=600)
        plt.close()
    else:
        plt.show()
