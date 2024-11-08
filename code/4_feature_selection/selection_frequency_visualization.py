import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

from config import FEATURE_SELECTION_DIR
from data_loader import load_feature_selection, load_processed_features

"""
Visualization of feature selection frequency using multiple grid plots (t= {1, 3, 6}).

"""

# LOAD FEATURE SELECTION & PROCESS SELECTION FREQUENCY ###########################

# investigate different t -> number of time windows
t_ls = [1, 3, 6]
feature_frequency_grid_ls = []
for t in t_ls:
    # feature selection file (processed by select_features())
    feature_id = f"{t}_60_60"
    fs_file = f"feature_selection_dir_country_vif5_"

    # load and process features
    feature_selection_dict = load_feature_selection(feature_selection_file=fs_file)

    # load and feature list
    processed_feature_df = load_processed_features(feature_id)
    feature_ls = processed_feature_df.columns[:-5]

    if t == 1:
        feature_names = list(feature_ls)
        # make feature dict to be filled
        feature_freqency_dict = {}
        for feature in feature_names:
            feature_freqency_dict[feature] = [0]

        count = 0
        for region, regional_fs_dict in feature_selection_dict.items():
            #if "Zambia" not in region:
            #    continue
            for year, fs_ls in regional_fs_dict.items():
                for feature in fs_ls:
                    feature_freqency_dict[feature][0] += 1
                count += 1
    else:
        feature_names = np.unique([feature[:-2] for feature in feature_ls])
        # make feature dict to be filled
        feature_freqency_dict = {}
        for feature in feature_names:
            feature_freqency_dict[feature] = np.zeros(t)

        count = 0
        for region, regional_fs_dict in feature_selection_dict.items():
            #if "Zambia" not in region:
            #    continue
            for year, fs_ls in regional_fs_dict.items():
                for feature in fs_ls:
                    feature_name = feature[:-2]
                    feature_idx = int(feature[-1]) - 1
                    feature_freqency_dict[feature_name][feature_idx] += 1
                count += 1

    feature_frequency_df = pd.DataFrame(feature_freqency_dict)
    feature_frequency_df /= count


    # Define clusters (Example: 3 clusters)
    clusters = {
        'ndvi': ['ndvi_max', 'ndvi_mean', 'ndvi_min', "svi"],
        'preci': ['preci_max', 'preci_max-cdd', 'preci_sum', 'spi1', 'spi6'],
        'temp': ['temp_max', 'temp_mean', 'temp_min', 'sti']
    }

    # Reorder the matrix based on clusters
    new_order = [idx for cluster in clusters.values() for idx in cluster]
    feature_frequency_df_clustered = feature_frequency_df[new_order]

    feature_frequency_grid_ls.append(feature_frequency_df_clustered)


# PLOT #####################

# Create cluster boundaries
cluster_sizes = [len(indices) for indices in clusters.values()]
cluster_boundaries = np.cumsum(cluster_sizes)

# Define feature names
feature_names = new_order

# Create a figure with two subplots (stacked vertically)
fig = plt.figure(figsize=(10, 6))
# Create a grid for the two heatmaps and one colorbar
gs = gridspec.GridSpec(3, 2, width_ratios=[1, 0.05], height_ratios=[1, 3, 6])  # The last column is for the shared colorbar

# Create axes for the heatmaps
ax1 = fig.add_subplot(gs[0, 0])  # First heatmap
ax2 = fig.add_subplot(gs[1, 0])  # Second heatmap
ax3 = fig.add_subplot(gs[2, 0])  # Second heatmap

# Create axis for the shared colorbar
cbar_ax = fig.add_subplot(gs[:, 1])  # Shared colorbar (on the side of both heatmaps)

# Plot the first heatmap
sns.heatmap(
    feature_frequency_grid_ls[0],  # Data for country-level modeling
    cmap='Blues',
    vmin=0,
    vmax=1,
    yticklabels=feature_frequency_grid_ls[0].index + 1,
    ax=ax1,
    cbar=False  # Disable the individual colorbar
)
#ax1.set_ylabel('Time Index')
ax1.set_title("For single time windows")
ax1.set_xticks([])
for boundary in cluster_boundaries[:-1]:
    ax1.axvline(boundary, color='grey', linewidth=2)

# Plot the second heatmap (assuming the same data for illustration)
sns.heatmap(
    feature_frequency_grid_ls[1],  # Data for admin-level modeling
    cmap='Blues',
    vmin=0,
    vmax=1,
    yticklabels=feature_frequency_grid_ls[1].index + 1,
    ax=ax2,
    cbar=False  # Disable the individual colorbar
)
#ax2.set_ylabel('Time Index')
ax2.set_title("For 3 time windows")
ax2.set_xticks([])
for boundary in cluster_boundaries[:-1]:
    ax2.axvline(boundary, color='grey', linewidth=2)

# Plot the second heatmap (assuming the same data for illustration)
sns.heatmap(
    feature_frequency_grid_ls[2],  # Data for admin-level modeling
    cmap='Blues',
    vmin=0,
    vmax=1,
    yticklabels=feature_frequency_grid_ls[2].index + 1,
    ax=ax3,
    cbar=False  # Disable the individual colorbar
)
#ax3.set_ylabel('Time Index')
ax3.set_title("For 6 time windows (monthly features)")
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
for boundary in cluster_boundaries[:-1]:
    ax3.axvline(boundary, color='grey', linewidth=2)

# Create a shared colorbar for both heatmaps
fig.colorbar(ax1.collections[0], cax=cbar_ax, label='Selection Frequency', shrink=0.75)

# Adjust layout
plt.tight_layout()
plt.savefig(FEATURE_SELECTION_DIR / f"plots/feature_selection_frequency.pdf", format="pdf")
plt.show()
