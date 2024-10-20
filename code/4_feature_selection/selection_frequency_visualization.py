import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import FEATURE_SELECTION_DIR
from data_loader import load_feature_selection, load_processed_features

# processed feature code (feature len _ days before sos _ days before eos)
feature_file = "6_60_60"
data_split = "country"

# feature selection file
vif_threshold = 5
fs_file = f"feature_selection_dir_{data_split}_vif{vif_threshold}_{feature_file}"

# load and process features
feature_selection_dict = load_feature_selection(feature_selection_file=fs_file)

# load and feature list
processed_feature_df = load_processed_features(feature_file)
feature_ls = processed_feature_df.columns[:-5]
feature_names = np.unique([feature[:-2] for feature in feature_ls])

# make feature dict to be filled
feature_freqency_dict = {}
for feature in feature_names:
    feature_freqency_dict[feature] = np.zeros(int(feature_file[0]))

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

# Create cluster boundaries
cluster_sizes = [len(indices) for indices in clusters.values()]
cluster_boundaries = np.cumsum(cluster_sizes)

# Define feature names
feature_names = new_order

# Plot the heatmap
plt.figure(figsize=(10, 4))
ax = sns.heatmap(
    feature_frequency_df_clustered,
    cmap='Blues',
    vmin=0,
    vmax=.7,#np.max(feature_frequency_df),
    #xticklabels=feature_names,
    yticklabels=feature_frequency_df_clustered.index + 1,
    cbar_kws={'label': 'Selection Frequency'}
)
plt.ylabel('Time Index')

# Add lines to separate clusters
for boundary in cluster_boundaries[:-1]:
    ax.axvline(boundary, color='grey', linewidth=2)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


######################

# Create a figure with two subplots (stacked vertically)
fig = plt.figure(figsize=(10, 6))
# Create a grid for the two heatmaps and one colorbar
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.05])  # The last column is for the shared colorbar

# Create axes for the heatmaps
ax1 = fig.add_subplot(gs[0, 0])  # First heatmap
ax2 = fig.add_subplot(gs[1, 0])  # Second heatmap

# Create axis for the shared colorbar
cbar_ax = fig.add_subplot(gs[:, 1])  # Shared colorbar (on the side of both heatmaps)

# Plot the first heatmap
sns.heatmap(
    feature_frequency_df_clustered,  # Data for country-level modeling
    cmap='Blues',
    vmin=0,
    vmax=0.7,
    yticklabels=feature_frequency_df_clustered.index + 1,
    ax=ax1,
    cbar=False  # Disable the individual colorbar
)
ax1.set_ylabel('Time Index')
ax1.set_title("For country-level modeling")
ax1.set_xticks([])
for boundary in cluster_boundaries[:-1]:
    ax1.axvline(boundary, color='grey', linewidth=2)

# Plot the second heatmap (assuming the same data for illustration)
sns.heatmap(
    feature_frequency_df_clustered_adm,  # Data for admin-level modeling
    cmap='Blues',
    vmin=0,
    vmax=0.7,
    yticklabels=feature_frequency_df_clustered_adm.index + 1,
    ax=ax2,
    cbar=False  # Disable the individual colorbar
)
ax2.set_ylabel('Time Index')
ax2.set_title("For admin-level modeling")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
for boundary in cluster_boundaries[:-1]:
    ax2.axvline(boundary, color='grey', linewidth=2)

# Create a shared colorbar for both heatmaps
fig.colorbar(ax1.collections[0], cax=cbar_ax, label='Selection Frequency', shrink=0.75)

# Adjust layout
plt.tight_layout()
plt.savefig(FEATURE_SELECTION_DIR / f"plots/feature_selection_frequency.pdf", format="pdf")
plt.show()
