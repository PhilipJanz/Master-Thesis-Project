import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR
from cluster_functions import calculate_yield_correlations, kmean_elbow, kmean_cluster, make_profiles, \
    plot_cluster_profiles, plot_cluster_map, cluster_validation_plot, save_cluster_data, make_cc
from crop_calendar.crop_calendar_functions import load_my_cc
from maps.map_functions import load_aoi_map
from yield_.yield_functions import load_yield_data

"""
This script clusters all available regions in groups that will be trained on.
There is a set of different criteria that will be used to perform this clustering.
Finally a visualization should help to understand whats happening.
"""


# DATA LOADING ######

# load crop calendar
cc_df = load_my_cc()
cc_df = cc_df[~cc_df.country.isin(["Ethiopia", "Kenya"])]

# load RS data
ndvi_df = pd.read_csv(PROCESSED_DATA_DIR / "remote sensing/smooth_ndvi_regional_matrix.csv", keep_default_na=False)
ndvi_df = ndvi_df[~ndvi_df.country.isin(["Ethiopia", "Kenya"])]

# load precipitation data
preci_df = pd.read_csv(PROCESSED_DATA_DIR / "climate/pr_sum_regional_matrix.csv", keep_default_na=False)
preci_df = preci_df[~preci_df.country.isin(["Ethiopia", "Kenya"])]

# check check
assert np.all(cc_df.iloc[:, :3].values == ndvi_df.iloc[:, :3].values)
assert np.all(cc_df.iloc[:, :3].values == preci_df.iloc[:, :3].values)


# DATA PREPROCESSING ######

# make ndvi & preci profiles
ndvi_profile_mtx = make_profiles(ndvi_df)
preci_profile_mtx = make_profiles(preci_df)

# rearrange the profiles based on CC so that it starts with sos
for i, cc in cc_df.iterrows():
    ndvi_profile_mtx[i] = np.concatenate([ndvi_profile_mtx[i], ndvi_profile_mtx[i]])[(cc.sos - 1):][:365]
    preci_profile_mtx[i] = np.concatenate([preci_profile_mtx[i], preci_profile_mtx[i]])[(cc.sos - 1):][:365]

# clip the profiles to the average season length to only consider the profiles subset that describes the crop season
season_ndvi_profile_mtx = ndvi_profile_mtx[:, :250]
season_preci_profile_mtx = preci_profile_mtx[:, :250]

# make matrix of the first derivative
season_diff_ndvi_profile_mtx = [np.diff(x) for x in season_ndvi_profile_mtx]
season_diff_preci_profile_mtx = [np.diff(x) for x in season_preci_profile_mtx]


# CLUSTERING ######

# make cluster based on smoothed NDVI values
kmean_elbow(data_mtx=season_ndvi_profile_mtx, max_k=21)
# choose k
k = 20
labels, _ = kmean_cluster(data_mtx=season_ndvi_profile_mtx, n_clusters=k)
cc_df[f"ndvi-{k}_cluster"] = labels

# make cluster based on diff NDVI values
kmean_elbow(data_mtx=season_diff_ndvi_profile_mtx, max_k=21)
# choose k
k = 20
labels, _ = kmean_cluster(data_mtx=season_diff_ndvi_profile_mtx, n_clusters=k)
cc_df[f"diff_ndvi-{k}_cluster"] = labels

# make cluster based on precipitation values
kmean_elbow(data_mtx=season_preci_profile_mtx, max_k=21)
# choose k
k = 20
labels, _ = kmean_cluster(data_mtx=season_preci_profile_mtx, n_clusters=k)
cc_df[f"preci-{k}_cluster"] = labels

# make cluster based on diff- precipitation values
kmean_elbow(data_mtx=season_diff_preci_profile_mtx, max_k=21)
# choose k
k = 20
labels, _ = kmean_cluster(data_mtx=season_diff_preci_profile_mtx, n_clusters=k)
cc_df[f"diff_preci-{k}_cluster"] = labels


# PLOT #####

# NDVI
plot_cluster_profiles(cluster_data=cc_df.reset_index(),
                      cluster_column=f"ndvi-{k}_cluster",
                      profile_data=season_ndvi_profile_mtx,
                      cluster_name=f"NDVI-{k}")
plot_cluster_map(cluster_data=cc_df.reset_index(),
                 cluster_column=f"ndvi-{k}_cluster",
                 cluster_name=f"NDVI-{k}")

# NDVI first derivative
plot_cluster_profiles(cluster_data=cc_df.reset_index(),
                      cluster_column=f"diff_ndvi-{k}_cluster",
                      profile_data=season_diff_ndvi_profile_mtx,
                      cluster_name=f"Diff-NDVI-{k}")
plot_cluster_map(cluster_data=cc_df,
                 cluster_column=f"diff_ndvi-{k}_cluster",
                 cluster_name=f"Diff-NDVI-{k}")

# Precipitation
plot_cluster_profiles(cluster_data=cc_df,
                      cluster_column=f"preci-{k}_cluster",
                      profile_data=season_preci_profile_mtx,
                      cluster_name=f"Precipitation-{k}")
plot_cluster_map(cluster_data=cc_df,
                 cluster_column=f"preci-{k}_cluster",
                 cluster_name=f"Precipitation-{k}")

# Precipitation first derivative
plot_cluster_profiles(cluster_data=cc_df,
                      cluster_column=f"diff_preci-{k}_cluster",
                      profile_data=season_diff_preci_profile_mtx,
                      cluster_name=f"Diff-Precipitation-{k}")
plot_cluster_map(cluster_data=cc_df,
                 cluster_column=f"diff_preci-{k}_cluster",
                 cluster_name=f"Diff-Precipitation-{k}")

# save
save_cluster_data(cluster_df=cc_df.drop(columns=['sos', 'eos', 'season_length']))

# Validation ######
# for validation we look at the yield distribution (overall and within-cluster)

# load yield data
yield_df = load_yield_data()
yield_df = yield_df[~yield_df.country.isin(["Ethiopia", "Kenya"])]

# generate yield correlation matrix
correlation_matrix = calculate_yield_correlations(yield_df)

# fusion datasets
clustered_yield_df = pd.merge(yield_df, cc_df, how="left", on=["country", "adm1", "adm2"])

# make cluster validation plot for selected clusters
cluster_validation_plot(clustered_yield_df=clustered_yield_df,
                        correlation_matrix=correlation_matrix,
                        cluster_columns=["ndvi_cluster", "diff_ndvi_cluster", "preci_cluster", "diff_preci_cluster"],
                        cluster_names=["NDVI", "Diff-NDVI", "Precipitation", "Diff-Precipitation"])
