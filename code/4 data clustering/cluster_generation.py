import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR
from cluster_functions import calculate_yield_correlations, kmean_elbow, kmean_cluster, make_profiles, \
    plot_cluster_profiles, plot_cluster_map, cluster_validation_plot
from maps.map_functions import load_aoi_map
from yield_.yield_functions import load_yield_data

"""
This script clusters all available regions in groups that will be trained on.
There is a set of different criteria that will be used to perform this clustering.
Finally a visualization should help to understand whats happening.
"""

# load administrative boundaries (AOI) with geographic information
adm_map = load_aoi_map()

# load RS data
ndvi_df = pd.read_csv(PROCESSED_DATA_DIR / "remote sensing/smooth_ndvi_regional_matrix.csv", keep_default_na=False)

# load precipitation data
preci_df = pd.read_csv(PROCESSED_DATA_DIR / "climate/pr_sum_regional_matrix.csv", keep_default_na=False)
preci_df = preci_df.replace({"": 1.}) # TODO delete when Ethiopia data is there!!!

# load crop calendar (cc)
cc_df = pd.read_csv(PROCESSED_DATA_DIR / "crop calendar/processed_crop_calendar.csv", keep_default_na=False)
cc_df = cc_df[~cc_df.season.isin(["Maize (Short rains)", "Maize (Maika/Bimodal)", "Maize (Vuli/Bimodal)"])].reset_index(drop=True)

#smooth_ndvi_data = pd.merge(smooth_ndvi_data, crop_calendar, on=["country", "adm1", "adm2"], how="left") # [["country", "adm1", "adm2", "ndvi_cluster"]]

# use curve fitting and cc information to generate functional profiles
season_ndvi_profile_mtx, season_preci_profile_mtx = make_profiles(cc_df=cc_df,
                                                                  ndvi_df=ndvi_df,
                                                                  preci_df=preci_df,
                                                                  profile_length=200,
                                                                  plot=False)
# make matrix of first derivate
season_diff_ndvi_profile_mtx = [np.diff(x) for x in season_ndvi_profile_mtx]

# make cluster based on smoothed NDVI values
kmean_elbow(data_mtx=season_ndvi_profile_mtx, max_k=21)
# choose k
k = 7
labels, _ = kmean_cluster(data_mtx=season_ndvi_profile_mtx, n_clusters=k)
cc_df["ndvi_cluster"] = labels

# make cluster based on smoothed NDVI values
kmean_elbow(data_mtx=season_diff_ndvi_profile_mtx, max_k=21)
# choose k
k = 7
labels, _ = kmean_cluster(data_mtx=season_diff_ndvi_profile_mtx, n_clusters=k)
cc_df["diff_ndvi_cluster"] = labels

# make cluster based on precipitation values
kmean_elbow(data_mtx=season_preci_profile_mtx, max_k=21)
# choose k
k = 7
labels, _ = kmean_cluster(data_mtx=season_preci_profile_mtx, n_clusters=k)
cc_df["preci_cluster"] = labels

plot_cluster_profiles(cluster_data=cc_df,
                      cluster_column="ndvi_cluster",
                      profile_data=season_ndvi_profile_mtx,
                      cluster_name="NDVI")
plot_cluster_map(cluster_data=cc_df,
                 cluster_column="ndvi_cluster",
                 cluster_name="NDVI")

plot_cluster_profiles(cluster_data=cc_df,
                      cluster_column="diff_ndvi_cluster",
                      profile_data=season_diff_ndvi_profile_mtx,
                      cluster_name="Diff-NDVI")
plot_cluster_map(cluster_data=cc_df,
                 cluster_column="diff_ndvi_cluster",
                 cluster_name="Diff-NDVI")

plot_cluster_profiles(cluster_data=cc_df,
                      cluster_column="preci_cluster",
                      profile_data=season_preci_profile_mtx,
                      cluster_name="Precipitation")
plot_cluster_map(cluster_data=cc_df,
                 cluster_column="preci_cluster",
                 cluster_name="Precipitation")

# Validation ######
# for validation we look at the yield distribution (overall and within-cluster)

# load yield data
yield_df = load_yield_data()

# generate yield correlation matrix
correlation_matrix = calculate_yield_correlations(yield_df)

# Visualize ######
clustered_yield_df = pd.merge(yield_df, cc_df, how="left", on=["country", "adm1", "adm2"])

cluster_validation_plot(clustered_yield_df=clustered_yield_df,
                        correlation_matrix=correlation_matrix,
                        cluster_columns=["ndvi_cluster", "diff_ndvi_cluster"],
                        cluster_names=["NDVI", "Diff-NDVI"])
