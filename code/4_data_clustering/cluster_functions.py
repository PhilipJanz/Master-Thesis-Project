import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns

from crop_calendar.crop_calendar_functions import detect_seasons
from crop_calendar.profile_generation import make_profiles
from maps.map_functions import load_africa_map, load_aoi_map


OMP_NUM_THREADS = 1

from config import PROCESSED_DATA_DIR, COUNTRY_COLORS


def kmean_cluster(data_mtx, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data_mtx)
    return kmeans.labels_, kmeans.inertia_


def kmean_elbow(data_mtx, max_k=10):
    # Let's find the optimal number of clusters using within-cluster variance (Elbow method)
    wcss_ls = []  # List to hold the within-cluster sum of squares
    k_values = range(1, max_k)  # We will test k from 1 to 10

    for k in k_values:
        _, wcss = kmean_cluster(data_mtx, n_clusters=k)
        wcss_ls.append(wcss)

    # Plot the within-cluster variance for each k to find the elbow
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, wcss_ls / np.max(wcss_ls), marker='o')
    plt.title('Elbow method for optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('normalized within-cluster Variance (WCSS)')
    plt.xticks(k_values)
    plt.grid(visible=True)
    #plt.savefig("Data Clustering/elbow_curve.jpg", dpi=300)
    plt.show()


def calculate_yield_correlations(df):
    # Group the data by the specified columns
    grouped = df.groupby(['country', 'adm1', 'adm2', 'season'])

    # Create a list of unique group identifiers
    unique_groups = list(grouped.groups.keys())

    # Initialize the correlation matrix with NaNs
    correlation_matrix = pd.DataFrame(np.nan, index=unique_groups, columns=unique_groups)

    # Calculate the correlation between each pair of groups
    for i, group1 in enumerate(unique_groups):
        for j, group2 in enumerate(unique_groups):
            if i <= j:  # Avoid redundant calculations
                yield1 = grouped.get_group(group1).set_index('harv_year')['yield']
                yield2 = grouped.get_group(group2).set_index('harv_year')['yield']

                # Find the common years between the two groups
                common_years = yield1.index.intersection(yield2.index)

                if len(common_years) >= 5:  # Only calculate if there are at least 5 common years
                    correlation = yield1.loc[common_years].corr(yield2.loc[common_years])
                    correlation_matrix.iloc[i, j] = correlation
                    correlation_matrix.iloc[j, i] = correlation

    return correlation_matrix


def get_day_of_year(date_str):
    if date_str[5:] == "02_30":
        date_str = date_str[:5] + "02_28"
    # Parse the date string into a datetime object
    date = datetime.strptime(date_str, '%Y_%m_%d')
    # Get the day of the year
    day_of_year = date.timetuple().tm_yday
    return day_of_year


def make_cc(comparison_cc_df, ndvi_df, preci_df, profile_length, plot):
    # detect value columns in dataframes and save day of year for fitting later
    ndvi_value_columns = [column for column in ndvi_df.columns if column[:4].isdigit()]
    ndvi_day_of_year = [get_day_of_year(date) for date in ndvi_value_columns]
    preci_value_columns = [column for column in preci_df.columns if column[:4].isdigit()]
    preci_day_of_year = [get_day_of_year(date) for date in preci_value_columns]

    # generate profile matrix (average over years)
    ndvi_profile_mtx = make_profiles(ndvi_df)
    preci_profile_mtx = make_profiles(preci_df)

    for i, (ndvi_profile, preci_profile) in enumerate(zip(ndvi_profile_mtx, preci_profile_mtx)):
        # extract CC name and region
        region_cc = comparison_cc_df[np.all(comparison_cc_df.iloc[:, :3].isin(ndvi_df.values[i, :3]), 1)]
        region_name = str(ndvi_df.values[i, :3]).replace('"', '').replace("'None' ", "").replace("/", "-")
        print(region_name)

        # extract values
        ndvi_values = ndvi_df[ndvi_value_columns].values[i]
        preci_values = preci_df[preci_value_columns].values[i].astype(dtype="float")

        # estimate sos, eos
        sos, eos = detect_seasons(ndvi_profile)

        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
            ax1.scatter(preci_day_of_year, preci_values, alpha=0.4)
            ax1.plot(np.arange(1, 366), preci_profile, linewidth=3, color="tab:orange", label="Fourier approximation (N=5)")
            ax1.set_ylabel("Precipitation (mm)")
            ax1.legend()

            ax2.scatter(ndvi_day_of_year, ndvi_values, color="tab:green", alpha=0.4)
            ax2.plot(np.arange(1, 366), ndvi_profile, linewidth=3, color="tab:orange", label="Fourier approximation (N=5)")
            ax2.vlines(x=[sos, eos], ymin=min(ndvi_values), ymax=max(ndvi_values),
                       linestyles="dotted",
                       color="tab:red",
                       label="My CC")
            for i, row in region_cc.iterrows():
                asap_sos = int((row.sos_s - 1) * 10 + 1)
                asap_eos = int((row.eos_e - 1) * 10 + 1)
                ax2.vlines(x=[asap_sos, asap_eos], ymin=min(ndvi_values), ymax=max(ndvi_values),
                           linestyles="dashed",
                           #color="tab:red",
                           label=f"{row.season} (ASAP)")
            ax2.set_ylabel("NDVI")
            ax2.set_xlabel("Day of the year")
            ax2.legend()
            fig.suptitle(f"NDVI & precipitation profile: {region_name}")
            plt.savefig(PROCESSED_DATA_DIR / f"crop calendar/plots/profiles/profile_{region_name}",
                        dpi=300)
            plt.close(fig)

        # rearrange data to make profile and append to data matrix
        if sos > eos:
            eos += 365
        ndvi_profile = np.append(ndvi_profile, ndvi_profile)[sos:eos][:profile_length]
        ndvi_profile_mtx.append(ndvi_profile)
        preci_profile = np.append(preci_profile, preci_profile)[sos:eos][:profile_length]
        preci_profile_mtx.append(preci_profile)
    return np.array(ndvi_profile_mtx), np.array(preci_profile_mtx)


def fit_fourier_series(days, values, N):
    """
    Fit a Fourier series to the given data.

    Parameters:
    days (array): Array of days of the year.
    values (array): Array of values corresponding to the days.
    N (int): Number of harmonics for the Fourier series.

    Returns:
    fitted_values (array): Fitted values for each day from 1 to 365.
    """
    # Define the Fourier series function
    def fourier_series(t, *a):
        ret = a[0] / 2  # a_0 / 2 term
        n_harmonics = (len(a) - 1) // 2
        for i in range(n_harmonics):
            ret += a[2 * i + 1] * np.cos(2 * np.pi * (i + 1) * t / 365) + a[2 * i + 2] * np.sin(2 * np.pi * (i + 1) * t / 365)
        return ret

    # Initial guess for the coefficients
    initial_guess = np.zeros(2 * N + 1)

    # Curve fitting
    params, params_covariance = curve_fit(fourier_series, days, values, p0=initial_guess)

    # Generate fitted values for each day from 1 to 365
    days = np.arange(1, 366)
    fitted_values = fourier_series(days, *params)

    return days, fitted_values


def fit_polynomial(days, values, degree):
    """
    Fit a polynomial to the given data.

    Parameters:
    days (array): Array of days of the year.
    values (array): Array of values corresponding to the days.
    degree (int): Degree of the polynomial.

    Returns:
    fitted_values (array): Fitted values for each day from 1 to 365.
    """
    # Fit the polynomial
    coefs = np.polyfit(days, values, degree)
    poly = np.poly1d(coefs)

    # Generate fitted values for each day from 1 to 365
    days = np.arange(1, 366)
    fitted_values = poly(days)

    return days, fitted_values


def plot_cluster_profiles(profile_data, cluster_data, cluster_column, cluster_name):

    fig, ax = plt.subplots(cluster_data[cluster_column].nunique(), 1, figsize=(7, 28))
    for label, data in cluster_data.groupby(cluster_column):
        for i in data.index:
            ax[label].plot(profile_data[i], alpha=0.6)
        ax[label].set_title(f"{cluster_name}-Cluster {label} containing parts of {data.country.unique()}")
        ax[label].set_ylim(np.min(profile_data) * 0.98, np.max(profile_data) * 1.02)
    plt.savefig(PROCESSED_DATA_DIR / f"data clustering/plots/results/{cluster_name}_clusters.jpg", dpi=1200)
    plt.show()


def plot_cluster_map(cluster_data, cluster_column, cluster_name):
    # load country-based map for better understanding for country borders
    africa_map = load_africa_map()
    # load map for areas of interest
    aoi_map = load_aoi_map()

    # fusion with cluster data
    geo_cluster_data = pd.merge(aoi_map, cluster_data, on=["country", "adm1", "adm2"])

    if cluster_data[cluster_column].nunique() <= 10:
        # Define a colormap
        cmap = plt.get_cmap('tab10')  # 'tab20' has 20 distinct colors
    else:
        # Define a colormap
        cmap = plt.get_cmap('tab20')  # 'tab20' has 20 distinct colors

    # start plotting
    fig, ax = plt.subplots(figsize=(10, 12))

    # Plot country maps as background
    africa_map[africa_map.NAME.isin(geo_cluster_data.country.unique())].plot(color="#e6e6e6", edgecolor="white", linewidth=2, ax=ax)

    # Create a list for legend entries
    legend_elements = []

    # Plot clusters one by one with distinct colors
    for i, (cluster, single_cluster_data) in enumerate(geo_cluster_data.groupby(cluster_column)):
        single_cluster_data.plot(color=cmap(i), edgecolor='white', linewidth=0.3, ax=ax, alpha=0.8)
        # Add a legend entry
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10, label=cluster))

    # Create the legend manually
    ax.legend(handles=legend_elements, loc="upper left", title=cluster_column)
    plt.show()

    # Annotate the map
    for idx, row in geo_cluster_data.iterrows():
        label = row['adm2'] if row['adm2'] != "None" else row['adm1']
        # calculated center point for comparison and easier plotting
        centroid = row["geometry"].representative_point()
        plt.annotate(text=label, xy=(centroid.x, centroid.y),
                     horizontalalignment='center', fontsize=4, color="black")

    # Save the plot
    plt.savefig(PROCESSED_DATA_DIR / f"data clustering/plots/results/{cluster_name}_cluster_map.jpg", dpi=1200)

    plt.show()


def cluster_validation_plot(clustered_yield_df, correlation_matrix, cluster_columns, cluster_names):
    # Create a subplot layout
    fig, ax = plt.subplots(len(cluster_columns) + 2, 2, figsize=(10, 6 + 3 * len(cluster_columns)))

    # 1. row: No grouping
    sns.kdeplot(clustered_yield_df["yield"], ax=ax[0, 0], fill=True)
    ax[0, 0].set_xlim(left=0, right=5)
    ax[0, 0].set_xlabel("Yield (t/ha)")
    ax[0, 0].set_title('Yield distribution over all data')

    sns.kdeplot(correlation_matrix.values.flatten(), ax=ax[0, 1], fill=True)
    ax[0, 1].set_xlim(left=-1, right=1)
    ax[0, 1].set_xlabel("Correlation")
    ax[0, 1].set_title('Yield correlation distribution over all data')
    print(f"Mean-correlation: {np.round(np.nanmean(correlation_matrix), 2)}")

    # 2. row: grouping by country
    grouped = clustered_yield_df.groupby("country")
    mean_correlation_ls = []

    for group, group_yield_df in grouped:
        ix_group = [group in x for x in correlation_matrix.index]

        # Plot on each subplot
        sns.kdeplot(group_yield_df["yield"], label=group, ax=ax[1, 0], fill=True, color=COUNTRY_COLORS[group])

        sns.kdeplot(correlation_matrix.iloc[ix_group, ix_group].values.flatten(), label=group, ax=ax[1, 1], fill=True, color=COUNTRY_COLORS[group])

        mean_correlation_ls.append(np.nanmean(correlation_matrix.iloc[ix_group, ix_group].values))

    ax[1, 0].set_xlim(left=0, right=5)
    ax[1, 0].set_xlabel("Yield (t/ha)")
    ax[1, 0].set_title('Yield distribution for each country')
    ax[1, 0].legend()
    ax[1, 1].set_xlim(left=-1, right=1)
    ax[1, 1].set_xlabel("Correlation")
    ax[1, 1].set_title('Yield correlation distribution for each country')
    ax[1, 1].legend(loc="upper left")
    print(f"Within-group-mean-correlation (country): {np.round(np.mean(mean_correlation_ls), 2)}")

    # following rows: grouping by cluster
    for i, (cluster_name, cluster_column) in enumerate(zip(cluster_names, cluster_columns)):
        grouped = clustered_yield_df.groupby(cluster_column)
        mean_correlation_ls = []

        for group, group_yield_df in grouped:
            group_regions = group_yield_df[["adm1", "adm2"]].drop_duplicates()
            ix_group = [list(x[1:3]) in group_regions.values for x in correlation_matrix.index]

            # Plot on each subplot
            sns.kdeplot(group_yield_df["yield"], label=group, ax=ax[i+2, 0], fill=True)

            sns.kdeplot(correlation_matrix.iloc[ix_group, ix_group].values.flatten(), label=group, ax=ax[i+2, 1], fill=True)
            mean_correlation_ls.append(np.nanmean(correlation_matrix.iloc[ix_group, ix_group].values))

        ax[i+2, 0].set_xlim(left=0, right=5)
        ax[i+2, 0].set_xlabel("Yield (t/ha)")
        ax[i+2, 0].set_title(f'Yield distribution for {cluster_name} clusters')
        ax[i+2, 0].legend()
        ax[i+2, 1].set_xlim(left=-1, right=1)
        ax[i+2, 1].set_xlabel("Correlation")
        ax[i+2, 1].set_title(f'Yield correlation distribution for {cluster_name} clusters')
        ax[i+2, 1].legend(loc="upper left")
        print(f"Within-group-mean-correlation ({cluster_name}): {np.round(np.mean(mean_correlation_ls), 2)}")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Save it
    plt.savefig(PROCESSED_DATA_DIR / f"data clustering/plots/results/{cluster_name}_validation.jpg", dpi=1200)
    # Show the plot
    plt.show()


def save_cluster_data(cluster_df):
    cluster_df.to_csv(PROCESSED_DATA_DIR / f"data clustering/cluster_data.csv", index=False)
