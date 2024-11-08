from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks

from config import PROCESSED_DATA_DIR
from crop_calendar.profile_generation import make_profiles
from maps.map_functions import load_africa_map


"""
Collection of functions to define a crop calendar based on vegetation index & threshold approach
"""


def detect_seasons(ndvi_profile, threshold=.25):
    # Detect local maxima
    maxima_indices, _ = find_peaks(ndvi_profile)
    # Detect local minima by inverting the NDVI data
    minima_indices, _ = find_peaks(-ndvi_profile)

    # Get the minimum and maximum NDVI values
    min_ndvi = ndvi_profile.min()
    max_ndvi = ndvi_profile.max()

    # Calculate amplitude
    amplitude = max_ndvi - min_ndvi

    # Define thresholds for the start and end of the season
    ndvi_threshold = min_ndvi + threshold * amplitude

    # find sos & eos
    sos = find_sos(ndvi_profile, ndvi_threshold)
    eos = find_eos(ndvi_profile, ndvi_threshold)

    return sos, eos


def find_sos(ndvi_window, start_threshold):
    # move from the minimum until SOS is found
    moving_point = np.argmin(ndvi_window) + 1

    while True:
        # if ndvi point exceeds the start threshold it is the SOS
        if ndvi_window[moving_point] > start_threshold:
            return moving_point + 1
        # set next step
        moving_point += 1
        if moving_point == 365:
            moving_point = 0


def find_eos(ndvi_window, end_threshold):
    # move from the minimum until EOS is found
    moving_point = np.argmin(ndvi_window) - 1

    while True:
        # if ndvi point exceeds the start threshold it is the day before EOS
        if ndvi_window[moving_point] > end_threshold:
            # + 2 comes because moving_point is index (starting at 0) and the day before EOS
            return moving_point + 2
        # set next step
        moving_point -= 1
        if moving_point == -1:
            moving_point = 364


def copy_cc(cc_df, column, from_name, to_name):
    copy_df = cc_df[cc_df[column] == from_name].copy()
    copy_df[column] = to_name
    if column == "adm2":
        copy_df["adm1"] = cc_df.loc[cc_df[column] == to_name, "adm1"][0]
    return pd.concat([cc_df[cc_df[column] != to_name], copy_df])


def plot_asap_crop_calendar(plot_cc_df, column, file_name=None):
    # rescaling target column to guarantee meaningful and intuitive colorspace
    min_value = min(plot_cc_df[column])
    plot_cc_df[column] = ((plot_cc_df[column] - min_value) % 36) # makes sure that 0 is the min value
    # Data filtering based on seasons
    plot_short_cc_df = plot_cc_df[plot_cc_df["season"].isin(["Maize (Short rains)", "Maize (Vuli/Bimodal)"])]
    plot_long_cc_df = plot_cc_df[plot_cc_df["season"].isin(["Maize (Meher)", "Maize (Long rains)", "Maize (Maika/Bimodal)"])]
    plot_annual_cc_df = plot_cc_df[plot_cc_df["season"].isin(["Maize", "Maize (Maimu/Unimodal)"])]
    # start plotting
    fig, ax = plt.subplots(1, 3, figsize=(12, 8))
    (ax1, ax2, ax3) = ax
    # Determine the common color scale across all data
    vmin = 0
    vmax = max(plot_annual_cc_df[column].max(), plot_long_cc_df[column].max(), plot_short_cc_df[column].max())
    # Plotting each subplot without a legend
    cmap = "YlGnBu"
    plot_annual_cc_df.plot(column=column, cmap=cmap, ax=ax1, vmin=vmin, vmax=vmax)
    plot_long_cc_df.plot(column=column, cmap=cmap, ax=ax2, vmin=vmin, vmax=vmax)
    plot_short_cc_df.plot(column=column, cmap=cmap, ax=ax3, vmin=vmin, vmax=vmax)
    # Adding titles to each subplot
    ax1.set_title("Annual")
    ax2.set_title("Long Season")
    ax3.set_title("Short Season")
    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    # Creating a colorbar with custom ticks and labels
    cbar = fig.colorbar(sm, ax=ax,  fraction=0.02, pad=0.04)
    ticks = np.arange(1, 37, 3) - min_value
    ticklabels = np.array(["January", "February", "March", "April", "May", 'June', 'July', 'August', "September", "October", "November", "December"])
    cbar.set_ticks(ticks[(ticks > 0) & (ticks <= np.max(vmax))])
    cbar.set_ticklabels(ticklabels[(ticks > 0) & (ticks <= np.max(vmax))])
    # Save plot if wanted
    if file_name:
        plt.savefig(PROCESSED_DATA_DIR / f"crop calendar/plots/ASAP/{file_name}.jpg", dpi=600)
    plt.show()


def plot_my_crop_calendar(cc_df):
    # load country-based map for better understanding for country borders
    africa_map = load_africa_map()

    # start plotting
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax1, ax2 = ax
    # Determine the common color scale across all data
    vmin = np.min(cc_df[["sos", "eos"]].values)
    vmax = np.max(cc_df[["sos", "eos"]].values)
    # Create a custom cyclic colormap
    #colors = ['#1f77b4', '#2ca02c', '#d68627', '#1f77b4']  # Blue, Green, Brown, Blue
    #n_bins = 100  # Discretizes the interpolation into bins
    #cmap_name = 'custom_seasons'
    #cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    cmap = "viridis"

    # Plot country maps as background
    africa_map[africa_map.NAME.isin(cc_df.country.unique())].plot(color="#e6e6e6", edgecolor="white", linewidth=2, ax=ax1)
    africa_map[africa_map.NAME.isin(cc_df.country.unique())].plot(color="#e6e6e6", edgecolor="white", linewidth=2, ax=ax2)

    # plot sos and eos
    cc_df.plot(column="sos", cmap=cmap, edgecolor='white', linewidth=0.2, ax=ax1, vmin=vmin, vmax=vmax)
    cc_df.plot(column="eos", cmap=cmap, edgecolor='white', linewidth=0.2, ax=ax2, vmin=vmin, vmax=vmax)

    # Adding titles to each subplot
    ax1.set_title("Start of the season")
    ax2.set_title("End of the season")

    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    # Creating a colorbar with custom ticks and labels
    cbar = fig.colorbar(sm, ax=ax,  fraction=0.02, pad=0.04)
    # make custom ticks
    ticks = np.array([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
    ticklabels = np.array(["January", "February", "March", "April", "May", 'June', 'July', 'August', "September", "October", "November", "December"])
    cbar.set_ticks(ticks[(ticks >= vmin) & (ticks <= vmax)])
    cbar.set_ticklabels(ticklabels[(ticks >= vmin) & (ticks <= vmax)])
    # Save plot if wanted
    plt.savefig(PROCESSED_DATA_DIR / f"crop calendar/plots/maps/my_cc_sos_eos.pdf", format="pdf")
    plt.show()


def plot_season_length(cc_df):
    # start plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    # Determine the common color scale across all data
    vmin = min(cc_df["season_length"])
    vmax = max(cc_df["season_length"])

    # Create a custom cyclic colormap
    #colors = ['#1f77b4', '#2ca02c', '#d68627', '#1f77b4']  # Blue, Green, Brown, Blue
    #n_bins = 100  # Discretizes the interpolation into bins
    #cmap_name = 'custom_seasons'
    #cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    cmap="viridis"

    cc_df.plot(column="season_length", cmap=cmap, ax=ax, vmin=vmin, vmax=vmax)
    # Adding titles to each subplot
    ax.set_title("Length of the season")
    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    # Creating a colorbar with custom ticks and labels
    cbar = fig.colorbar(sm, ax=ax,  fraction=0.02, pad=0.04)

    # Save plot if wanted
    plt.savefig(PROCESSED_DATA_DIR / f"crop calendar/plots/maps/my_cc_season_length.jpg", dpi=900)
    plt.show()


def plot_asap_growth_time(plot_cc_df):
    column = "growth_time"
    assert column in plot_cc_df.columns
    # Data filtering based on seasons
    plot_short_cc_df = plot_cc_df[plot_cc_df["season"].isin(["Maize (Short rains)", "Maize (Vuli/Bimodal)"])]
    plot_long_cc_df = plot_cc_df[plot_cc_df["season"].isin(["Maize (Meher)", "Maize (Long rains)", "Maize (Maika/Bimodal)"])]
    plot_annual_cc_df = plot_cc_df[plot_cc_df["season"].isin(["Maize", "Maize (Maimu/Unimodal)"])]
    # start plotting
    fig, ax = plt.subplots(1, 3, figsize=(12, 8))
    (ax1, ax2, ax3) = ax
    # Determine the common color scale across all data
    vmin = plot_cc_df[column].min()
    vmax = plot_cc_df[column].max()
    # Plotting each subplot without a legend
    cmap = "YlGnBu"
    plot_annual_cc_df.plot(column=column, cmap=cmap, ax=ax1, vmin=vmin, vmax=vmax)
    plot_long_cc_df.plot(column=column, cmap=cmap, ax=ax2, vmin=vmin, vmax=vmax)
    plot_short_cc_df.plot(column=column, cmap=cmap, ax=ax3, vmin=vmin, vmax=vmax)
    # Adding titles to each subplot
    ax1.set_title("Annual")
    ax2.set_title("Long Season")
    ax3.set_title("Short Season")
    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.04, label="Dekades")
    # Save plot if wanted
    plt.savefig(PROCESSED_DATA_DIR / f"crop calendar/plots/growth_time.jpg", dpi=600)
    plt.show()


def get_day_of_year(date_str):
    if date_str[5:] == "02_30":
        date_str = date_str[:5] + "02_28"
    # Parse the date string into a datetime object
    date = datetime.strptime(date_str, '%Y_%m_%d')
    # Get the day of the year
    day_of_year = date.timetuple().tm_yday
    return day_of_year


def make_cc(fao_cc_df, ndvi_df, preci_df, threshold=.25, plot=False):
    # detect value columns in dataframes and save day of year for fitting later
    ndvi_value_columns = [column for column in ndvi_df.columns if column[:4].isdigit()]
    ndvi_day_of_year = [get_day_of_year(date) for date in ndvi_value_columns]
    preci_value_columns = [column for column in preci_df.columns if column[:4].isdigit()]
    preci_day_of_year = [get_day_of_year(date) for date in preci_value_columns]

    # generate profile matrix (average over years)
    ndvi_profile_mtx = make_profiles(ndvi_df)
    preci_profile_mtx = make_profiles(preci_df)

    # lists to be filled
    sos_ls = []
    eos_ls = []

    regions_exceeding_fao_cc = []

    for i, (ndvi_profile, preci_profile) in enumerate(zip(ndvi_profile_mtx, preci_profile_mtx)):
        # extract CC name and region
        region_cc = fao_cc_df[fao_cc_df["adm"] == ndvi_df.iloc[i].adm]
        region_name = str(ndvi_df.values[i, :3]).replace('"', '').replace("'None' ", "").replace("/", "-")

        # extract values
        ndvi_values = ndvi_df[ndvi_value_columns].values[i]
        preci_values = preci_df[preci_value_columns].values[i].astype(dtype="float")

        # estimate sos, eos
        sos, eos = detect_seasons(ndvi_profile, threshold=threshold)
        sos_ls.append(sos)
        eos_ls.append(eos)

        # note if eos is beyond FAO crop calendars harvest periode
        if eos > region_cc.values[0, -1]:
            regions_exceeding_fao_cc.append(region_name)

        if plot:
            #fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
            # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
            fig = plt.figure(figsize=(9, 8))
            gs = gridspec.GridSpec(3, 1, height_ratios=[2, 3, 1])

            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])
            ax1.scatter(preci_day_of_year, preci_values, alpha=0.2)
            ax1.plot(np.arange(1, 366), preci_profile, linewidth=3, color="tab:orange",
                     label="Fourier approximation (N=5)")
            ax1.set_ylabel("Precipitation (mm)")
            ax1.legend()

            ax2.scatter(ndvi_day_of_year, ndvi_values, color="tab:green", alpha=0.3)
            ax2.plot(np.arange(1, 366), ndvi_profile, linewidth=3, color="tab:orange",
                     label="Fourier approximation (N=5)")
            ax2.vlines(x=[sos], ymin=min(ndvi_values), ymax=max(ndvi_values),
                       alpha=.6,
                       linestyles="dotted",
                       color="tab:red",
                       label="Start of Season")
            ax2.vlines(x=[eos], ymin=min(ndvi_values), ymax=max(ndvi_values),
                       alpha=.6,
                       linestyles="dashed",
                       color="tab:red",
                       label="End of Season")
            ax2.set_ylabel("NDVI")
            ax2.legend()

            interval_names = ["Planting", "Harvest"]
            interval_doys = [(region_cc.values[0, -4], region_cc.values[0, -3]),
                             (region_cc.values[0, -2], region_cc.values[0, -1])]

            # Plot each season as a rectangle
            for i, (name, (start, end), color) in enumerate(zip(interval_names, interval_doys, ["brown", "green"])):
                height = i * .4 + .1
                if start < end:
                    rect = Rectangle((start, height), end - start, 0.3, color=color, alpha=0.5)
                    ax3.add_patch(rect)
                else:
                    rect1 = Rectangle((start, height), 365 - start, 0.3, color=color, alpha=0.5)
                    ax3.add_patch(rect1)
                    rect2 = Rectangle((0, height), end, 0.2, color=color, alpha=0.5)
                    ax3.add_patch(rect2)
                ax3.text(-50, height, name)

            # Set the limits of the plot
            ax3.set_ylim(0, len(region_cc))
            ax3.set_yticks([])
            ax3.set_xlabel("Day of the year")

            # Set the limits of the plot with padding
            ax1.set_xlim(-10, 375)
            ax2.set_xlim(-10, 375)
            ax3.set_xlim(-10, 375)

            #fig.suptitle(
            #    f"Crop Calendar (CC) estimation based on NDVI profile for '{region_name}'\nwith FAO CC for comparison")
            plt.savefig(PROCESSED_DATA_DIR / f"crop calendar/plots/profiles/profile_{region_name}.pdf",
                        format="pdf")
            plt.close(fig)
        #plt.show()
        #break

    if regions_exceeding_fao_cc:
        print(len(regions_exceeding_fao_cc), " region's eos exceed the FAO harvest periode: ", regions_exceeding_fao_cc)

    return sos_ls, eos_ls


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
