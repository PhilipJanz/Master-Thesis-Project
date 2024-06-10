from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks

from config import PROCESSED_DATA_DIR
from crop_calendar.profile_generation import make_profiles


def detect_one_season(ndvi_profile):
    # Get the minimum and maximum NDVI values
    min_ndvi = ndvi_profile.min()
    max_ndvi = ndvi_profile.max()

    # Calculate amplitude
    amplitude = max_ndvi - min_ndvi

    # Define thresholds for the start and end of the season
    start_threshold = min_ndvi + 0.2 * amplitude
    end_threshold = max_ndvi - 0.8 * amplitude



def detect_seasons(ndvi_profile):
    # Detect local maxima
    maxima_indices, _ = find_peaks(ndvi_profile)
    # Detect local minima by inverting the NDVI data
    minima_indices, _ = find_peaks(-ndvi_profile)

    if True: #len(maxima_indices) == 1:
        # Get the minimum and maximum NDVI values
        min_ndvi = ndvi_profile.min()
        max_ndvi = ndvi_profile.max()

        # Calculate amplitude
        amplitude = max_ndvi - min_ndvi

        # Define thresholds for the start and end of the season
        start_threshold = min_ndvi + 0.1 * amplitude
        end_threshold = max_ndvi - 0.7 * amplitude

        # find sos & eos
        sos = find_sos(ndvi_profile, start_threshold)
        eos = find_eos(ndvi_profile, end_threshold)

        return sos, eos



def find_sos(ndvi_window, start_threshold):
    # make a list of values above threshold
    above_threshold = ndvi_window > start_threshold

    # find the first value above threshold
    first_above_threshold = np.where(above_threshold[1:] * ~above_threshold[:-1])[0][0] + 1

    # plus 1 because we want the day of the year (starts with 1)
    return first_above_threshold + 1


def find_eos(ndvi_window, end_threshold):
    # make a list of values below threshold
    below_threshold = ndvi_window < end_threshold

    # find the first value above threshold
    first_below_threshold = np.where(below_threshold[1:] * ~below_threshold[:-1])[0][0] + 1

    return first_below_threshold


def copy_cc(cc_df, column, from_name, to_name):
    copy_df = cc_df[cc_df[column] == from_name].copy()
    copy_df[column] = to_name
    if column == "adm2":
        copy_df["adm1"] = cc_df.loc[cc_df[column] == to_name, "adm1"][0]
    return pd.concat([cc_df[cc_df[column] != to_name], copy_df])


def plot_seasonal_crop_calendar(plot_cc_df, column, file_name=None):
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
        plt.savefig(PROCESSED_DATA_DIR / f"crop calendar/plots/{file_name}.jpg", dpi=600)
    plt.show()


def plot_growth_time(plot_cc_df):
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


def make_cc(asap_cc_df, ndvi_df, preci_df, plot):
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

    for i, (ndvi_profile, preci_profile) in enumerate(zip(ndvi_profile_mtx, preci_profile_mtx)):
        # extract CC name and region
        region_cc = asap_cc_df[np.all(asap_cc_df.iloc[:, :3].isin(ndvi_df.values[i, :3]), 1)]
        region_name = str(ndvi_df.values[i, :3]).replace('"', '').replace("'None' ", "").replace("/", "-")
        print(region_name)

        # extract values
        ndvi_values = ndvi_df[ndvi_value_columns].values[i]
        preci_values = preci_df[preci_value_columns].values[i].astype(dtype="float")

        # estimate sos, eos
        sos, eos = detect_seasons(ndvi_profile)
        sos_ls.append(sos)
        eos_ls.append(eos)

        if plot:
            #fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
            fig = plt.figure(figsize=(9, 8))
            gs = gridspec.GridSpec(3, 1, height_ratios=[2, 3, 2])

            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])
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
            ax2.set_ylabel("NDVI")
            ax2.legend()

            for j, (ix, row) in enumerate(region_cc.iterrows()):
                # Define the data (start_dekade, end_dekade)
                seasons_data = [(row.sos_s, row.sos_e), (row.sos_e, row.eos_s), (row.eos_s, row.eos_e)]

                # height
                height = j + 0.3

                # Plot each season as a rectangle
                for i, ((start, end), color) in enumerate(zip(seasons_data, ["brown", "green", "orange"])):
                    if start < end:
                        rect = Rectangle((start, height), end - start, 0.4, color=color, alpha=0.5)
                        ax3.add_patch(rect)
                    else:
                        rect1 = Rectangle((start, height), 365 - start, 0.4, color=color, alpha=0.5)
                        ax3.add_patch(rect1)
                        rect2 = Rectangle((0, height), end, 0.4, color=color, alpha=0.5)
                        ax3.add_patch(rect2)
                    ax3.text(-50, height, row.season)

            # Set the limits of the plot
            ax3.set_ylim(0, len(region_cc))
            ax3.set_yticks([])
            ax3.set_xlabel("Day of the year")

            # Set the limits of the plot with padding
            ax1.set_xlim(-10, 375)
            ax2.set_xlim(-10, 375)
            ax3.set_xlim(-10, 375)

            fig.suptitle(f"Crop Calendar (CC) estimation based on NDVI profile for '{region_name}'\nwith ASAP CC for comparison")
            plt.savefig(PROCESSED_DATA_DIR / f"crop calendar/plots/profiles/profile_{region_name}",
                        dpi=300)
            plt.close(fig)
        #plt.show()
        #break

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
