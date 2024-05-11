import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from config import POCESSED_DATA_DIR


def copy_cc(cc_df, column, from_name, to_name):
    cc_df = cc_df[cc_df[column] != to_name]
    copy_df = cc_df[cc_df[column] == from_name].copy()
    copy_df[column] = to_name
    return pd.concat([cc_df, copy_df])


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
        plt.savefig(POCESSED_DATA_DIR / f"crop calendar/plots/{file_name}.jpg", dpi=600)
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
    plt.savefig(POCESSED_DATA_DIR / f"crop calendar/plots/growth_time.jpg", dpi=600)
    plt.show()
