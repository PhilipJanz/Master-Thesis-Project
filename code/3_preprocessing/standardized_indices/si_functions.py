import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from crop_calendar.profile_generation import make_profiles, get_day_of_year


def generate_si(folder_path, file_name, min_avg=None):
    """

    :param min_avg: this parameter defines a theshold under which the si values are not calculated
    and instead it all gives 0.
    The background is that for precipitaiton there are areas and dekades with barely any precipitation.
    Those can be ignored. (usually not even part of the crop season)
    :return: df of si values
    """
    data_df = pd.read_csv(folder_path / file_name, keep_default_na=False)
    data_df = data_df[~data_df.country.isin(["Ethiopia", "Kenya"])].reset_index(drop=True) #TODO: delete line

    # generate profile matrix (average over years)
    profile_mtx = make_profiles(data_df)

    # extract the day of year (doy) for each value column
    value_columns = [column for column in data_df.columns if column[:4].isdigit()]
    doy_ls = np.array([get_day_of_year(date) for date in value_columns])

    # create ni dataframe to be filled with ni values in a loop over the rows
    si_df = data_df.copy()

    for i, row in data_df.iterrows():
        print(i, end='\r')
        values = row[value_columns].values
        # Convert strings to floats and use None for empty strings
        values = [float(x) if isinstance(x, str) and x else np.nan if x == "" else x for x in values]
        # Fill empty stings with nearest kneighbors
        if np.any(np.isnan(values)):
            values = [np.nanmean(values[j-1:j+2]) if not x else x for j, x in enumerate(values)]

        avg_values = profile_mtx[i][doy_ls - 1]
        divergence = values - avg_values

        # calculate the stnd. dev. for each doy individually (its very different across the year)
        for doy in doy_ls:
            doy_ix_ls = doy_ls == doy

            # check if average is above threhold
            if min_avg:
                if np.mean(np.array(values)[doy_ix_ls]) < min_avg:
                    si_df.loc[i, np.array(value_columns)[doy_ix_ls]] = 0
                    continue

            # calculate ni
            si_values = divergence[doy_ix_ls] / np.std(divergence[doy_ix_ls])
            # fill into dataframe
            si_df.loc[i, np.array(value_columns)[doy_ix_ls]] = si_values

    # save the hard work
    si_df.to_csv(folder_path / "_".join(["si", file_name]), index=False)
    return si_df
