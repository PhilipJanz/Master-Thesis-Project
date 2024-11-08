import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gamma, norm

from crop_calendar.profile_generation import make_profiles, get_day_of_year
import statsmodels.api as sm

def generate_si(folder_path, file_name, si_file_name, gaussian_rolling_averge_window_size=None, distibution="gaussian", trend_cleaning=True):
    """

    :param gaussian_rolling_averge_window_size: (optional) applies rolling average. Especially interesting for
    precipitation data with sparse data or noisy data (remote sensing)
    :return: df of si values
    """
    data_df = pd.read_csv(folder_path / file_name, keep_default_na=False)

    # extract the day of year (doy) for each value column
    value_columns = [column for column in data_df.columns if column[:4].isdigit()]
    doy_ls = np.array([get_day_of_year(date) for date in value_columns])
    # handle leap year
    doy_ls[doy_ls > 365] = 365

    # create ni dataframe to be filled with ni values in a loop over the rows
    si_2darray = np.zeros((len(data_df), len(value_columns)))

    trend_ls = []
    p_value_ls = []
    for i, row in data_df.iterrows():
        print(f"Processing {si_file_name} ({i + 1}/{len(data_df)})", end='\r')
        values = row[value_columns].values
        # Convert strings to floats and use None for empty strings
        values = [float(x) if isinstance(x, str) and x else np.nan if x == "" else x for x in values]

        # Fill empty stings with nearest kneighbors
        if np.any(np.isnan(values)):
            values = [np.nanmean(values[j-1:j+2]) if not x else x for j, x in enumerate(values)]

        # apply gaussian rolling average if wanted
        if gaussian_rolling_averge_window_size:
            values = pd.Series(values).rolling(window=gaussian_rolling_averge_window_size,
                                               win_type='gaussian',
                                               min_periods=1,
                                               center=True).mean(std=gaussian_rolling_averge_window_size/3)
        values = np.array(values)

        if trend_cleaning:
            # test on linear trends (like climate change)
            X = sm.add_constant(np.arange(len(values)))
            model = sm.OLS(values, X).fit()
            p_value = model.pvalues[1]
            p_value_ls.append(p_value)
            trend_ls.append(model.params[1])
            # clean trend if significant
            if p_value < .05:
                values = values - model.predict(X)

        # calculate the stnd. dev. for each doy individually (its very different across the year)
        for doy in doy_ls:
            doy_ix_ls = doy_ls == doy

            doy_values = values[doy_ix_ls]

            # if one day has the value 0 (usually in the case of exact 0 precipitation) we skip it (set to 0)
            if np.any(doy_values == 0):
                si_2darray[i, doy_ix_ls] = 0
                continue

            # calculate si
            if distibution == "gaussian":
                doy_values = doy_values - np.mean(doy_values)
                assert np.std(doy_values) > 0
                si_values = doy_values / np.std(doy_values)
            elif distibution == "gamma":
                # Fit a gamma distribution to the non-zero precipitation data
                params = gamma.fit(doy_values[doy_values > 0], floc=0)
                shape, loc, scale_param = params

                # Calculate cumulative probabilities for all data (including zeros)
                cdf_values = gamma.cdf(doy_values, shape, loc=loc, scale=scale_param)

                # Convert cumulative probabilities to SPI values using the standard normal distribution
                si_values = norm.ppf(cdf_values)
            else:
                raise AssertionError(f"Unexpected probability distribution input: {distibution}")

            si_2darray[i, doy_ix_ls] = si_values

    # report on trend detection:
    print(f"Found {np.sum(np.array(p_value_ls) < .05)}/{len(p_value_ls)} time series with significant trend. ({np.sum(np.array(trend_ls)[np.array(p_value_ls) < .05] > 0)} times positive trend)")
    print(f"Averge trend: {np.mean(trend_ls)}")

    # fill into dataframe
    si_df = data_df.copy()
    si_df.loc[:, np.array(value_columns)] = si_2darray

    # save the hard work
    si_df.to_csv(folder_path / si_file_name, index=False)
    return si_df
