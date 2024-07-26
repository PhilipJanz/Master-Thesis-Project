import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *


##### IMPORT #####

def read_ethiopia_yield():
    """
    Read raw yield data for Ethiopia
    :return: yield df
    """
    raw_yield_df = pd.read_csv(SOURCE_DATA_DIR / "yield/FEWS NET/maize_yield_east_africa.csv")#, header=None)
    raw_yield_df["country"] = raw_yield_df.admin_0
    ethiopia_yield_df = raw_yield_df[raw_yield_df.country == "Ethiopia"].copy()
    #print(len(ethiopia_yield_df))
    ethiopia_yield_df = ethiopia_yield_df.rename(columns={"admin_1": "adm1", "admin_2": "adm2", "season_name": "season"})
    ethiopia_yield_df.loc[ethiopia_yield_df["adm2"] == "none", "adm2"] = "None"
    ethiopia_yield_df["harv_year"] = [int(date[:4]) for date in ethiopia_yield_df.season_date]
    ethiopia_yield_df = ethiopia_yield_df[["country", 'adm1', 'adm2', 'season', 'harv_year', 'indicator', 'value']]
    #print(len(ethiopia_yield_df))
    # drop duplicates
    ethiopia_yield_df = ethiopia_yield_df.drop_duplicates()
    #print(len(ethiopia_yield_df))
    # if duplicates are found by leaving out the value, those duplicates have unequal values are have to be discarded
    duplicates = ethiopia_yield_df.duplicated(subset=['country', 'adm1', 'adm2', 'season', 'harv_year', 'indicator'], keep=False)
    if any(duplicates):
        print(f"{sum(duplicates) / 6} duplicates found")
        duplicate_df = ethiopia_yield_df[duplicates]
        # Keep only the rows that are not duplicates
        ethiopia_yield_df = ethiopia_yield_df[~duplicates]
    #print(len(ethiopia_yield_df))
    ethiopia_yield_df = ethiopia_yield_df.pivot(index=['country', 'adm1', 'adm2', 'season', 'harv_year'],
                                                columns='indicator', values='value').reset_index()
    ethiopia_yield_df = ethiopia_yield_df.rename(columns={"Area Planted": "area", "Quantity Produced": "production", "Yield": "yield"})
    ethiopia_yield_df = ethiopia_yield_df.drop("Area Harvested", axis=1)
    ethiopia_yield_df = ethiopia_yield_df.dropna(subset=["yield"])
    # filter out the short rainy season 'Belg' since there is not enough data for modeling (68 after cleaning)
    ethiopia_yield_df = ethiopia_yield_df[ethiopia_yield_df.season != 'Belg']
    ethiopia_yield_df = ethiopia_yield_df.dropna()
    # further renaming
    ethiopia_yield_df.loc[ethiopia_yield_df.adm2 == "Jijiga", "adm2"] = "Fafan"
    ethiopia_yield_df.loc[ethiopia_yield_df.adm2 == "Kembata Alaba Tembaro\r\n", "adm2"] = "Kembata Tembaro"
    ethiopia_yield_df.loc[ethiopia_yield_df.adm2 == "Alaba", "adm2"] = "Halaba"
    ethiopia_yield_df.loc[ethiopia_yield_df.adm2 == "Jijiga", "adm2"] = "Fafan"
    ethiopia_yield_df.loc[ethiopia_yield_df.adm2 == "Silltie", "adm2"] = "Siltie"
    ethiopia_yield_df.loc[ethiopia_yield_df.adm2 == "Sidama", "adm1"] = "Sidama"
    ethiopia_yield_df.loc[ethiopia_yield_df.adm2 == "Konta", "adm1"] = "South West Ethiopia"
    ethiopia_yield_df.loc[ethiopia_yield_df.adm2 == "Sheka", "adm1"] = "South West Ethiopia"
    ethiopia_yield_df.loc[ethiopia_yield_df.adm2 == "Keffa", "adm1"] = "South West Ethiopia"
    ethiopia_yield_df.loc[ethiopia_yield_df.adm2 == "Dawro", "adm1"] = "South West Ethiopia"
    ethiopia_yield_df.loc[ethiopia_yield_df.adm2 == "Bench Maji", "adm1"] = "South West Ethiopia"
    print(f"Loaded {len(ethiopia_yield_df)} yield datapoints for Ethiopia.")
    return ethiopia_yield_df

def read_malawi_yield():
    """
    Read raw yield data for Malawi
    :return: yield df
    """
    raw_yield_df = pd.read_csv(SOURCE_DATA_DIR / "yield/FEWS NET/maize_yield_south_africa.csv")#, header=None)
    malawi_yield_df = raw_yield_df[raw_yield_df.country == "Malawi"].copy()
    #print(len(malawi_yield_df))
    # filtering irrigation season and others
    malawi_yield_df = malawi_yield_df[malawi_yield_df.crop_production_system.isin(["Rainfed (PS)", "All (PS)"])]
    malawi_yield_df = malawi_yield_df.rename(columns={"admin_1": "adm1", "admin_2": "adm2", "season_name": "season"})
    malawi_yield_df.loc[malawi_yield_df["adm2"] == "none", "adm2"] = "None"
    malawi_yield_df["harv_year"] = [int(date[:4]) for date in malawi_yield_df.season_date]
    malawi_yield_df = malawi_yield_df[["country", 'adm1', 'adm2', 'season', 'harv_year', 'indicator', 'value']]
    #print(len(malawi_yield_df))
    # drop duplicates
    malawi_yield_df = malawi_yield_df.drop_duplicates()
    #print(len(malawi_yield_df))
    # if duplicates are found by leaving out the value, those duplicates have unequal values are have to be discarded
    duplicates = malawi_yield_df.duplicated(subset=['country', 'adm1', 'adm2', 'season', 'harv_year', 'indicator'], keep=False)
    if any(duplicates):
        print(f"{sum(duplicates) / 6} duplicates found")
        duplicate_df = malawi_yield_df[duplicates]
        # Keep only the rows that are not duplicates
        malawi_yield_df = malawi_yield_df[~duplicates]
    #print(len(malawi_yield_df))
    malawi_yield_df = malawi_yield_df.pivot(index=['country', 'adm1', 'adm2', 'season', 'harv_year'],
                                            columns='indicator', values='value').reset_index()
    malawi_yield_df = malawi_yield_df.rename(columns={"Area Planted": "area", "Quantity Produced": "production", "Yield": "yield"})
    malawi_yield_df = malawi_yield_df.dropna()
    print(f"Loaded {len(malawi_yield_df)} yield datapoints for Malawi.")
    return malawi_yield_df

def read_kenya_and_zambia_yield():
    """
    Read raw yield data for Kenya and Zambia
    :return: yield df
    """
    # Load data into a pandas DataFrame
    raw_yield_df = pd.read_csv(SOURCE_DATA_DIR / "yield/burkinafaso_kenya_zambia/adm_crop_production_BF_KE_ZM.csv", header=None)
    raw_yield_df = raw_yield_df[[country in ["Zambia", "Kenya"] for country in raw_yield_df[2]]]
    raw_yield_df = raw_yield_df[[crop == "Maize" for crop in raw_yield_df[7]]]
    yield_df = pd.DataFrame()
    yield_df["country"] = raw_yield_df[2]
    yield_df["adm1"] = raw_yield_df[4]
    yield_df["adm2"] = raw_yield_df[5]
    yield_df.loc[yield_df["adm2"] == "none", "adm2"] = "None"
    yield_df["season"] = raw_yield_df[8]
    yield_df["harv_year"] = raw_yield_df[11]
    yield_df["indicator"] = raw_yield_df[14]
    yield_df["value"] = raw_yield_df[15]
    # drop duplicates
    yield_df = yield_df.drop_duplicates()
    # if duplicates are found by leaving out the value, those duplicates have unequal values are have to be discarded
    duplicates = yield_df.duplicated(subset=['country', 'adm1', 'adm2', 'season', 'harv_year', 'indicator'], keep=False)
    if any(duplicates):
        print(f"{sum(duplicates) / 6} duplicates found")
        # Keep only the rows that are not duplicates
        yield_df = yield_df[~duplicates]
    yield_df = yield_df.pivot(index=['country', 'adm1', 'adm2', 'season', 'harv_year'],
                              columns='indicator', values='value').reset_index()
    yield_df = yield_df.dropna(subset=["yield"])
    # too few short season datapoints for Kenya
    yield_df = yield_df[yield_df.season != 'Short']
    yield_df = yield_df.dropna()
    print(f"Loaded {len(yield_df)} yield datapoints for Kenya & Zambia.")
    return yield_df

def read_tanzania_yield():
    """
    Read raw Tanzanian yield data
    :return: yield df
    """
    raw_tanzania_yield_df = pd.read_csv(SOURCE_DATA_DIR / "yield/tanzania/tanzania_yield_basic_data_booklet_2020.csv", sep=" ", na_values="-", dtype={"2011": float, "2011": float, "2012": float, "2013": float, "2014": float, "2015": float, "2016": float, "2017": float, "2018": float, "2019": float})
    my_regions = ["Arusha", 'Dar es Salaam', 'Dodoma', 'Geita', 'Iringa',
                  'Kagera', 'Katavi', 'Kigoma', 'Kilimanjaro', 'Lindi', 'Manyara',
                  'Mara', 'Mbeya', 'Morogoro', 'Mtwara', 'Mwanza', 'Njombe',
                  'Pwani', 'Rukwa', 'Ruvuma',
                  'Shinyanga', 'Simiyu', 'Singida', 'Songwe', 'Tabora', 'Tanga']
    area_ls = []
    production_ls = []
    yield_ls = []
    for i, region in enumerate(my_regions):
        area_ls.append(raw_tanzania_yield_df.values[i*3, 2:])
        production_ls.append(raw_tanzania_yield_df.values[1 + i*3, 2:])
        yield_ls.append(raw_tanzania_yield_df.values[2 + i*3, 2:])
    tanzania_yield_df = pd.DataFrame({"adm1": np.repeat(my_regions, 10),
                                      "harv_year": np.tile(np.arange(2011, 2021), len(my_regions)),
                                      "area": np.array(area_ls).astype("float").flatten() * 1000,
                                      "production": np.array(production_ls).astype("float").flatten()* 1000,
                                      "yield": np.array(yield_ls).astype("float").flatten()})
    tanzania_yield_df["country"] = "Tanzania"
    tanzania_yield_df["adm2"] = "None"
    tanzania_yield_df["season"] = "Annual"
    tanzania_yield_df = tanzania_yield_df.dropna()
    print(f"Loaded {len(tanzania_yield_df)} yield datapoints for Tanzania.")
    return tanzania_yield_df


##### CLEANING #####

def detect_outliers_with_polyfit(data, column, degree=2, threshold=3, plot=False):
    """
    Detects outliers based on a polynomial trend.

    :param data: DataFrame with 'harv_year' and the 'column' columns
    :param column: column of interest for outliers
    :param degree: degree of the polynomial to fit.
    :param threshold: The Z-score threshold to identify outliers. Default is 3.
    :param plot: make a plot or not (recommended to understand behaviour)
    :return: The dataframe containing the outliers.
    """
    # sufficient data size
    if len(data) < 5:
        return []

    # sort data
    data = data.sort_values("harv_year")

    # Fitting a polynomial of specified degree
    coefficients = np.polyfit(data['harv_year'], data[column], degree)
    polynomial = np.poly1d(coefficients)

    # Calculating the fitted values
    fitted_values = polynomial(data['harv_year'])

    # Calculating residuals
    residuals = data[column] - fitted_values
    #return residuals
    # Calculating Z-scores of the residuals
    std = np.std(residuals)
    data['z_score'] = residuals / std

    # Identifying outliers
    outliers = data[abs(data['z_score']) > threshold]

    if plot:
        plt.figure(figsize=(8, 5))
        plt.fill_between(x=data['harv_year'], y1=fitted_values - 3*std, y2=fitted_values + 3*std, color='red', label='3 standard deviations', alpha=0.15)
        plt.fill_between(x=data['harv_year'], y1=fitted_values - 2*std, y2=fitted_values + 2*std, color='seashell', label='2 standard deviations', alpha=0.9)
        plt.scatter(data['harv_year'], data[column], label='Data Points')
        plt.plot(data['harv_year'], fitted_values, color='red', label='Polynomial Fit')
        plt.scatter(outliers['harv_year'], outliers[column], color='orange', label='Outliers')
        # Annotating Z-scores on the plot
        for i, point in outliers.iterrows():
            plt.annotate(f"{abs(point['z_score']):.2f}",  # Format to 2 decimal places
                         (point['harv_year'], point[column]),
                         textcoords="offset points",  # how to position the text
                         xytext=(0,10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
        plt.ylim(ymin=0)
        plt.legend()
        plt.xlabel('Year')
        plt.ylabel(column)
        plt.title(f'Outlier detection based on {column} for {data[["country", "adm1", "adm2"]].values[0]}')
        plt.show()

    return outliers


def detect_suspicious_data(data, column, threshold=0.0001, plot=False):
    """
    Detect implausible consecutive yield values that are too similar,
    based on a given threshold.

    :param data: DataFrame with a column named 'yield'
    :param column:
    :param threshold: represents the minimum allowed difference between consecutive yield values to be considered plausible
    :param plot: make a plot or not (recommended to understand behaviour)
    :return: The dataframe with suspicious data
    """
    # sort data
    data = data.sort_values("harv_year")

    # Calculate the absolute difference between consecutive yield values
    data[column + '_diff'] = data[column].diff().abs()

    # Filter out the rows where the yield difference is below the threshold
    threshold_surpassed = data[column + '_diff'].values < threshold
    for i, x in enumerate(threshold_surpassed):
        if x:
            threshold_surpassed[i-1] = True
    suspicious_df = data[threshold_surpassed].copy()

    if plot:
        plt.figure(figsize=(8, 5))
        plt.scatter(data['harv_year'], data[column], label='Data points')
        plt.scatter(suspicious_df['harv_year'], suspicious_df[column], color='orange', label=f'Suspicious {column} data')
        # Annotating Z-scores on the plot
        for i, point in suspicious_df.iterrows():
            plt.annotate(f"{point[column]:.5f}",  # Format to 3 decimal places
                         (point['harv_year'], point[column]),
                         textcoords="offset points",  # how to position the text
                         xytext=(0,10),  # distance from text to points (x,y)
                         ha='center', rotation=90)  # horizontal alignment can be left, right or center
        plt.ylim(ymin=0)
        plt.legend()
        plt.xlabel('Year')
        plt.ylabel(column)
        plt.title(f'Suspicious data detection based on {column} for {data[["country", "adm1", "adm2"]].values[0]}')
        plt.show()

    return suspicious_df

def is_close(a, b, rel_tol=0.01):
    """
    Determines if two values are relatively close
    :param a: first value
    :param b: second value
    :param rel_tol: percentage of bigger value to be underneath
    :return: (bool) is close
    """
    return abs(a-b) <= rel_tol * max(abs(a), abs(b))

def my_merge(df1, df2, on, rel_tol=0.01):
    """
    This function (inner-) merges two dataframes with a certain tolerance to small divergence in numeric values.
    This tolerance it chosen by setting 'rel_tol'
    It is the maximum allowed relative divergence. See is_close() for code.
    :param df1: df 1
    :param df2: df 2
    :param on: list of columns that give unique values
    :param rel_tol: see is_close()
    :return: merged df
    """
    assert np.all(~df1.isna()), "Found na value in df1. Check it out."
    assert np.all(~df2.isna()), "Found na value in df2. Check it out."
    # make lists for indicies to keep
    index1 = list(df1.index)
    index2 = list(df2.index)
    n_inconsistency = 0
    n_consistency = 0
    for i, row1 in df1.iterrows():
        # find match
        for j, row2 in df2.iterrows():
            if np.all(row1[on] == row2[on]):
                if is_close(a=row1["yield"], b=row2["yield"], rel_tol=rel_tol):
                    n_consistency += 1
                    index2.remove(j)
                else:
                    print(f'Found inconsistency for {row1[on].values}: {row1["yield"]} | {row2["yield"]}')
                    n_inconsistency += 1
                    index1.remove(i)
                    index2.remove(j)
                break
    # merge together the remaining data
    merged_df = pd.concat([df1.loc[index1], df2.loc[index2]]).reset_index()
    if n_inconsistency:
        print(f"Found {n_inconsistency} inconsistencies.")
        print(f"Remaining datapoints after merge: {len(merged_df)} (both: {n_consistency}, df1: {len(index1) - n_consistency}, df2: {len(index2)})")
    else:
        print(f"Datapoints after merge: {len(merged_df)} (both: {n_consistency}, df1: {len(index1) - n_consistency}, df2: {len(index2)})")
    return merged_df

def clean_pipeline(yield_df,
                   group_columns = ["country", "adm1", "adm2", "season"],
                   min_area = 1000,
                   max_yield = 10,
                   min_datapoints = 7,
                   plot=False
                   ):
    """

    :param yield_df: dataframe with yield information
    :param group_columns: group columns for which each individual has only one yield value per year
    :param min_area: filter too small planting areas
    :param max_yield: filter yield with more than this
    :param min_datapoints: minimum number of datapoints for one group
    :param plot: make plots of cleaning steps or not
    :return:
    """
    #assert yield_df.columns == ['country', 'adm1', 'adm2', 'season', 'harv_year', 'area', 'production', 'yield'], "Check your input dataframe. It has not the required columns or unexprected one."

    # 1. Check if yield, production and area are consistent
    yield_values = yield_df["yield"].copy()
    yield_df["yield"] = (yield_df["production"] / yield_df.area)
    yield_inconsistencies = abs(yield_df["yield"] - yield_values) >= 0.05
    if sum(yield_inconsistencies) > 0:
        print(f"Found {sum(yield_inconsistencies)} yield inconsistencies. Check them out: yield_inconsistency_df")
        yield_inconsistency_df = yield_df[yield_inconsistencies]
        yield_inconsistency_df["yield_in_database"] = yield_values[yield_inconsistencies]
        yield_df = yield_df[~yield_inconsistencies]

    # 2. filter crisis events like political instability
    yield_df = yield_df[~((yield_df.country == "Kenya") & (yield_df.harv_year == 1993))] # post-election crisis
    yield_df = yield_df[~((yield_df.country == "Kenya") & (yield_df.harv_year == 2008))] # 2007–2008 Kenyan crisis
    yield_df = yield_df[~((yield_df.country == "Kenya") & (yield_df.harv_year == 2009))] # 2007–2008 Kenyan crisis

    # 3.1 filter based on avg planted area:
    avg_df = yield_df.groupby(group_columns).mean("area").reset_index()
    for _, row in avg_df[avg_df.area < min_area].iterrows():
        too_small = np.all(yield_df[group_columns] == row[group_columns], axis=1)
        yield_df = yield_df[~too_small]
        print(f"Discard {sum(too_small)} datapoints from ({row[group_columns].values}) due to avg planted area of {round(row.area)}ha")

    # 3.2 filter based on planted area for each datapoint
    too_small = yield_df.area < min_area
    print(f"Discard {sum(too_small)} datapoints from due to small planted area (of <{min_area}ha)")
    yield_df = yield_df[~too_small]

    # 4. filter suspicious data (consecutive equal values)
    for group, group_df in yield_df.groupby(group_columns):
        sus_df = detect_suspicious_data(group_df, column="yield", threshold=1e-3, plot=False)
        if len(sus_df) > 0:
            detect_suspicious_data(group_df, column="yield", threshold=1e-3, plot=plot)
            print(f"Discard {len(sus_df)} suspicious 'yield'-values from {group}.")
            yield_df = yield_df.drop(index=sus_df.index)

    # 5. filter unrealistic high yields (before outlier detection since they distort the search with high variance)
    unrealistic_yield = yield_df["yield"] >= max_yield
    if sum(unrealistic_yield):
        print(f"Discard {sum(unrealistic_yield)} datapoints with unrealistic yields (>= {max_yield} t/ha)")
        yield_df = yield_df[~unrealistic_yield]

    # 6.1 filter outliers in the planted area based on Z-scores and 3-sigma margin
    for group, group_df in yield_df.groupby(group_columns):
        outlier_df = detect_outliers_with_polyfit(group_df, column="area", plot=False)
        if len(outlier_df) > 0:
            detect_outliers_with_polyfit(group_df, column="area", plot=plot)
            print(f"Discard {len(outlier_df)} outliers in 'area' from {group} with Z-scores of {outlier_df.z_score.values}")
            yield_df = yield_df.drop(index=outlier_df.index)

    # 6.2 filter outliers in yield based on Z-scores and 3-sigma margin
    for group, group_df in yield_df.groupby(group_columns):
        outlier_df = detect_outliers_with_polyfit(group_df, column="yield", plot=False)
        if len(outlier_df) > 0:
            detect_outliers_with_polyfit(group_df, column="yield", plot=plot)
            outlier_df = outlier_df[outlier_df.z_score > 0]
            if len(outlier_df) > 0:
                print(f"Discard {len(outlier_df)} outliers in 'yield' from {group} with Z-scores of {outlier_df.z_score.values}")
                yield_df = yield_df.drop(index=outlier_df.index)

    # 7. filter regions with few datapoints
    count_datapoints = yield_df.groupby(group_columns).count().reset_index()
    for _, row in count_datapoints[count_datapoints["yield"] < min_datapoints].iterrows():
        too_little_data = np.all(yield_df[group_columns] == row[group_columns], axis=1)
        print(f"Discard {sum(too_little_data)} datapoints from {row[group_columns].values}) due to little number of datapoints in this area.")
        yield_df = yield_df[~too_little_data]

    return yield_df


def clean_pipeline_yield(yield_df,
                         group_columns=["country", "adm1", "adm2", "season"],
                         max_yield=10,
                         min_datapoints = 7,
                         plot=False
                         ):
    """

    :param yield_df: dataframe with yield information
    :param group_columns: group columns for which each individual has only one yield value per year
    :param max_yield: filter yield with more than this
    :param min_datapoints: minimum number of datapoints for one group
    :param plot: make plots of cleaning steps or not
    :return:
    """
    #assert yield_df.columns == ['country', 'adm1', 'adm2', 'season', 'harv_year', 'area', 'production', 'yield'], "Check your input dataframe. It has not the required columns or unexprected one."

    # 1. filter suspicious data (consecutive equal values)
    for group, group_df in yield_df.groupby(group_columns):
        sus_df = detect_suspicious_data(group_df, column="yield", threshold=1e-3, plot=False)
        if len(sus_df) > 0:
            detect_suspicious_data(group_df, column="yield", threshold=1e-3, plot=plot)
            print(f"Discard {len(sus_df)} suspicious 'yield'-values from {group}.")
            yield_df = yield_df.drop(index=sus_df.index)

    # 2. filter unrealistic high yields (before outlier detection since they distort the search with high variance)
    unrealistic_yield = yield_df["yield"] >= max_yield
    if sum(unrealistic_yield):
        print(f"Discard {sum(unrealistic_yield)} datapoints with unrealistic yields (>= {max_yield} t/ha)")
        yield_df = yield_df[~unrealistic_yield]

    # 3 filter outliers in yield based on Z-scores and 3-sigma margin
    for group, group_df in yield_df.groupby(group_columns):
        outlier_df = detect_outliers_with_polyfit(group_df, column="yield", plot=False)
        if len(outlier_df) > 0:
            detect_outliers_with_polyfit(group_df, column="yield", plot=plot)
            outlier_df = outlier_df[outlier_df.z_score > 0]
            if len(outlier_df) > 0:
                print(f"Discard {len(outlier_df)} outliers in 'yield' from {group} with Z-scores of {outlier_df.z_score.values}")
                yield_df = yield_df.drop(index=outlier_df.index)

    # 4. filter regions with few datapoints
    count_datapoints = yield_df.groupby(group_columns).count().reset_index()
    for _, row in count_datapoints[count_datapoints["yield"] < min_datapoints].iterrows():
        too_little_data = np.all(yield_df[group_columns] == row[group_columns], axis=1)
        print(f"Discard {sum(too_little_data)} datapoints from {row[group_columns].values}) due to little number of datapoints in this area.")
        yield_df = yield_df[~too_little_data]

    return yield_df
