import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xarray as xr

from config import SOURCE_DATA_DIR, PROCESSED_DATA_DIR

"""
This script is a small cosy libray to load and allocate the climate data provided by PIK with following structure;
Under the 'climate' folder you find folders for each country, eg. MWI (Malawi)
The next level of folders specify the dekade and origin of the data, eg. ERA5_Malawi_MWI_04_21_04_30 (ERA5 data and 21.-30. April)
Inside there are files where each contains maps of one feature for all possible years, eg. tas_mean_ERA5_historical_1981-2022_MWI_04_21_04_30_Malawi.nc (mean temperature for the years 1981 until 2022)
Data-dimensions: {'time': 42, 'bnds': 2, 'longitude': 15, 'latitude': 35}
"""


def load_one_file(file_path, get_transform=True):
    data = xr.open_dataset(file_path)
    if get_transform:
        # Enable the rio accessor
        data = data.rio.write_crs("EPSG:4326")
        # ERA5 data have an offset in the coordinate points (the are the edge of the pixel instead of the center)
        if "ERA5" in str(file_path):
            data = data.assign_coords(longitude=data.longitude.values + 0.125)
            data = data.assign_coords(latitude=data.latitude.values - 0.125)
        # check of flipped axis:
        if data.latitude.values[0] > data.latitude.values[-1]:
            data = data.sel(latitude=slice(None, None, -1))
        # Get the affine transform
        transform = data.rio.transform()
        return data, transform
    else:
        return data


def make_dekadal_data_matrix(adm_df, first_year, last_year):
    # Create a DataFrame copy
    data_matrix = adm_df.copy()

    # Generate year, month, and dekadal combinations using list comprehensions
    columns = [f"{year}_{month:02d}_{day}" for year in range(first_year, last_year + 1)
               for month in range(1, 13)
               for day in [10, 20, 30]]

    # Create a new DataFrame with NaN values for all new columns
    new_columns_df = pd.DataFrame(np.nan, index=adm_df.index, columns=columns)

    # Concatenate the new columns DataFrame with the original data_matrix
    data_matrix = pd.concat([data_matrix, new_columns_df], axis=1)

    return data_matrix


def calculate_geo_boundaries(dims, transform, round):
    lon_min = transform[2]
    lat_min = transform[5]
    lon_max = lon_min + transform[0] * dims["longitude"]
    lat_max = lat_min + transform[4] * dims["latitude"]
    return np.round([lon_min, lon_max, lat_min, lat_max], round)


def calculate_weighted_average(image, mask):
    """

    :param image: any image
    :param mask: a mask same size of the image with weights adding up to 1
    :return: weighted average
    """
    return np.nansum(image * mask)


def calculate_weighted_average_on_xarray_images(images, mask):
    weighted_averages = xr.apply_ufunc(
        calculate_weighted_average,  # the function to apply
        images,                # the DataArray to apply the function to
        kwargs={'mask': mask},  # additional arguments to the function
        input_core_dims=[['latitude', 'longitude']],  # dimensions of input to be processed
        vectorize=True,               # whether to vectorize the function
        dask='parallelized',          # enables parallel processing with Dask
        output_dtypes=[images.dtype]  # specify the output data type
    )
    return weighted_averages


def save_data_matrices(data_matrix_dict):
    features = data_matrix_dict.keys()
    for feature in features:
        data_matrix_dict[feature].to_csv(PROCESSED_DATA_DIR / f"climate/{feature}_regional_matrix.csv", index=False)
