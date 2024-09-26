import numpy as np
import rasterio
import rasterio.features
from rasterio.enums import Resampling

from config import *


def load_worldcereal_crop_mask(lon_min=None, lat_max=None, lon_max=None, lat_min=None, binary=True):
    """
    Loads the WorldCereal crop mask and optionally crops it based on provided longitude and latitude boundaries.

    The function reads the "worldcereal_crop_mask.tif" raster file from the predefined source directory.
    It retrieves the crop mask layer and metadata such as the coordinate reference system (CRS),
    transform parameters, and the image dimensions. If longitude and latitude boundaries are provided,
    the crop mask is cropped accordingly.

    Args:
        lon_min (float, optional): Minimum longitude of the bounding box. Defaults to None.
        lat_max (float, optional): Maximum latitude of the bounding box. Defaults to None.
        lon_max (float, optional): Maximum longitude of the bounding box. Defaults to None.
        lat_min (float, optional): Minimum latitude of the bounding box. Defaults to None.
        binary (bool, optional): perform classification into 1 (crop) and 0 (no crop) based on 50% threshold

    Returns:
        tuple:
            - crop_mask (numpy.ndarray): 2D array containing the crop mask data, potentially cropped.
            - transform (affine.Affine): Affine transform object representing the pixel-to-world transformation.
            - crs (CRS): Coordinate reference system object of the crop mask.
    """
    crop_mask_path = SOURCE_DATA_DIR / "crop mask/worldcereal_crop_mask.tif"

    # Load the crop mask to determine the target resolution and transform
    with rasterio.open(crop_mask_path) as crop_mask_src:
        if lon_min is not None and lat_max is not None and lon_max is not None and lat_min is not None:
            # Compute the window of the raster to be read based on the geographical bounds
            window = crop_mask_src.window(lon_min, lat_min, lon_max, lat_max)
            crop_mask = crop_mask_src.read(1, window=window)
            transform = crop_mask_src.window_transform(window)
        else:
            crop_mask = crop_mask_src.read(1)
            transform = crop_mask_src.transform

        crs = crop_mask_src.crs

    # make binary map based on 50% threshold
    if binary:
        crop_mask = crop_mask > 50

    return crop_mask, transform, crs


def weighted_avg_over_crop_mask(crop_mask, data_image, instance_name, region_name, warn_spread_above=False):
    # again filter crop mask for nan values in the cc data to respect them in the weighted average
    upd_crop_mask = np.where(np.isnan(data_image), np.nan, crop_mask)

    # check if we have pixels left:
    if np.nansum(upd_crop_mask) == 0:
        return np.nan

    # normalize the crop mask to make it easy to calculate a weighted average based on percentage cropland per pixel
    normalized_crop_mask = upd_crop_mask / np.nansum(upd_crop_mask)

    # calc average
    weighted_avg = np.nansum(normalized_crop_mask * data_image)

    # check for big spread inside one region
    quantiles = np.nanquantile(np.where(upd_crop_mask > 0, data_image, np.nan), q=[0.025, 0.975])
    if warn_spread_above:
        if quantiles[1] - quantiles[0] > warn_spread_above:
            print(f'Detected strong divergence in {instance_name} inside region {region_name}. Quant.: {np.round(quantiles, 2)}, weighted avg: ({np.round(weighted_avg, 2)})')

    return weighted_avg


def load_geoglam_crop_mask(lon_min=None, lat_max=None, lon_max=None, lat_min=None, resolution=None):
    """
    Loads the GEOGLAM crop mask and optionally crops it based on provided longitude and latitude boundaries.

    The function reads the "GEOGLAM_Percent_Maize.tif" raster file from the predefined source directory.
    It retrieves the crop mask layer and metadata such as the coordinate reference system (CRS),
    transform parameters, and the image dimensions. If longitude and latitude boundaries are provided,
    the crop mask is cropped accordingly.

    Args:
        lon_min (float, optional): Minimum longitude of the bounding box. Defaults to None.
        lat_max (float, optional): Maximum latitude of the bounding box. Defaults to None.
        lon_max (float, optional): Maximum longitude of the bounding box. Defaults to None.
        lat_min (float, optional): Minimum latitude of the bounding box. Defaults to None.
        resolution (float, optional): resolution of map in ° (make sure you prepossessed that map using shrink_image())

    Returns:
        tuple:
            - crop_mask (numpy.ndarray): 2D array containing the crop mask data, potentially cropped.
            - transform (affine.Affine): Affine transform object representing the pixel-to-world transformation.
            - crs (CRS): Coordinate reference system object of the crop mask.
    """
    if resolution:
        crop_mask_path = SOURCE_DATA_DIR / f"crop mask/GEOGLAM_Percent_Maize_{resolution}.tif"
    else:
        crop_mask_path = SOURCE_DATA_DIR / "crop mask/GEOGLAM_Percent_Maize.tif"

    # Load the crop mask to determine the target resolution and transform
    with rasterio.open(crop_mask_path) as crop_mask_src:
        if lon_min is not None and lat_max is not None and lon_max is not None and lat_min is not None:
            # Compute the window of the raster to be read based on the geographical bounds
            window = crop_mask_src.window(lon_min, lat_min, lon_max, lat_max)
            crop_mask = crop_mask_src.read(1, window=window)
            transform = crop_mask_src.window_transform(window)
        else:
            crop_mask = crop_mask_src.read(1)
            transform = crop_mask_src.transform

        crs = crop_mask_src.crs

    return crop_mask, transform, crs


def shrink_image(input_path, output_path, factor):
    # Open the source TIFF file
    with rasterio.open(input_path) as src:
        # Calculate the new dimensions
        new_width = src.width // factor
        new_height = src.height // factor
        new_transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )

        # Define the metadata for the output file
        kwargs = src.meta.copy()
        kwargs.update({
            'width': new_width,
            'height': new_height,
            'transform': new_transform
        })

        # Open the destination file
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                # Read the data and resample it
                data = src.read(
                    i,
                    out_shape=(
                        src.count,
                        new_height,
                        new_width
                    ),
                    resampling=Resampling.bilinear
                )
                # Write the resampled data to the destination file
                dst.write(data, i)
