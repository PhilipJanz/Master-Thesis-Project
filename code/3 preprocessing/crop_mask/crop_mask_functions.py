import rasterio
import rasterio.features

from config import *

def load_geoglam_crop_mask(lon_min=None, lat_max=None, lon_max=None, lat_min=None):
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

    Returns:
        tuple:
            - crop_mask (numpy.ndarray): 2D array containing the crop mask data, potentially cropped.
            - transform (affine.Affine): Affine transform object representing the pixel-to-world transformation.
            - crs (CRS): Coordinate reference system object of the crop mask.
    """
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
