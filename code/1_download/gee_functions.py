import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import ee


# ee.Authenticate()
ee.Initialize()


def apply_quality_filter(image):
    # Use the 'SummaryQA' band to mask poor quality pixels.
    # SummaryQA values of 0 are considered good quality
    quality = image.select('SummaryQA').eq(0)
    return image.updateMask(quality)


def apply_crop_mask(image, geometry):
    # Load the WorldCereal crop mask for maize and select the 'classification' band
    crop_mask = ee.ImageCollection("ESA/WorldCereal/2021/MODELS/v100") \
        .filter(ee.Filter.eq('product', 'maize')) \
        .mosaic() \
        .select('classification')  # Select the classification band

    # Crop mask values: 0 (non-maize areas), 100 (maize areas). Convert to binary mask
    crop_mask = crop_mask.neq(0)

    # Clip the crop mask to the area of interest
    crop_mask = crop_mask.clip(geometry)

    # Apply the crop mask to the MODIS image (where maize is present)
    return image.updateMask(crop_mask)


def download_image_collection(image_collection, geometry, band, folder, scale, modis_quality_filter=False):
    # Reduce the collection to a list of image IDs (note: do this only if the list is small!)
    image_ids = image_collection.aggregate_array('system:id').getInfo()

    # Now iterate over image IDs client-side
    for image_id in image_ids:
        image = ee.Image(image_id)
        if modis_quality_filter:
            # Apply the quality filter to the individual image
            image = apply_quality_filter(image)
        # Apply the crop mask to the image
        #image = apply_crop_mask(image, geometry)
        # Perform operations with the image
        print(image_id)
        task = ee.batch.Export.image.toDrive(
            image=image.select(band).clip(geometry),
            folder=folder,
            fileNamePrefix=image_id + "_" + band,
            crs='EPSG:4326',
            #crsTransform=[0.05, 0, 22, 0, -0.05, 15],
            scale=scale,
            fileFormat='GeoTIFF'
        )
        task.start()
        time.sleep(5)
        while task.status()['state'] == 'RUNNING':
            print('Running...')
            # Perhaps task.cancel() at some point.
            time.sleep(1)
        print('Done.', task.status())


def download_worldcereal_cm(geometry, folder):
    # Load the WorldCereal crop mask for maize and select the 'classification' band
    crop_mask = ee.ImageCollection("ESA/WorldCereal/2021/MODELS/v100") \
        .filter(ee.Filter.eq('product', 'maize')) \
        .mosaic() \
        .select('classification') \
        .clip(geometry)

    # Perform operations with the image
    task = ee.batch.Export.image.toDrive(
        image=crop_mask,
        folder=folder,
        fileNamePrefix="worldcereal_crop_mask",
        crs='EPSG:4326',
        scale=250,
        #crsTransform=[0.05, 0, 22, 0, -0.05, 15],
        fileFormat='GeoTIFF'
    )
    task.start()
    time.sleep(1)
    while task.status()['state'] == 'RUNNING':
        print('Running...')
        # Perhaps task.cancel() at some point.
        time.sleep(5)
    print('Done.', task.status())


def download_soilgrids(geometry, folder, soil_properties=["sand", "silt", "clay", "soc", "nitrogen", "phh2o"]):
    for property in soil_properties:
        # Load the SoilGrids data
        soil_map = ee.Image(f"projects/soilgrids-isric/{property}_mean").clip(geometry)

        # Perform operations with the image
        task = ee.batch.Export.image.toDrive(
            image=soil_map,
            folder=folder,
            fileNamePrefix=property,
            crs='EPSG:4326',
            scale=250,
            #crsTransform=[0.05, 0, 22, 0, -0.05, 15],
            fileFormat='GeoTIFF'
        )
        task.start()
        time.sleep(1)
        while task.status()['state'] == 'RUNNING':
            print('Running...')
            # Perhaps task.cancel() at some point.
            time.sleep(5)
        print('Done.', task.status())
