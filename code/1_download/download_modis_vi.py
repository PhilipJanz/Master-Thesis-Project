import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import ee


"""
Before running this script, ensure that you have authenticated your Earth Engine Python API session using 
ee.Authenticate() if it's your first time or if your authentication token has expired.
"""

# ee.Authenticate()
ee.Initialize()


def apply_quality_filter(image):
    # Use the 'SummaryQA' band to mask poor quality pixels.
    # SummaryQA values of 0 are considered good quality
    quality = image.select('SummaryQA').eq(0)
    return image.updateMask(quality)


def download_image_collection(image_collection, geometry, band, folder):
    # Reduce the collection to a list of image IDs (note: do this only if the list is small!)
    image_ids = image_collection.aggregate_array('system:id').getInfo()

    # Now iterate over image IDs client-side
    for image_id in image_ids:
        image = ee.Image(image_id)
        # Apply the quality filter to the individual image
        image = apply_quality_filter(image)
        # Perform operations with the image
        print(image_id)
        task = ee.batch.Export.image.toDrive(
            image=image.select(band).clip(geometry),
            folder=folder,
            fileNamePrefix=image_id + "_" + band,
            crs='EPSG:4326',
            crsTransform=[0.05, 0, 22, 0, -0.05, 15],
            fileFormat='GeoTIFF'
        )
        task.start()
        time.sleep(1)
        while task.status()['state'] == 'RUNNING':
            print('Running...')
            # Perhaps task.cancel() at some point.
            time.sleep(10)
        print('Done.', task.status())


# Define the geographic range using coordinates of a bounding box
# Example coordinates for a rectangular area: (min Lon, min Lat, max Lon, max Lat)
geometry = ee.Geometry.Rectangle([22, -18, 48, 15])

# Define the time range
start_date = '2000-01-01'
end_date = '2020-12-31'

# Load the MOD13C1 dataset
modis = ee.ImageCollection('MODIS/061/MOD13C1').filterDate(start_date, end_date)

download_image_collection(image_collection=modis, geometry=geometry, band='NDVI', folder="MODIS_VI")
download_image_collection(image_collection=modis, geometry=geometry, band='EVI', folder="MODIS_VI")
