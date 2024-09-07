import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import ee

from config import AOI
from gee_functions import download_image_collection

"""
Before running this script, ensure that you have authenticated your Earth Engine Python API session using 
ee.Authenticate() if it's your first time or if your authentication token has expired.
"""

# Define the geographic range using coordinates of a bounding box
# Example coordinates for a rectangular area: (min Lon, min Lat, max Lon, max Lat)
geometry = ee.Geometry.Rectangle(AOI)

# Define the time range
start_date = '2000-01-01'
end_date = '2023-12-31'

# Load the MOD13C1 dataset
modis = ee.ImageCollection('MODIS/061/MOD13Q1').filterDate(start_date, end_date)

download_image_collection(image_collection=modis, geometry=geometry, band='NDVI', folder="MODIS_NDVI")
#download_image_collection(image_collection=modis, geometry=geometry, band='EVI', folder="MODIS_VI")
