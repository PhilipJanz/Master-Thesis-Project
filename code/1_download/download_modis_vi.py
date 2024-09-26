import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import ee

from config import AOI, START_DATE, END_DATE
from gee_functions import download_image_collection

"""
Before running this script, ensure that you have authenticated your Earth Engine Python API session using 
ee.Authenticate() if it's your first time or if your authentication token has expired.
"""

# Define the geographic range using coordinates of a bounding box
# Example coordinates for a rectangular area: (min Lon, min Lat, max Lon, max Lat)
geometry = ee.Geometry.Rectangle(AOI)

# Load the MOD13C1 dataset
modis = ee.ImageCollection('MODIS/061/MOD13Q1').filterDate(START_DATE, END_DATE)

download_image_collection(image_collection=modis, geometry=geometry, band='NDVI', folder="MODIS_NDVI", scale=250,
                          modis_quality_filter=True)
#download_image_collection(image_collection=modis, geometry=geometry, band='EVI', folder="MODIS_VI")
