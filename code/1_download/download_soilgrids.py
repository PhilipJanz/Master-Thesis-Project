import ee

from config import AOI
from gee_functions import download_soilgrids

# Define the geographic range using coordinates of a bounding box
geometry = ee.Geometry.Rectangle(AOI)

download_soilgrids(geometry, folder="SoilGrids")
