import ee

from config import AOI
from gee_functions import download_worldcereal_cm

# Define the geographic range using coordinates of a bounding box
geometry = ee.Geometry.Rectangle(AOI)

download_worldcereal_cm(geometry=geometry, folder="Worldcereal")
