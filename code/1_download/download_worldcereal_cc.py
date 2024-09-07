import ee

from config import AOI
from gee_functions import download_worldcereal_cc

# Define the geographic range using coordinates of a bounding box
geometry = ee.Geometry.Rectangle(AOI)

download_worldcereal_cc(geometry=geometry, folder="Worldcereal")
