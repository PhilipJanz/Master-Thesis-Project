from config import PROCESSED_DATA_DIR
from standardized_indices.si_functions import generate_si

"""
This script generated standardized indices (si) for detecting extreme values in 
NDVI, EVI, precipitaiton & temperature
by respecting local and temporal differences. 
Each value is brought into relation with values from the same region and the same day of year.
"""

# generate and automatically save ni values for all values in the
generate_si(folder_path=PROCESSED_DATA_DIR / "remote sensing/",
            file_name="smooth_ndvi_regional_matrix.csv")
generate_si(folder_path=PROCESSED_DATA_DIR / "remote sensing/",
            file_name="smooth_evi_regional_matrix.csv")
generate_si(folder_path=PROCESSED_DATA_DIR / "climate/",
            file_name="tas_median_regional_matrix.csv")
generate_si(folder_path=PROCESSED_DATA_DIR / "climate/",
            file_name="pr_sum_regional_matrix.csv",
            min_avg=1)
