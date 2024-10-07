from config import PROCESSED_DATA_DIR
from standardized_indices.si_functions import generate_si

"""
This script generated standardized indices (si) for detecting extreme values in 
NDVI, precipitaiton & temperature
by respecting local and temporal differences. 
Each value is brought into relation with values from the same region and the same day of year.
"""

# generate and automatically save ni values for all values in the
generate_si(folder_path=PROCESSED_DATA_DIR / "remote sensing/",
            file_name="cleaned_ndvi_regional_matrix.csv",
            si_file_name="../../../data/processed/remote sensing/svi_regional_matrix.csv",
            gaussian_rolling_averge_window_size=5,
            distibution="gaussian")
generate_si(folder_path=PROCESSED_DATA_DIR / "climate/",
            file_name="preci_regional_matrix.csv",
            si_file_name="../../../data/processed/climate/spi1_regional_matrix.csv",
            gaussian_rolling_averge_window_size=30,
            distibution="gamma"
            )
generate_si(folder_path=PROCESSED_DATA_DIR / "climate/",
            file_name="preci_regional_matrix.csv",
            si_file_name="../../../data/processed/climate/spi6_regional_matrix.csv",
            gaussian_rolling_averge_window_size=180,
            distibution="gamma"
            )
generate_si(folder_path=PROCESSED_DATA_DIR / "climate/",
            file_name="temp_regional_matrix.csv",
            si_file_name="../../../data/processed/climate/sti_regional_matrix.csv",
            distibution="gaussian")
