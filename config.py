from pathlib import Path

"""
Define global paths & constants here.
"""

# always use this seed when random numbers are generated (especially during modeling & hyperparameter optimization)
# this guarantees reproducibility of any results
SEED = 42

# path structure
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / 'data'
GRAPHICS_DIR = BASE_DIR / 'graphics'
SOURCE_DATA_DIR = DATA_DIR / 'raw_source'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
FEATURE_SELECTION_DIR = DATA_DIR / 'feature_selection'
RESULTS_DATA_DIR = DATA_DIR / 'modeling_results'

RUN_DIR = RESULTS_DATA_DIR / 'yield_predictions'

# colors & markers for consistent visualization
TANZANIA_COLOR = "#ca8a60"
ZAMBIA_COLOR = "#94d6c7"
MALAWI_COLOR = "#ada4e2"
COUNTRY_COLORS = {"Zambia": ZAMBIA_COLOR,
                  "Malawi": MALAWI_COLOR,
                  "Tanzania": TANZANIA_COLOR}
COUNTRY_MARKERS = {"Tanzania": "o",
                   "Zambia": "v",
                   "Malawi": "p"}


# (area of interest) coordinates for a rectangular area: (min Lon, min Lat, max Lon, max Lat)
AOI = [22, -18, 40.5, -1]

# time range of interest (dependend on available yield data)
START_DATE = '2000-01-01'
END_DATE = '2023-12-31'
