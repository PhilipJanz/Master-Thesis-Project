from pathlib import Path

SEED = 42

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / 'data'
SOURCE_DATA_DIR = DATA_DIR / 'raw_source'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
FEATURE_SELECTION_DIR = DATA_DIR / 'feature_selection'
RESULTS_DATA_DIR = DATA_DIR / 'modeling_results'

RUN_DIR = RESULTS_DATA_DIR / 'yield_predictions'

# colors
TANZANIA_COLOR = "#2ca02c"
ZAMBIA_COLOR = "#9467bd"
KENYA_COLOR = "#1f77b4"
MALAWI_COLOR = "#d62728"
ETHIOPIA_COLOR = "#ff7f0e"
COUNTRY_COLORS = {"Tanzania": TANZANIA_COLOR,
                  "Zambia": ZAMBIA_COLOR,
                  "Malawi": MALAWI_COLOR}
COUNTRY_MARKERS = {"Tanzania": "o",
                   "Zambia": "v",
                   "Malawi": "p"}


# (area of interest) coordinates for a rectangular area: (min Lon, min Lat, max Lon, max Lat)
AOI = [22, -18, 40.5, -1]

# time range of interest (dependend on available yield data)
START_DATE = '2000-01-01'
END_DATE = '2023-12-31'
