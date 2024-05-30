from pathlib import Path

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / 'data'
SOURCE_DATA_DIR = DATA_DIR / 'raw_source'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# colors
TANZANIA_COLOR = "#2ca02c"
ZAMBIA_COLOR = "#9467bd"
KENYA_COLOR = "#1f77b4"
MALAWI_COLOR = "#d62728"
ETHIOPIA_COLOR = "#ff7f0e"
COUNTRY_COLORS = {"Ethiopia": ETHIOPIA_COLOR,
                  "Kenya": KENYA_COLOR,
                  "Tanzania": TANZANIA_COLOR,
                  "Zambia": ZAMBIA_COLOR,
                  "Malawi": MALAWI_COLOR}
