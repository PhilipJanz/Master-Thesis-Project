import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / 'data'
SOURCE_DATA_DIR = DATA_DIR / 'raw_source'
POCESSED_DATA_DIR = DATA_DIR / 'processed'