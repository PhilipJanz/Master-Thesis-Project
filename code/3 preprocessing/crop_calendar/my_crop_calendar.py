import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR
from crop_calendar.crop_calendar_functions import make_cc
from maps.map_functions import load_aoi_map

"""
This script creates a crop calendar for each region of interest and compares it to the ASAP crop calendar for validation. 
It is based on the same principle like the ASAP CC but is needed because the ASAP calendar is not covering all regions.
"""

# load administrative boundaries (AOI) with geographic information
adm_map = load_aoi_map()
adm_map['centroid'] = adm_map.representative_point()
adm_map = adm_map[~adm_map.country.isin(["Ethiopia", "Kenya"])]

# load RS data
ndvi_df = pd.read_csv(PROCESSED_DATA_DIR / "remote sensing/smooth_ndvi_regional_matrix.csv", keep_default_na=False)
ndvi_df = ndvi_df[~ndvi_df.country.isin(["Ethiopia", "Kenya"])]
# load precipitation data
preci_df = pd.read_csv(PROCESSED_DATA_DIR / "climate/pr_sum_regional_matrix.csv", keep_default_na=False)
preci_df = preci_df.replace({"": 1.}) # TODO delete when Ethiopia data is there!!!
preci_df = preci_df[~preci_df.country.isin(["Ethiopia", "Kenya"])]

# load ASAP crop calendar (cc) for comparison
asap_cc_df = pd.read_csv(PROCESSED_DATA_DIR / "crop calendar/processed_crop_calendar.csv", keep_default_na=False)
asap_cc_df = asap_cc_df[~asap_cc_df.country.isin(["Ethiopia", "Kenya"])]
#cc_df = cc_df[~cc_df.season.isin(["Maize (Short rains)", "Maize (Maika/Bimodal)", "Maize (Vuli/Bimodal)"])].reset_index(drop=True)
asap_cc_df.iloc[:, 4:] = (asap_cc_df.iloc[:, 4:] - 1) * 10 + 1

make_cc(asap_cc_df=asap_cc_df,
                                         ndvi_df=ndvi_df, preci_df=preci_df,
                                         plot=True)
# adm_map["sos"], adm_map["eos"] =