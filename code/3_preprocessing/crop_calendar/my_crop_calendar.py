import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR
from crop_calendar.crop_calendar_functions import make_cc, plot_my_crop_calendar, plot_season_length
from maps.map_functions import load_aoi_map

"""
This script creates a crop calendar for each region of interest and compares it to the ASAP crop calendar for validation. 
It is based on the same principle like the ASAP CC but is needed because the ASAP calendar is not covering all regions.
"""

# load administrative boundaries (AOI) with geographic information
adm_map = load_aoi_map()
adm_map['centroid'] = adm_map.representative_point()

# load RS data
ndvi_df = pd.read_csv(PROCESSED_DATA_DIR / "remote sensing/smooth_ndvi_regional_matrix.csv", keep_default_na=False)

# load precipitation data
preci_df = pd.read_csv(PROCESSED_DATA_DIR / "climate/pr_sum_regional_matrix.csv", keep_default_na=False)
preci_df = preci_df[~preci_df.country.isin(["Ethiopia", "Kenya"])]

# load ASAP crop calendar (cc) for comparison
asap_cc_df = pd.read_csv(PROCESSED_DATA_DIR / "crop calendar/processed_crop_calendar.csv", keep_default_na=False)
asap_cc_df = asap_cc_df[~asap_cc_df.country.isin(["Ethiopia", "Kenya"])]
asap_cc_df.iloc[:, 4:] = (asap_cc_df.iloc[:, 4:] - 1) * 10 + 1

adm_map["sos"], adm_map["eos"] = make_cc(asap_cc_df=asap_cc_df,
                                         ndvi_df=ndvi_df, preci_df=preci_df,
                                         plot=False)

adm_map["season_length"] = adm_map["eos"] - adm_map["sos"]
adm_map.loc[adm_map.season_length < 0, "season_length"] = adm_map.loc[adm_map.season_length < 0, "season_length"] + 365

# visualize
plot_my_crop_calendar(adm_map)
plot_season_length(adm_map)

# save
adm_map[["country", "adm1", "adm2", "sos", "eos", "season_length"]].to_csv(PROCESSED_DATA_DIR / f"crop calendar/my_crop_calendar.csv", index=False)
