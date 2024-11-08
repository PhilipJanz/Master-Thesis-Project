import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR, SOURCE_DATA_DIR
from crop_calendar.crop_calendar_functions import make_cc, plot_my_crop_calendar, plot_season_length
from maps.map_functions import load_aoi_map

"""
This script creates a crop calendar for each region of interest and compares it to the FAO crop calendar for validation. 
It is based on the same principle like the ASAP CC but is needed because the ASAP calendar is not covering all regions.
This approach is applying Worldcereal crop mask for maize and is therefore crop specific.
"""


# load administrative boundaries (AOI) with geographic information
adm_map = load_aoi_map()
adm_map['centroid'] = adm_map.representative_point()

# load RS data
ndvi_df = pd.read_csv(PROCESSED_DATA_DIR / "remote sensing/cleaned_ndvi_regional_matrix.csv", keep_default_na=False)

# load precipitation data
preci_df = pd.read_csv(PROCESSED_DATA_DIR / "climate/preci_regional_matrix.csv", keep_default_na=False)

# load FAO crop calendar (cc) for comparison
fao_cc_df = pd.read_csv(SOURCE_DATA_DIR / "crop calendar/FAO/crop_calendar.csv", keep_default_na=False)

adm_map["sos"], adm_map["eos"] = make_cc(fao_cc_df=fao_cc_df,
                                         ndvi_df=ndvi_df, preci_df=preci_df,
                                         threshold=.5,
                                         plot=True)

adm_map["season_length"] = adm_map["eos"] - adm_map["sos"]
adm_map.loc[adm_map.season_length < 0, "season_length"] = adm_map.loc[adm_map.season_length < 0, "season_length"] + 365

# visualize
plot_my_crop_calendar(adm_map)
plot_season_length(adm_map)

# save
adm_map[["country", "adm1", "adm2", "sos", "eos", "season_length"]].to_csv(PROCESSED_DATA_DIR / f"crop calendar/my_crop_calendar.csv", index=False)
