import numpy as np
import pandas as pd
import geopandas as gpd

from config import *
from crop_calendar.crop_calendar_functions import copy_cc, plot_seasonal_crop_calendar, plot_growth_time

"""
This script reads crop calendars (CC) from different sources to merge, clean and plots them. 
"""


# load  administrative boundaries with geographic information
adm_map = gpd.read_file(PROCESSED_DATA_DIR / "admin map/comb_map.shp")
adm_map['centroid'] = adm_map.representative_point()

# view different seasons for each country
yield_df = pd.read_csv(PROCESSED_DATA_DIR / "yield/processed_comb_yield.csv")
print(yield_df.groupby(["country", "season"]).count().reset_index()[["country", "season"]])

# load and prepare ASAP CC
asap_cc = pd.read_csv(SOURCE_DATA_DIR / f"crop calendar/asap_crop_calendar_gaul1.csv", sep=";")
asap_cc = asap_cc[[crop.split(" ")[0] == "Maize" for crop in asap_cc.crop_name]]
asap_cc = asap_cc.rename(columns={"name0_shr": "country", "name1_shr": "adm1", "crop_name": "season"}).drop(["asap0_id", "asap1_id"], axis=1)
# some renaming
asap_cc.loc[asap_cc.adm1 == "North-West", "adm1"] = "North-Western"
asap_cc.loc[asap_cc.adm1 == "West", "adm1"] = "Western"
asap_cc.loc[asap_cc.adm1 == "Central Region", "adm1"] = "Central"
asap_cc.loc[asap_cc.adm1 == "Northern Region", "adm1"] = "Northern"
asap_cc.loc[asap_cc.adm1 == "Southern Region", "adm1"] = "Southern"
asap_cc.loc[asap_cc.adm1 == "Muranga", "adm1"] = "Murang'a"
asap_cc.loc[asap_cc.adm1 == "Keiyo-Marakwet", "adm1"] = "Elgeyo-Marakwet"
asap_cc.loc[asap_cc.adm1 == "Muranga", "adm1"] = "Murang'a"
asap_cc.loc[asap_cc.adm1 == "North Gonder", "adm1"] = "North Gondar"
asap_cc.loc[asap_cc.adm1 == "South Gonder", "adm1"] = "South Gondar"
asap_cc.loc[asap_cc.adm1 == "North Shewa R3", "adm1"] = "North Shewa (OR)"
asap_cc.loc[asap_cc.adm1 == "North Shewa R4", "adm1"] = "North Shewa (AM)"

# correct suspicious crop calendars
#asap_cc.loc[(asap_cc.adm1 == "Baringo") & (asap_cc.season == "Maize (Long rains)"), ["sos_s", "sos_e", "eos_s", "eos_e"]] = [7, 9, 21, 24]

# adding missing regions by merging
# Important: Ethiopia is left out at first because the crop calendar confused adm1 and adm2. It is merged later in the code
my_cc = pd.merge(adm_map[adm_map.country != "Ethiopia"].drop("geometry", axis=1), asap_cc, on=["country", "adm1"], how="left")
print("Missing CC information to be filled:\n", my_cc[np.any(my_cc.isna(), axis=1)][["country", "adm1"]].drop_duplicates())
# add missing information (based on FAO crop calendar)
missing_columns = ["season", "sos_s", "sos_e", "eos_s", "eos_e"]
my_cc.loc[my_cc.adm1 == "Muchinga", missing_columns] = my_cc.loc[my_cc.adm1 == "Eastern", missing_columns].values[0]
my_cc = copy_cc(my_cc, column="adm1", from_name="Lamu", to_name="Tana River")
my_cc = copy_cc(my_cc, column="adm1", from_name="Lamu", to_name="Mandera")
my_cc = copy_cc(my_cc, column="adm1", from_name="Lamu", to_name="Wajir")
my_cc = copy_cc(my_cc, column="adm1", from_name="West Pokot", to_name="Turkana")
my_cc = copy_cc(my_cc, column="adm1", from_name="Baringo", to_name="Embu")
my_cc = copy_cc(my_cc, column="adm1", from_name="Baringo", to_name="Meru")
my_cc = copy_cc(my_cc, column="adm1", from_name="Baringo", to_name="Marsabit")
my_cc = copy_cc(my_cc, column="adm1", from_name="Baringo", to_name="Samburu")
my_cc = copy_cc(my_cc, column="adm1", from_name="Baringo", to_name="Laikipia")
my_cc = copy_cc(my_cc, column="adm1", from_name="Baringo", to_name="Tharaka Nithi")
my_cc = copy_cc(my_cc, column="adm1", from_name="Baringo", to_name="Machakos")
my_cc = copy_cc(my_cc, column="adm1", from_name="Baringo", to_name="Kitui")
my_cc = copy_cc(my_cc, column="adm1", from_name="Baringo", to_name="Makueni")
my_cc = copy_cc(my_cc, column="adm1", from_name="Kakamega", to_name="Busia")
my_cc = copy_cc(my_cc, column="adm1", from_name="Kericho", to_name="Kisii")
my_cc = copy_cc(my_cc, column="adm1", from_name="Kericho", to_name="Nyamira")
my_cc = copy_cc(my_cc, column="adm1", from_name="Kericho", to_name="Bomet")
my_cc = copy_cc(my_cc, column="adm1", from_name="Baringo", to_name="Kajiado")
my_cc = copy_cc(my_cc, column="adm1", from_name="Narok", to_name="Uasin Gishu")
my_cc = copy_cc(my_cc, column="adm1", from_name="Kakamega", to_name="Siaya")
my_cc = copy_cc(my_cc, column="adm1", from_name="Meru", to_name="Nyeri")
my_cc = copy_cc(my_cc, column="adm1", from_name="Meru", to_name="Nyandarua")
my_cc = copy_cc(my_cc, column="adm1", from_name="Nandi", to_name="Murang'a")
my_cc = copy_cc(my_cc, column="adm1", from_name="Nandi", to_name="Kiambu")
my_cc = copy_cc(my_cc, column="adm1", from_name="Nandi", to_name="Nyeri")
my_cc = copy_cc(my_cc, column="adm1", from_name="Nandi", to_name="Nyandarua")
my_cc = copy_cc(my_cc, column="adm1", from_name="Nandi", to_name="Kirinyaga")
my_cc = copy_cc(my_cc, column="adm1", from_name="Tana River", to_name="Kwale")
my_cc = copy_cc(my_cc, column="adm1", from_name="Tana River", to_name="Kilifi")
my_cc = copy_cc(my_cc, column="adm1", from_name="Kisii", to_name="Homa Bay")
my_cc = copy_cc(my_cc, column="adm1", from_name="Kisii", to_name="Migori")
my_cc = copy_cc(my_cc, column="adm1", from_name="Siaya", to_name="Vihiga")
my_cc = copy_cc(my_cc, column="adm1", from_name="Uasin Gishu", to_name="Elgeyo-Marakwet")
my_cc = copy_cc(my_cc, column="adm1", from_name="Kajiado", to_name="Taita Taveta")
assert not np.any(my_cc.isna()), "There is still missing data in 'my_cc' suggesting you missed to specify a case. Check it out!"

# for Ethiopia the CC confuses adm1 and adm2
ethiopia_asap_cc = asap_cc[asap_cc.country == "Ethiopia"].copy().rename(columns={"adm1": "adm2"})
my_ethiopia_cc = pd.merge(adm_map[adm_map.country == "Ethiopia"].drop("geometry", axis=1), ethiopia_asap_cc, on=["country", "adm2"], how="left")
print("Missing CC information to be filled:\n", my_ethiopia_cc[np.any(my_ethiopia_cc.isna(), axis=1)][["country", "adm2"]].drop_duplicates())
filled_cc = my_ethiopia_cc[~np.any(my_ethiopia_cc.isna(), axis=1)]
for _, row in my_ethiopia_cc[np.any(my_ethiopia_cc.isna(), axis=1)].iterrows():
    point1 = my_ethiopia_cc.loc[my_ethiopia_cc.adm2 == row.adm2, "centroid"].values[0]
    ix_nn = np.argmin([point1.distance(point2) for point2 in filled_cc.centroid])
    print("Fill empty (", row.adm2, ") with nearest neighbor:", filled_cc.iloc[ix_nn].adm2)
    my_ethiopia_cc = copy_cc(my_ethiopia_cc, column="adm2", from_name=filled_cc.iloc[ix_nn].adm2, to_name=row.adm2)


# merge the two cc
my_cc = pd.concat([my_cc, my_ethiopia_cc])
assert not np.any(my_cc.isna()), "There is still missing data in 'my_cc' suggesting you missed to specify a case. Check it out!"

# estimate growth time
my_cc["growth_time"] = (my_cc["eos_s"] - my_cc["sos_s"]) % 36

# save
my_cc.drop("centroid", axis=1).to_csv(PROCESSED_DATA_DIR / "crop calendar/processed_crop_calendar.csv", index=False)

# plot
plot_cc_df = pd.merge(adm_map, my_cc.drop("centroid", axis=1), on=["country", "adm1", "adm2"])

plot_seasonal_crop_calendar(plot_cc_df=plot_cc_df.copy(), column="sos_s", file_name="start_of_season")
plot_seasonal_crop_calendar(plot_cc_df=plot_cc_df.copy(), column="eos_e", file_name="end_of_season")
plot_growth_time(plot_cc_df)
