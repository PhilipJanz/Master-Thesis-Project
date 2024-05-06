from config import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

from maps.map_functions import merge_regions

"""
The following code prepares a geopandas dataframe that delivers geometric data (shapely.geometry.multipolygon.MultiPolygon)
that represents the map of a certain area. 
The areas of interest are determined by the availability of yield data. See code/3 preprocessing/yield_/main.py for 
generating the file available_admin.csv that will be used here.
"""

# 1. read information on areas of interest dependent on yield data availability
adm_df = pd.read_csv(POCESSED_DATA_DIR / "yield/available_admin.csv", keep_default_na=False)

# 2. Load shapefiles for selected countries
# Tanzania provides adm1
tanzania_map = gpd.read_file(SOURCE_DATA_DIR / 'admin map/tanzania_maps/tza_admbnda_adm1_20181019.shp')
tanzania_map["country"] = "Tanzania"
tanzania_map["adm1"] = tanzania_map.ADM1_EN
tanzania_map["adm2"] = "None"
tanzania_map = merge_regions(tanzania_map, column="adm1", regions=["Songwe", "Mbeya"], new_region_name="Mbeya")
print("Check the following regions that were not found in the yield data: ", [region for region in tanzania_map.adm1.values if region not in adm_df.adm1.values])

# Kenya provides adm1
kenya_map = gpd.read_file(SOURCE_DATA_DIR / 'admin map/kenya_maps/ken_admbnda_adm1_iebc_20191031.shp')
kenya_map["country"] = "Kenya"
kenya_map["adm1"] = kenya_map.ADM1_EN
kenya_map["adm2"] = "None"
kenya_map.loc[kenya_map.adm1 == "Tharaka-Nithi", "adm1"] = "Tharaka Nithi"
print("Check the following regions that were not found in the yield data: ", [region for region in kenya_map.adm1.values if region not in adm_df.adm1.values])

# Malawi provides adm2
malawi_map = gpd.read_file(SOURCE_DATA_DIR / 'admin map/malawi_maps/mwi_admbnda_adm2_nso_hotosm_20230405.shp')
malawi_map["country"] = "Malawi"
malawi_map["adm1"] = malawi_map.ADM1_EN
malawi_map["adm2"] = malawi_map.ADM2_EN
malawi_map.loc[malawi_map.adm2 == "Nkhatabay", "adm2"] = "Nkhata Bay"
malawi_map = merge_regions(malawi_map, column="adm2", regions=["Mzuzu City", "Mzimba"], new_region_name="Mzimba")
malawi_map = merge_regions(malawi_map, column="adm2", regions=["Lilongwe City", "Lilongwe"], new_region_name="Lilongwe")
malawi_map = merge_regions(malawi_map, column="adm2", regions=["Zomba City", "Zomba"], new_region_name="Zomba")
malawi_map = merge_regions(malawi_map, column="adm2", regions=["Blantyre City", "Blantyre"], new_region_name="Blantyre")
print("Check the following regions that were not found in the yield data: ", [region for region in malawi_map.adm2.values if region not in adm_df.adm2.values])

# Ethiopia provides adm2
ethiopia_map = gpd.read_file(SOURCE_DATA_DIR / 'admin map/ethiopia_maps/eth_admbnda_adm2_csa_bofedb_2021.shp')
ethiopia_map["country"] = "Ethiopia"
ethiopia_map["adm1"] = ethiopia_map.ADM1_EN
ethiopia_map["adm2"] = ethiopia_map.ADM2_EN
ethiopia_map.loc[ethiopia_map.adm2 == "Awsi /Zone 1", "adm2"] = "Zone 1"
ethiopia_map.loc[ethiopia_map.adm2 == "North Wello", "adm2"] = "North Wollo"
ethiopia_map.loc[ethiopia_map.adm2 == "South Wello", "adm2"] = "South Wollo"
ethiopia_map.loc[ethiopia_map.adm2 == "Wag Hamra", "adm2"] = "Wag Himra"
ethiopia_map.loc[ethiopia_map.adm1 == "Benishangul Gumz", "adm1"] = "Benishangul Gumuz"
ethiopia_map.loc[ethiopia_map.adm1 == "SNNP", "adm1"] = "SNNPR"
ethiopia_map.loc[ethiopia_map.adm2 == "Guji", "adm2"] = "Gujii"
ethiopia_map.loc[ethiopia_map.adm2 == "Horo Gudru Wellega", "adm2"] = "Horo Guduru"
ethiopia_map.loc[ethiopia_map.adm2 == "Ilu Aba Bora", "adm2"] = "Ilubabor"
ethiopia_map.loc[ethiopia_map.adm2 == "Kelem Wellega", "adm2"] = "Kelem"
ethiopia_map.loc[ethiopia_map.adm2 == "Central", "adm2"] = "Central Tigray"
ethiopia_map.loc[ethiopia_map.adm2 == "Eastern", "adm2"] = "East Tigray"
ethiopia_map.loc[ethiopia_map.adm2 == "North Western", "adm2"] = "Northwest Tigray"
ethiopia_map.loc[ethiopia_map.adm2 == "Southern", "adm2"] = "South Tigray"
ethiopia_map.loc[ethiopia_map.adm2 == "Western", "adm2"] = "West Tigray"
ethiopia_map.loc[ethiopia_map.adm2 == "Liban", "adm2"] = "Liben"
ethiopia_map.loc[ethiopia_map.adm2 == "Yem Special", "adm2"] = "Yem"
ethiopia_map.loc[ethiopia_map.adm2 == "Kamashi", "adm2"] = "Kemashi"
ethiopia_map.loc[ethiopia_map.adm2 == "Mao Komo Special", "adm2"] = "Mao-Komo"
ethiopia_map.loc[ethiopia_map.adm2 == "Konta Special", "adm2"] = "Konta"
ethiopia_map.loc[ethiopia_map.adm2 == "Kefa", "adm2"] = "Keffa"
ethiopia_map.loc[ethiopia_map.adm2 == "Guraghe", "adm2"] = "Gurage"
ethiopia_map.loc[ethiopia_map.adm2 == "Dawuro", "adm2"] = "Dawro"
ethiopia_map.loc[ethiopia_map.adm2 == "Bench Sheko", "adm2"] = "Bench Maji"
ethiopia_map = merge_regions(ethiopia_map, column="adm2", regions=["Agnewak", "Agniwak"], new_region_name="Agniwak")
ethiopia_map = merge_regions(ethiopia_map, column="adm2", regions=["Assosa", "Asosa"], new_region_name="Asosa")
ethiopia_map = merge_regions(ethiopia_map, column="adm2", regions=["Gamo", "Gofa"], new_region_name="Gamo Gofa")
print("Check the following regions that were not found in the yield data: ", [region for region in ethiopia_map.adm2.values if region not in adm_df.adm2.values])

# Zambia provides adm2
zambia_map = gpd.read_file(SOURCE_DATA_DIR / 'admin map/zambia_maps/zmb_admbnda_adm2_dmmu_20201124.shp')
zambia_map["country"] = "Zambia"
zambia_map["adm1"] = zambia_map.ADM1_EN
zambia_map["adm2"] = zambia_map.ADM2_EN
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Chasefu", "Lundazi"], new_region_name="Lundazi")
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Chembe", "Mansa"], new_region_name="Mansa")
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Chisamba", "Chibombo"], new_region_name="Chibombo")
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Chitambo", "Serenje"], new_region_name="Serenje") # 2012
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Luano", "Mkushi"], new_region_name="Mkushi") # 2012
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Ngabwe", "Kapiri Mposhi"], new_region_name="Kapiri Mposhi") # 2012
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Chipangali", "Kasenengwa", "Chipata"], new_region_name="Chipata") # 2018
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Lumezi",  "Lundazi"], new_region_name="Lundazi") # 2018
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Sinda", "Petauke"], new_region_name="Petauke") # 2012
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Vubwi", "Chadiza"], new_region_name="Chadiza") # 2012
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Chipili", "Mwense"], new_region_name="Mwense") # 2012
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Lunga", "Chifunabuli", "Samfya"], new_region_name="Samfya") #
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Mwansabombwe", "Kawambwa"], new_region_name="Kawambwa") # 2012
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Chilanga", "Kafue"], new_region_name="Kafue") # 2012
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Chirundu", "Siavonga"], new_region_name="Siavonga") # 2012
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Rufunsa", "Chongwe"], new_region_name="Chongwe") # 2012
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Shibuyunji", "Lusaka"], new_region_name="Lusaka") # 2012
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Kanchibiya", "Lavushimanda", "Mpika"], new_region_name="Mpika") # 2017
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Ikelenge", "Mwinilunga"], new_region_name="Mwinilunga") # 2011
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Manyinga", "Kabompo"], new_region_name="Kabompo") # 2012
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Kalumbila","Mushindano", "Solwezi"], new_region_name="Solwezi") # 2016
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Lunte District", "Mporokoso"], new_region_name="Mporokoso") # 2017
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Lupososhi", "Luwingu"], new_region_name="Luwingu") # 2018
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Nsama", "Kaputa"], new_region_name="Kaputa") #
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Mpulungu", "Senga Hill", "Mbala"], new_region_name="Mbala") # 2016
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Chikankanta", "Mazabuka"], new_region_name="Mazabuka") # 2011
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Pemba", "Choma"], new_region_name="Choma") # 2012
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Zimba", "Kalomo"], new_region_name="Kalomo") # 2012
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Limulunga", "Mongu"], new_region_name="Mongu") # 2012
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Mitete", "Lukulu"], new_region_name="Lukulu") # 2012
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Mulobezi", "Sesheke"], new_region_name="Sesheke") # 2013
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Mwandi", "Sesheke"], new_region_name="Sesheke") # 2012
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Nalolo", "Senanga"], new_region_name="Senanga") # 2012
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Nkeyema", "Luampa", "Kaoma"], new_region_name="Kaoma") # 2012
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Shang'ombo"], new_region_name="Shangombo") # rename
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Sikongo", "Kalabo"], new_region_name="Kalabo") # 2012
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Sioma", "Shangombo"], new_region_name="Shangombo") # 2012
zambia_map = merge_regions(zambia_map, column="adm2", regions=["Shiwamg'andu", "Chinsali"], new_region_name="Chinsali") # 2013
#rename
zambia_map.loc[zambia_map.adm2 == "Milengi", "adm2"] = "Milenge"
zambia_map.loc[zambia_map.adm2 == "Chiengi", "adm2"] = "Chienge"
zambia_map.loc[zambia_map.adm2 == "Itezhi-tezhi", "adm2"] = "Itezhi-Tezhi"
zambia_map.loc[zambia_map.adm2 == "Itezhi-Tezhi", "adm1"] = "Southern"
print([region for region in zambia_map.adm2.values if region not in adm_df.adm2.values])

#### COMBINING AND SAVING #####
comb_map = pd.concat([ethiopia_map, tanzania_map, kenya_map, malawi_map, zambia_map])[["country", "adm1", "adm2", "geometry"]]
comb_map = pd.merge(comb_map, adm_df, how="right", on=["country", "adm1", "adm2"])
assert not np.any(comb_map.geometry.isna()), f"There is missing geometry information for: {adm_df[comb_map.geometry.isna()]}"
# make plot TODO gets much better
comb_map.plot()
plt.savefig(POCESSED_DATA_DIR / "admin map/plots/plain_comb_map.jpg", dpi=600)
# save geodata
comb_map.to_file(POCESSED_DATA_DIR / "admin map/comb_map.shp", index=False)
