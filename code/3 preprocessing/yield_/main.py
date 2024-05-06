from yield_.yield_functions import *

from config import *

ethiopia_yield_df = read_ethiopia_yield()
ethiopia_yield_df = clean_pipeline(ethiopia_yield_df, plot=False)
ethiopia_yield_df.head()

malawi_yield_df = read_malawi_yield()
malawi_yield_df = clean_pipeline(malawi_yield_df, plot=False)
malawi_yield_df.head()

tanzania_yield_df = read_tanzania_yield()
tanzania_yield_df = clean_pipeline(tanzania_yield_df, plot=False)
tanzania_yield_df.head()

kenya_and_zambia_yield_df = read_kenya_and_zambia_yield()
kenya_and_zambia_yield_df = clean_pipeline(kenya_and_zambia_yield_df, plot=False)
kenya_and_zambia_yield_df.head()

#### SAVE ####
comb_df = pd.concat([malawi_yield_df, tanzania_yield_df, kenya_and_zambia_yield_df, ethiopia_yield_df])
comb_df.to_csv(POCESSED_DATA_DIR / "yield/processed_comb_yield.csv", index=False)

adm_df = comb_df.groupby(["country", "adm1", "adm2"]).count().reset_index()[["country", "adm1", "adm2"]]
adm_df.to_csv(POCESSED_DATA_DIR / "yield/available_admin.csv", index=False)
