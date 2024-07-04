import numpy as np

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

# unite all yield data
comb_df = pd.concat([malawi_yield_df, tanzania_yield_df, kenya_and_zambia_yield_df, ethiopia_yield_df])

# Yield anomalies processing
comb_df["rel_yield_anomaly"] = np.nan
comb_df["yield_anomaly"] = np.nan
for adm, adm_df in comb_df.groupby(["country", "adm1", "adm2"]):
    # calc rolling average as the 'expected value'
    rolling_avg = adm_df["yield"].rolling(window=5, min_periods=1, center=True).mean()

    # calc anomalies
    comb_df.loc[adm_df.index, "rel_yield_anomaly"] = adm_df["yield"] / rolling_avg
    comb_df.loc[adm_df.index, "yield_anomaly"] = adm_df["yield"] - rolling_avg


#### SAVE ####
comb_df.to_csv(PROCESSED_DATA_DIR / "yield/processed_comb_yield.csv", index=False)

adm_df = comb_df.groupby(["country", "adm1", "adm2"]).count().reset_index()[["country", "adm1", "adm2"]]
adm_df.to_csv(PROCESSED_DATA_DIR / "yield/available_admin.csv", index=False)
