import pickle

import numpy as np
from sklearn.decomposition import PCA

import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import PROCESSED_DATA_DIR
from data_loader import load_soil_data, load_yield_data, make_adm_column

"""
Applied PCA to reduce dimensionality and correlation in soil data
"""

# number of principal components that are used to make soil-features (using PCA)
soil_pc_number = 2

# load yield data and benchmark
yield_df = load_yield_data()

# load soil characteristics
soil_df = load_soil_data()
soil_df = make_adm_column(soil_df)
pca = PCA(n_components=soil_pc_number)
soil_df[["soil_1", "soil_2"]] = pca.fit_transform(soil_df[['clay', 'nitrogen', 'phh2o', 'sand', 'silt', 'soc']])
print("explained_variance_ratio_ of PCA on soil:", pca.explained_variance_ratio_)
soil_df = pd.merge(yield_df["adm"], soil_df, on=["adm"], how="left")
soil_df = soil_df[["soil_1", "soil_2"]] # soil_df[['clay', 'elevation', 'nitrogen', 'phh2o', 'sand', 'silt', 'soc']]
# scale each column (soil property) individually
soil_df.iloc[:, :] = StandardScaler().fit_transform(soil_df.values)

# save
soil_df.to_csv(PROCESSED_DATA_DIR / f"features/soil_pca_{soil_pc_number}.csv", index=False)
