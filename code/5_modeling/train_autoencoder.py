import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from sklearn.decomposition import PCA

from feature_selection import AutoencoderFeatureSelector

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_loader import load_yield_data, load_my_cc, load_soil_data
from feature_location import feature_location_dict
from data_assembly import process_list_of_feature_df, make_adm_column, make_X, make_dummies

"""

"""

# LOAD & PREPROCESS ####################################################################################################

# load yield data and benchmark
yield_df = load_yield_data()
yield_df = yield_df[(yield_df.harv_year > 2001) & (yield_df.harv_year < 2023)].reset_index(drop=True)
yield_df = make_adm_column(yield_df)

# load crop calendar (CC)
cc_df = load_my_cc()

# load and process features
length = 10
processed_feature_df_dict = process_list_of_feature_df(yield_df=yield_df, cc_df=cc_df,
                                                       feature_dict=feature_location_dict,
                                                       length=length,
                                                       start_before_sos=30, end_before_eos=60)

# load soil characteristics
soil_df = load_soil_data()
soil_df = make_adm_column(soil_df)
pca = PCA(n_components=2)
soil_df[["soil_1", "soil_2"]] = pca.fit_transform(soil_df[['clay', 'elevation', 'nitrogen', 'phh2o', 'sand', 'silt', 'soc']])
print("explained_variance_ratio_ of PCA on soil:", pca.explained_variance_ratio_)
soil_df = pd.merge(yield_df["adm"], soil_df, on=["adm"], how="left")
soil_df = soil_df[["soil_1", "soil_2"]] # soil_df[['clay', 'elevation', 'nitrogen', 'phh2o', 'sand', 'silt', 'soc']]
# scale each column (soil property) individually
soil_df.iloc[:, :] = StandardScaler().fit_transform(soil_df.values)


# INITIALIZATION #######################################################################################################


# INFERENCE ############################################################################################################

# define year
years = yield_df.harv_year

# prepare predictors # [cluster_yield_df.harv_year] +
predictors_list = [df.loc[yield_df.index] for df in processed_feature_df_dict.values()]
# add soil
# add dummies for regions
predictors_list.append(soil_df.loc[yield_df.index])
predictors_list.append(make_dummies(yield_df))
# form the regressor-matrix X
X, feature_names = make_X(df_ls=predictors_list, standardize=True)
indicator_ix = np.array([np.any([feature_name in feature for feature_name in list(processed_feature_df_dict.keys())]) for feature in feature_names])
X_indicator = X[:, indicator_ix]
indicator_names = feature_names[indicator_ix]

dim_ls = [5, 10, 15, 20]
error_ls = []
ndvi_error_ls = []
for dim in dim_ls:
    ae = AutoencoderFeatureSelector(input_dim=X.shape[1],
                                    encoding_dim=dim,
                                    encoder_layers=[128, 128, 128, dim],
                                    decoder_layers=[128, 128, 128, X_indicator.shape[1]])


    # Train the autoencoder
    print("Training the autoencoder...")
    for batch_size in [10, 20, 30, 50, 80, 100, 200, 400, 1000, 2000]:
        ae.fit(X_train=X, X_target=X_indicator, epochs=1000, batch_size=batch_size, shuffle=True)
    ae.plot_history(start_epoch=300)

    # Transform (encode) the data
    print("Transforming the data...")
    X_ = ae.run(X)
    error_ls.append(np.mean(np.abs(X_ - X_indicator)))
    ndvi_error_ls.append(np.mean(np.abs(X_[:, :10] - X_indicator[:, :10])))

xy = pd.DataFrame((np.abs(X_ - X_indicator)).mean(0), index=indicator_names)





# VISUALIZATION ########################################################################################################
