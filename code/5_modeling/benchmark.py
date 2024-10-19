import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from metrics import calc_r2, calc_rmse

from config import BASE_DIR, PROCESSED_DATA_DIR

from data_loader import load_yield_data

"""
This script is ...
"""

# INITIALIZATION #######################################################################################################

# define objective (target)
objective = "yield"

# choose model from ["linear", "quadratic", "knn"]
model_type = "knn"

# LOAD & PREPROCESS ####################################################################################################

# load yield data and benchmark
yield_df = load_yield_data()
# init benchmark prediction column
yield_df["y_bench"] = np.nan

# INFERENCE ############################################################################################################

for adm, adm_yield_df in yield_df.groupby("adm"):

    # define target and year
    y = adm_yield_df[objective]
    X = adm_yield_df.harv_year - np.mean(adm_yield_df.harv_year)

    if model_type == "quadratic":
        X = pd.concat([X, X ** 2], axis=1)

    # LOYOCV - leave one year out cross validation
    for year_out in X.index:
        year_out_bool = (X.index == year_out)

        X_train, y_train = X[~year_out_bool].values, y[~year_out_bool].values
        X_test, y_test = X[year_out_bool].values, y[year_out_bool].values

        # train model
        if model_type in "linear":
            X_train = X_train.reshape(1, -1).T
            X_test = X_test.reshape(1, -1).T
            model = LinearRegression()
        elif model_type == "quadratic":
            model = LinearRegression()
        elif model_type == "knn":
            X_train = X_train.reshape(1, -1).T
            X_test = X_test.reshape(1, -1).T
            if X_train.shape[0] < 5:
                model = KNeighborsRegressor(n_neighbors=2)
            else:
                model = KNeighborsRegressor(n_neighbors=4)
        else:
            raise AssertionError('Model should be in ["linear", "quadratic", "knn"]')

        trained_model = model.fit(X=X_train, y=y_train)

        # predict test-data
        y_pred_test = trained_model.predict(X_test)

        yield_df.loc[adm_yield_df.index[year_out_bool], "y_bench"] = y_pred_test

bench_rmse = calc_rmse(y_pred=yield_df["y_bench"], y_true=yield_df[objective])
print(f"Benchmark model '{model_type}' acieves RMSE: {np.round(bench_rmse, 3)}")

yield_df.to_csv(PROCESSED_DATA_DIR / "yield/processed_comb_yield_and_benchmark.csv", index=False)

# VISUALIZATION ########################################################################################################
