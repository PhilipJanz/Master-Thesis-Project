import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from config import RESULTS_DATA_DIR
from data_assembly import make_dummies
from ml_functions import loyocv
from yield_.yield_functions import load_yield_data

"""
This script trains and predicts very simple benchmark models that can be used to validate other prediction approaches.
Every model is simply based on the year and the region to capture regional averages and trends over time.
It is also used to define yield anomalies (simply the deviation of the benchmark model).  
"""


# load yield data
yield_df = load_yield_data()
yield_df = yield_df[~yield_df.country.isin(["Ethiopia", "Kenya"])]
yield_df = yield_df[yield_df.harv_year > 2001]
yield_df.sort_values(["country", "adm1", "adm2", "harv_year"], inplace=True, ignore_index=True)


# list of benchmark models to pick from
benchmark_models = [LinearRegression(),
                    RandomForestRegressor(random_state=42, n_jobs=-1)]
model_names = ["bench_lin_reg", "bench_rf"]

# columns to be filled with predictions
for model_name in model_names:
    yield_df[model_name] = np.nan

# loop over the countries
for adm, adm_yield_df in yield_df.groupby(["country", "adm1", "adm2"]):  # ["adm1", "adm2"]
    print("\n")

    # first run without any features, just year and admin
    #adm_yield_df["harv_year"
    X = pd.concat([adm_yield_df["harv_year"]], axis=1).values # , adm_yield_df["harv_year"] ** 2
    y = adm_yield_df["yield"]

    for model, model_name in zip(benchmark_models, model_names):
        y_preds, model_mse = loyocv(X, y, years=adm_yield_df["harv_year"], model=model, model_name=f"{adm}, {model_name}", print_result=True)
        if np.any(y_preds < 0):
            y_preds[y_preds < 0] = 0
        yield_df.loc[adm_yield_df.index, model_name] = y_preds

# save
yield_df.to_csv(RESULTS_DATA_DIR / f"benchmark/yield_benchmark_prediction.csv", index=False)

# plot
for adm, adm_yield_df in yield_df.groupby(["country", "adm1", "adm2"]):
    adm = str(adm).replace(", ", "-").replace("'", "").replace(" None", "")
    plt.plot(adm_yield_df.harv_year, adm_yield_df["yield"], label=f"ture yield (var = {np.round(np.var(adm_yield_df['yield']), 3)})")
    plt.plot(adm_yield_df.harv_year, adm_yield_df["bench_lin_reg"], label=f"pred: bench_lin_reg (mse = {np.round(np.mean((adm_yield_df['bench_lin_reg'] - adm_yield_df['yield']) ** 2), 3)})", linestyle="dotted")
    plt.plot(adm_yield_df.harv_year, adm_yield_df["bench_rf"], label=f"pred: bench_rf (mse = {np.round(np.mean((adm_yield_df['bench_rf'] - adm_yield_df['yield']) ** 2), 3)})", linestyle="dotted")
    plt.legend()
    plt.title(f"Benchmarking yield for: {adm}")
    plt.savefig(RESULTS_DATA_DIR / f"benchmark/plots/bench_pred_{adm}", dpi=300)
    plt.close()
