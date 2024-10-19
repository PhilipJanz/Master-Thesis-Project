import numpy as np
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from yield_.yield_functions import *

from config import *

malawi_yield_df = read_malawi_yield()
malawi_yield_df = clean_pipeline(malawi_yield_df, group_columns=["country", "adm1", "adm2"], plot=False)
malawi_yield_df.head()

tanzania_yield_df = read_tanzania_yield()
tanzania_yield_df = clean_pipeline(tanzania_yield_df, plot=False)
tanzania_yield_df.head()

kenya_and_zambia_yield_df = read_kenya_and_zambia_yield()
kenya_and_zambia_yield_df = clean_pipeline(kenya_and_zambia_yield_df, plot=False)
kenya_and_zambia_yield_df.head()

# unite all yield data
comb_df = pd.concat([malawi_yield_df, tanzania_yield_df, kenya_and_zambia_yield_df])
comb_df = comb_df[~comb_df.country.isin(["Kenya"])]
comb_df = comb_df[comb_df.harv_year > 2000]
comb_df.sort_values(["country", "adm1", "adm2", "harv_year"], inplace=True, ignore_index=True)

# Yield anomalies processing
comb_df["standardized_yield"] = np.nan
pvalue_ls = []
for adm, adm_yield_df in comb_df.groupby(["country", "adm1", "adm2"]):
    # make linear and quadratic fit of yield data to capture trends
    X_lin = adm_yield_df["harv_year"].values.reshape(-1, 1)
    X_quad = np.hstack([X_lin, X_lin ** 2])
    model = LinearRegression()
    linear_fit = model.fit(X=X_lin, y=adm_yield_df["yield"].values).predict(X_lin)
    quadratic_fit = model.fit(X=X_quad, y=adm_yield_df["yield"].values).predict(X_quad)
    #n_years = adm_yield_df["harv_year"].nunique()
    #rolling_fit = adm_yield_df["yield"].rolling(window=int(n_years/4), center=False, min_periods=1, win_type='gaussian').mean(std=2)

    # cal p-values
    pvalue = sm.OLS(adm_yield_df["yield"].values, sm.add_constant(X_lin, prepend=False)).fit().pvalues[0]
    pvalue_ls.append(pvalue)


    # calc standardization
    detrended_yield = adm_yield_df["yield"] - linear_fit
    comb_df.loc[adm_yield_df.index, "standardized_yield"] = detrended_yield / np.std(detrended_yield)

    # plot
    adm = str(adm).replace(", ", "-").replace("'", "").replace(" None", "")
    plt.plot(adm_yield_df.harv_year, adm_yield_df["yield"],
             label=f"yield")
    plt.plot(adm_yield_df.harv_year, linear_fit,
             label=f"linear fit",
             linestyle="dotted")
    plt.plot(adm_yield_df.harv_year, quadratic_fit,
             label=f"quadratic fit",
             linestyle="dotted")
    #plt.plot(adm_yield_df.harv_year, rolling_fit,
    #         label=f"rolling avg",
    #         linestyle="dotted")
    plt.legend()
    plt.title(f"Trend in yield time series for: {adm}")
    plt.savefig(PROCESSED_DATA_DIR / f"yield/plots/yield_and_trends_{adm}", dpi=300)
    plt.close()

    """
    # check yield anomaly variance:
    yield_anomaly_var = np.var(comb_df.loc[adm_yield_df.index, "yield_anomaly"])
    if yield_anomaly_var < 0.01:
        print(f"Discard {adm} for negligible yield-anomaly variance of {np.round(yield_anomaly_var, 5)}")
        comb_df.drop(index=adm_yield_df.index, inplace=True)
    """


#### SAVE ####
#comb_df = comb_df[comb_df.harv_year > 2001]
comb_df.to_csv(PROCESSED_DATA_DIR / "yield/processed_comb_yield.csv", index=False)

adm_df = comb_df.groupby(["country", "adm1", "adm2"]).count().reset_index()[["country", "adm1", "adm2"]]
adm_df.to_csv(PROCESSED_DATA_DIR / "yield/available_admin.csv", index=False)
