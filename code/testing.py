import matplotlib.pyplot as plt
import numpy as np

opti_alpha = np.exp(np.mean(np.log(best_alphas))) # np.median(best_alphas) # opti.best_params["alpha"]
alphas = np.exp(np.arange(-20.8, 0, 0.2))

train_mse = []
test_mse = []
for alpha in alphas:
    opti.best_params["alpha"] = alpha
    X_train_trans, new_predictor_names, trained_model = opti.train_best_model(X=X_train, y=y_train,
                                                                          predictor_names=predictor_names)

    # prepare test-data
    X_test_trans, _ = opti.transform_X(X=X_test, predictor_names=predictor_names, params=opti.best_params)

    # predict train- & test-data
    y_pred_train = trained_model.predict(X_train_trans)
    y_pred = trained_model.predict(X_test_trans)

    # write the predictions into the result df
    train_mse.append(np.mean((y_pred_train - y_train) ** 2))
    test_mse.append(np.mean((y_pred - y_test) ** 2))

opti_alpha = alphas[np.argmin(np.abs((np.max(train_mse) / 2 + np.min(train_mse) / 2) - train_mse))]

plt.plot(alphas, test_mse, label=f"test mse (min at {alphas[np.argmin(test_mse)]})")
plt.plot(alphas, train_mse, label=f"train mse (min at {alphas[np.argmin(train_mse)]})")
plt.hlines(y_test ** 2, xmin=alphas[0], xmax=alphas[-1], color="red", linestyles="dotted", label="zero-predictor",  alpha=0.8)
plt.vlines(opti_alpha, ymin=np.min([test_mse, train_mse]), ymax=np.max([test_mse, train_mse]), color="tab:green", linestyles="dashed", label=f"optuna best parameter: {opti_alpha}",  alpha=0.8)
plt.xlabel("alpha")
plt.ylabel("mse")
plt.title(f"LOYOCV test on {cluster_name} hold-out: {year_out}")
#plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()




opti_alpha = opti.best_params["alpha"]
alphas = np.exp(np.arange(-20.8, 0, 0.1))

train_mse = []
test_mse = []
for alpha in alphas:
    opti.best_params["alpha"] = alpha
    X_train_trans, new_predictor_names, trained_model = opti.train_best_model(X=X_train,
                                                                              y=y_train,
                                                                              predictor_names=predictor_names)

    # prepare test-data
    X_test_trans, _ = opti.transform_X(X=X_test, predictor_names=predictor_names, params=opti.best_params)

    # predict train- & test-data
    y_pred_train = trained_model.predict(X_train_trans)
    y_pred = trained_model.predict(X_test_trans)

    # write the predictions into the result df
    train_mse.append(np.mean((y_pred_train - y_train) ** 2))
    test_mse.append(np.mean((y_pred - y_test) ** 2))

plt.plot(alphas, test_mse, label=f"test mse (min at {alphas[np.argmin(test_mse)]})")
plt.plot(alphas, train_mse, label=f"train mse (min at {alphas[np.argmin(train_mse)]})")
plt.hlines(y_test ** 2, xmin=alphas[0], xmax=alphas[-1], color="red", linestyles="dotted", label="zero-predictor",  alpha=0.8)
plt.vlines(opti_alpha, ymin=np.min([test_mse, train_mse]), ymax=np.max([test_mse, train_mse]), color="tab:green", linestyles="dashed", label=f"optuna best parameter: {opti_alpha}",  alpha=0.8)
plt.xlabel("alpha")
plt.ylabel("mse")
plt.title(f"LOYOCV test on {cluster_name} hold-out: {year_out}")
#plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()



best_alphas = []
# LOYOCV - leave one year out cross validation
for year_out_alpha in np.unique(years_train):
    # if year_out==2011:
    #    break
    year_out_bool = (years_train == year_out_alpha)

    # splitt the data
    X_train_alpha, y_train_alpha = X_train[~year_out_bool], y_train[~year_out_bool]
    X_test_alpha, y_test_alpha = X_train[year_out_bool], y_train[year_out_bool]

    test_mse = []
    for alpha in alphas:
        opti.best_params["alpha"] = alpha
        X_train_trans, new_predictor_names, trained_model = opti.train_best_model(X=X_train_alpha,
                                                                                  y=y_train_alpha,
                                                                                  predictor_names=predictor_names)

        # prepare test-data
        X_test_trans, _ = opti.transform_X(X=X_test_alpha, predictor_names=predictor_names, params=opti.best_params)

        # predict train- & test-data
        y_pred = trained_model.predict(X_test_trans)

        test_mse.append(np.mean((y_pred - y_test_alpha) ** 2))

    best_alphas.append(alphas[np.argmin(test_mse)])

print(np.mean(best_alphas))
print(np.median(best_alphas))


cluster_set = "adm"

corr_mtx = []
adm_ls = []
for cluster_name, cluster_yield_df in cluster_yield_df.groupby("adm"):
    corr_ls = []
    feature_ls = []
    for feature, processed_feature_df in processed_feature_df_dict.items():
        for feature_num in processed_feature_df.columns:
            feature_values = processed_feature_df.loc[cluster_yield_df.index, feature_num].values
            corr = np.corrcoef(np.vstack([feature_values, cluster_yield_df["yield_anomaly"].values]))[0, 1]
            corr_ls.append(corr)
            feature_ls.append(feature_num)
    corr_mtx.append(corr_ls)
    adm_ls.append(cluster_name)
corr_df = pd.DataFrame(corr_mtx, columns=feature_ls, index=adm_ls)

corr_df.mean()
xy = pd.DataFrame({"rolling new": corr_df.mean(), "rolling": roll_corr_df.mean(), "quadratic": quad_corr_df.mean(), "linear": old_corr_df.mean()}, index=corr_df.columns).transpose()



kmean_elbow(data_mtx=corr_df_imputed.iloc[:, :10], max_k=41)
labels, _ = kmean_cluster(data_mtx=corr_df_imputed.iloc[:, :10], n_clusters=20)
yield_df = pd.merge(yield_df, pd.DataFrame({"adm": corr_df.index, "corr_cluster_20": labels}), on="adm")

