

rf_param_grid = {
    'n_estimators': [300], # 100,
    'max_depth': [None, 2, 5, 10, 20],
    'max_features': [1,  'log2', None],
    'min_samples_split': [2, 5, 10],  #
    'min_samples_leaf': [1, 2, 4],
    #'bootstrap': [True, False]
}
