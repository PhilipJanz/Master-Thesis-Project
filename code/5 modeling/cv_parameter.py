

rf_param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [None, 10, 20, 30],
    'max_features': [1,  'sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
