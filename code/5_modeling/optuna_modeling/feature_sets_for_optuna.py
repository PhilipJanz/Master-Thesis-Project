
"""
Specify feature sets and their location
"""

# location of features
feature_location_dict = {"ndvi": "remote sensing/smooth_ndvi_regional_matrix.csv",
                         "si-ndvi": "remote sensing/si_smooth_ndvi_regional_matrix.csv",
                         "preci-max": "climate/pr_max_regional_matrix.csv",
                         "preci-sum": "climate/pr_sum_regional_matrix.csv",
                         "si-preci-sum": "climate/si_pr_sum_regional_matrix.csv",
                         "preci-cdd": "climate/pr_cddMax_regional_matrix.csv",
                         "preci-belowP01": "climate/pr_belowP01_regional_matrix.csv",
                         "preci-aboveP99": "climate/pr_aboveP99_regional_matrix.csv",
                         "si-temp-median": "climate/si_tas_median_regional_matrix.csv",
                         "min-temp-median": "climate/tasmin_median_regional_matrix.csv",
                         "max-temp-median": "climate/tasmax_median_regional_matrix.csv",
                         "min-temp-belowP01": "climate/tasmin_belowP01_regional_matrix.csv",
                         "max-temp-aboveP99": "climate/tasmax_aboveP99_regional_matrix.csv"}

ndvi_feature_set = ["ndvi", "si-ndvi"]

preci_set = ["preci-max", "preci-sum", "si-preci-sum", "preci-cdd", "preci-belowP01", "preci-aboveP99"]

temp_set = ["min-temp-median", "max-temp-median", "si-temp-median", "min-temp-belowP01", "max-temp-aboveP99"]

abs_set = ["ndvi", "preci-sum", "preci-max", "preci-cdd", "min-temp-median", "max-temp-median"]

rel_set = ["si-ndvi", "si-preci-sum", "pr-belowP01", "pr-aboveP99", "si-temp-median", "min-temp-belowP01", "max-temp-aboveP99"]

soil_set = ['clay', 'elevation', 'nitrogen', 'phh2o', 'sand', 'silt', 'soc']

# feature sets to be trained on
feature_sets = {"year": ["harv_year"],
                "soil": soil_set,
                "ndvi": ndvi_feature_set,
                "preci": preci_set,
                "temp": temp_set,
                }

# test if all features are mentioned properly. Typos can happen ;)
for feature_set_name, feature_set in feature_sets.items():
    if feature_set_name in ["soil", "year"]:
        continue
    for feature in feature_set:
        assert feature in feature_location_dict.keys(), f"Feature '{feature}' without location reference."


# TODO del

#feature_location_dict = {"si-ndvi": "remote sensing/si_smooth_ndvi_regional_matrix.csv"}
