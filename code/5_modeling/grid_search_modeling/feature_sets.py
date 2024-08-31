
"""
Specify feature sets and their location
"""

# location of features
feature_location_dict = {"ndvi": "remote sensing/smooth_ndvi_regional_matrix.csv",
                         "si-ndvi": "remote sensing/si_smooth_ndvi_regional_matrix.csv",
                         "si-preci-sum": "climate/si_pr_sum_regional_matrix.csv",
                         "pr-max": "climate/pr_max_regional_matrix.csv",
                         "pr-belowP01": "climate/pr_belowP01_regional_matrix.csv",
                         "pr-aboveP99": "climate/pr_aboveP99_regional_matrix.csv",
                         "si-temp-median": "climate/si_tas_median_regional_matrix.csv",
                         "min-temp-median": "climate/tasmin_median_regional_matrix.csv",
                         "max-temp-median": "climate/tasmax_median_regional_matrix.csv",
                         "min-temp-belowP01": "climate/tasmin_belowP01_regional_matrix.csv",
                         "max-temp-aboveP99": "climate/tasmax_aboveP99_regional_matrix.csv"}

relative_feature_set = ["si-ndvi", "si-preci-sum", "pr-belowP01", "pr-aboveP99",
                        "si-temp-median", "min-temp-belowP01", "max-temp-aboveP99"]

absolute_feature_set = ["ndvi", "pr-max", "min-temp-median", "max-temp-median"]

ndvi_feature_set = ["si-ndvi"]

preci_set = ["si-preci-sum", "pr-belowP01", "pr-aboveP99"]

temp_set = ["si-temp-median", "min-temp-belowP01", "max-temp-aboveP99"]

# feature sets to be trained on
feature_sets = {"all": feature_location_dict,
                "relative_features": {key: feature_location_dict[key] for key in relative_feature_set},
                "absolute_features": {key: feature_location_dict[key] for key in absolute_feature_set},
                "ndvi": {key: feature_location_dict[key] for key in ndvi_feature_set},
                "preci": {key: feature_location_dict[key] for key in preci_set},
                "temp": {key: feature_location_dict[key] for key in temp_set}}
