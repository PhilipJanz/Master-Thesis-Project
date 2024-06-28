
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

ndvi_feature_set = ["ndvi", "si-ndvi"]

preci_set = ["pr-max", "si-preci-sum", "pr-belowP01", "pr-aboveP99"]

temp_set = ["min-temp-median", "max-temp-median", "si-temp-median", "min-temp-belowP01", "max-temp-aboveP99"]

soil_set = ['clay', 'elevation', 'nitrogen', 'phh2o', 'sand', 'silt', 'soc']

# feature sets to be trained on
feature_sets = {"soil": soil_set,
                "ndvi": ndvi_feature_set,
                "preci": preci_set,
                "temp": temp_set}
