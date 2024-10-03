# Master-Thesis-Project

## 1. Data Collection
### Yield Data
Download and save yield data into the folder data/raw_source/yield

## 3. Data Processing
### 3.1 Yield Data
Execute yield_preprocessing.py and make sure that you define functions for all available data. 
Just follow the example of the others. This script saves the processed yield data needed in the next step.

The script also applies filters on yield timeseries to clean them. Set plot=True when calling clean_pipeline() to see
what is going on under the hood. 

### 3.2 Map Data Preparation
Execute admin_map_preparation.py the defines geografic boundaries for the adminastrative units. Watch out: there might be 
different spellings for the same district. Be inspired by the code to handle those inconsistencies. 

The script also visualizes the number of yield datapoints per region.

### 3.3 Soil Data Preprocessing
Execute soil_data_preparation.py to apply map data for the first time to process SoilGrids soil properties.

### 3.4 PIK Climate Data
Execute precipitation_data_preparation.py and temperature_data_preparation.py for applying regional and crop mask to 
calculate averged features from map data.