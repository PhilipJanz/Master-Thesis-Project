# Modeling Maize Yield in Sub-Saharan Africa Using Machine
Learning: Overcoming Data Scarcity with Transfer Learning
This project units all methods needed to apply maize yield modeling, using various models. It was developed to predict maize yields for Tanzania, Malawi and Zambia.
The structure allows to add other countries as well. In this case the following steps should guide the user to accomplish this goal.
The order of those steps is generally important since the processing steps are dependent on each  other.

All scripts are written in python and use a specific set of packages. The requirements can be found in requirements_cpu.txt and requirements_gpu.txt
Those files correspond to two different python environments that are necessary for CPU and GPU tasks.
Note that the GPU environment is exclusively for tensorflow deep learning (transfer_learning_source_model_cnn.py). 
While, all other scripts should be executed using the cpu environment.

It is recommended to use a conda environment to construct the python environments. 
Run conda in the terminal to create a ready-to-use environment:

    conda create --name cpu_env_name --file requirements_cpu.txt
The GPU environment is a bit more tricky, because it requires a lot of different installations that vary depending on the operating system.
Therefore, the user is guided to https://www.tensorflow.org/install/pip for a detailed instruction to install tensorflow using CPU or GPU. 
After successfully installing tensorflow run this to install all other packages required:
    
    pip install -r requirements_gpu.txt

## 1. Data Collection
### Yield Data
Download and save yield data into the folder data/raw_source/yield. This code was designed especially for Tanzania, Malawi & Zambia.
Be free to add new data from different countries or crops. When you do that, take a look at the config.py file that organizes global variables, like area of interest.
Those variables are crucial because they are uses by all different kinds of scripts through this repository.
E.g. the coordinates of area of interest are used as filters when downloading maps. 
### All other Data
Find the collection of scripts to download all other nesessary data using Google Earth Engine, including precipitation, temperature, vegetation condition, soil profiles, crop mask.... 
This is highly recommended since it incorporates data preprocessing that is done for free on Google servers. 
Data gets stored in Google Drive, which means you might need to pay a couple of $ to get extra space.

Before running those scripts, ensure that you have authenticated your Earth Engine Python API session using 
ee.Authenticate() if it's your first time or if your authentication token has expired.

## 2. Exploration
Play around in a jupyter-notebook to look at your data. This section is not meant to influence any other code.

## 3. Data Processing
### 3.1 Yield Data
Execute yield_preprocessing.py and make sure that you define functions for all available data. 
Just follow the example of the others. This script saves the processed yield data needed in the next step.

The script also applies filters on yield timeseries to clean them. Set plot=True when calling clean_pipeline() to see
what is going on under the hood. 

### 3.2 Map Data Preparation
Execute admin_map_preparation.py the defines geografic boundaries for the administrative units. Watch out: there might be 
different spellings for the same district. Be inspired by the code to handle those inconsistencies. 

The script also visualizes the number of yield datapoints per region.

### 3.3 Soil Data Preprocessing
Execute soil_data_preparation.py to apply map data for the first time to process SoilGrids soil properties.

### 3.4 Meterological & Remote Sensing Data Aggregation
Execute rs_data_preparation.py, precipitation_data_preparation.py and temperature_data_preparation.py for applying regional and crop mask to 
calculate averged features from grid data.
Those scripts even incorporate the histogram approach by You et al. (2017) which was not further followed in this repository.

### 3.5 Crop Calendar
After the NDVI data is processed, it can be used to estimate the costmized crop calendar for each region. 
Adjust the NDVI-threshold to get the best fit with the FAO crop calendar. 
The function make_cc() will tell you how many regions respect the boundaries of FAO CC.

### 3.6 Feature Standardization 
Execute si_generation.py to process standardized indices for precipitation (SPI), temperature (STI) and NDVI (SVI)

### 3.7 Feature Engineering 
Scripts for preprocessing and aggregating features, ready to be used for inference 
- pca_on_soil_data.py: Dimensional reduction on soil profiles
- process_designed_features.py: Features for conventional regression models 
- process_time_series.py: Harmonized feature matrix for deep learning (CNN)

## 4. Feature Selection
This script select_features.py executes feature selection based on a selected feature set (produced by process_designed_features()) using MI-VIF.
Change the VIF threshold if required (the lower, the fewer features and less multicollinearity)

Visualize feature frequency selection using selection_frequency_visualization.py

## 5. Modeling
All modeling (except the benchmark) involves hyperparameter optimization that is 
organized by the Optuna framework. The scripts optuna_optimizer.py and optuna_optimizer_tf.py 
are used to initialize an optuna study and set up the hyperparameter spaces. 

On top of that, there is an infrastructure that supports long optimization processes.
When optimizing a complex model the entire process might take days. It gets very unfeasible
to execute all of this in one run. The computer might unexpectedly shut down and all progress is lost.
To avoid this, all modeling approaches are organized as a 'run' (see run.py), which represents
a single approach that predicts each datapoint in the yield dataset exactly ones. 
The run stores all modeling results, so it knows where to continue, in case it got interrupted.

### 5.1 Benchmark Model
Run the benchmark.py script to produce benchmark predictions based on the raw
yield time series and nothing else. 
### 5.2 Scikit-learn & XGBoost
Now the fun part begins: the models do the work, and you get a coffee.
Start the script prediction.py and initialize a set of variables that determine
one run, like model type, feature set, feature selection, optimization parameters...

Look into optuna_optimization.py to find the settings for hyperparameters.

### 5.3 Deep Learning (CNN)
Execute transfer_learning_source_model_cnn.py which is very similar to prediction.py, with some changes.
The hyperparameter search takes a lot of time. 
It is very likely that your script won't finish the desired amount of trails before the timeout stops it. 
Don't worry: in this case, simple run the script again. For each optuna study (for each hold-out-year) the script searches for existing 
studies and load them if possible. It will continue the study until the desired number of trials is reached. 

The trained CNN will save its predictions and additionally saves the last hidden layer as transferable features in the folder transfer_features.
Those files are necessary to later perform transfer learning in prediction_using_transfer_model.py

Additionally, the script makes plots of the training process of the final model for out-of-sample prediction and transfer.
It visualizes the progression of train- and test-error, which gives a nice insight about potential model overfitting (when test error rises after a certain number of epoch).
Those plots help to understand model performance. They should not be used to set hyperparameters. Hyperparameters should only be determined by the LOYOCV on the training data.

### 5.4 Feature transfer
Given a complete run of a CNN modeling approach as explained above, you're ready to start prediction_using_transfer_model.py.
It will train a model exclusively on transferred- and static features.

You can choose between external and internal transfer. The internal model got trained on all data and the external transfer builds on a leave-on-country-out approach, where the source model was not trained on any data form the county that is the target for transfer learning.

## 6. Metrics & Visualizations
Until now, you probably collected a number of modeling runs that you'd like to compare and see if they can beat the benchmark model.

For individual run inspection the script performance_visualization.py offers a collection of plots that give quick insights into model performance.
Be aware that those scripts were developed for East-Africa. Fundamental adjustments might bring up unexpected errors. 

## X. Feedback
Feedback is gold. If you find errors or wrong descriptions in the code, please leave me a comment. 
