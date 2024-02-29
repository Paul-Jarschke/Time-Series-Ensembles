# run pipeline wrapper function here


# takes df as input
# takes arguments passed to pipe1, pipe2, pipe3, pipe4 ... as input

# WORK IN PROGRESS

# todo: create timestamp folder for data storage
# todo: create log file storing relevant information like hyperparameter settings on top, data properties, and then console output
# todo: Print total time elapsed to console and log(auch in zwischensteps)

# Input: pandas dataframe oder series!
import pandas as pd
from paths import *
from models.forecasting_models import models

from pipe1_data_preprocessing import pipe1_data_preprocessing
from pipe2_individual_forecasts import pipe2_individual_forecasts
from pipe3_ensemble_forecasts import pipe3_ensemble_forecasts
from pipe4_metrics_ranking import pipe4_metrics_ranking

# noch adjusten: 

# Define the directory path and file name
sim_dir = r'C:/Users/Work/OneDrive/GAU/3. Semester/Statistisches Praktikum/Git/NEW_Ensemble_Techniques_TS_FC/project/data/simulations/'
file_name = 'noisy_simdata.csv'

# Combine the directory path and file name
file_path = sim_dir + file_name


# Read and preprocess Dataset
df = pd.read_csv(file_path, index_col = "Date")

# todo add all relevant input args
def run_pipeline(df):
    target, covariates = pipe1_data_preprocessing(df=df)
    # print colname target, covariates
    # Target: ...
    # Covariates: ...
    
    individual_predictions = pipe2_individual_forecasts(models=models, target=target, covariates=covariates, init_train_ratio=0.3, csv_export=EXPORT_DIR)
    print(f"{individual_predictions}\n") 
    
    full_predictions = pipe3_ensemble_forecasts(individual_predictions=individual_predictions, ens_init_train_ratio=0.3, csv_export=EXPORT_DIR)
    print(f"{full_predictions}\n") 
    
    metrics_ranking = pipe4_metrics_ranking(full_predictions=full_predictions, csv_export=EXPORT_DIR)
    print(f"{metrics_ranking}\n") 
    
    return metrics_ranking # return irgendwie alle als objekt? Dictionary, damit auch Plots damit arbeiten können?
    
metrics_ranking = run_pipeline(df=df)
    
    