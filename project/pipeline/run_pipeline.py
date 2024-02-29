# WORK IN PROGRESS
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


def run_pipeline(df):
    """
    Run pipeline of data preprocessing, individual, and ensemble forecasting, and subsequent model ranking.

    Args:
    - df: input data containing targets and covariates. pandas DataFrame or Series
    - target:
    - covariates:
    - models:
    - init_splits:
    
    Returns:
    - target:
    - covariates:
    - individual_predictions:
    - full_predictions:
    - metrics_ranking:
    """
    # Pipeline step 1: Perform data preprocessing
    target, covariates = pipe1_data_preprocessing(df=df)
    print("Target:", target.name)
    print("Covariates:", ", ".join(covariates.columns))
    
    # Pipeline step 2: Compute individual predictions
    individual_predictions = pipe2_individual_forecasts(models=models, target=target, covariates=covariates, indiv_init_train_ratio=0.3, csv_export=EXPORT_DIR)
    print(individual_predictions, "\n") 
    
    # Pipeline step 3: Compute ensemble predictions
    full_predictions = pipe3_ensemble_forecasts(individual_predictions=individual_predictions, ens_init_train_ratio=0.3, csv_export=EXPORT_DIR)
    print(full_predictions, "\n") 
    
    # Pipeline step 4: Ranking by metrics
    metrics_ranking = pipe4_metrics_ranking(full_predictions=full_predictions, csv_export=EXPORT_DIR)
    print(metrics_ranking, "\n") 
    
    # Return results
    return target, covariates, individual_predictions, full_predictions, metrics_ranking

################
# Run pipeline    
################

# Later in notebook maybe
target, covariates, individual_predictions, full_predictions, metrics_ranking = run_pipeline(df=df)
    
    