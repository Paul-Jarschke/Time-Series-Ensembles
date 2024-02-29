from paths import *
from models.forecasting_models import models

from pipeline.pipe1_data_preprocessing import pipe1_data_preprocessing
from pipeline.pipe2_individual_forecasts import pipe2_individual_forecasts
from pipeline.pipe3_ensemble_forecasts import pipe3_ensemble_forecasts
from pipeline.pipe4_metrics_ranking import pipe4_metrics_ranking

def run_pipeline(df, verbose=False):
    """
    Run pipeline of data preprocessing, individual, and ensemble forecasting, and subsequent model ranking.

    Args:
    - df: pandas DataFrame or Series, input data containing targets and covariates. 
    - verbose: bool, whether to print intermediate steps (default: False).
    - target:
    - covariates:
    - models:
    - init_splits:
    - verbose:
    
    Returns:
    - target:
    - covariates:
    - individual_predictions:
    - full_predictions:
    - metrics_ranking:
    """
    if verbose:
        print("=======================")
        print("== Starting Pipeline ==")
        print("=======================")
        
    # Pipeline step 1: Perform data preprocessing
    target, covariates = pipe1_data_preprocessing(df=df, verbose=verbose)
    
    # Pipeline step 2: Compute individual predictions
    individual_predictions = pipe2_individual_forecasts(models=models, target=target, covariates=covariates, indiv_init_train_ratio=0.3, csv_export=EXPORT_DIR, verbose=verbose)
    
    # Pipeline step 3: Compute ensemble predictions
    full_predictions = pipe3_ensemble_forecasts(individual_predictions=individual_predictions, ens_init_train_ratio=0.3, csv_export=EXPORT_DIR, verbose=verbose)
    
    # Pipeline step 4: Ranking by metrics
    metrics_ranking = pipe4_metrics_ranking(full_predictions=full_predictions, csv_export=EXPORT_DIR, verbose=verbose)
    
    if verbose:
        print("Finished pipeline!")
    
    # Return results
    return target, covariates, individual_predictions, full_predictions, metrics_ranking

################
# Run pipeline    
################

# Later in notebook maybe
#target, covariates, individual_predictions, full_predictions, metrics_ranking = run_pipeline(df=df, verbose=True)
    
    