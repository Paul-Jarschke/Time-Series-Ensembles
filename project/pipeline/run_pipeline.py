from utils.paths import *
from utils.helpers import vprint

from pipeline.pipe1_data_preprocessing import pipe1_data_preprocessing
from pipeline.pipe2_individual_forecasts import pipe2_individual_forecasts
from pipeline.pipe3_ensemble_forecasts import pipe3_ensemble_forecasts
from pipeline.pipe4_metrics_ranking import pipe4_metrics_ranking


def run_pipeline(df, forecasting_models, ensemble_methods, metrics,
                 date_col='infer', date_format=None, target='infer', covariates='infer', exclude=None,
                 agg_method=None, agg_freq=None,
                 select_individual_models='all', autosarimax_refit_interval=None, forecast_init_train=0.3,
                 select_ensemble_methods='all', ensemble_init_train=0.3,
                 sort_by='MAPE',
                 csv_export=EXPORT_DIR, errors='raise', verbose=False,
                 *args, **kwargs):
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
    vprint("================================================================================="
           "\nStarting  Pipeline...")

    # Pipeline step 1: Perform data preprocessing
    target, covariates = (
        pipe1_data_preprocessing(
            df=df,
            date_col=date_col, date_format=date_format,
            target=target, covariates=covariates, exclude=exclude,
            agg_method=agg_method, agg_freq=agg_freq,
            errors=errors, verbose=verbose,
            *args, **kwargs
        )
    )

    # Pipeline step 2: Compute individual predictions
    individual_predictions = (
        pipe2_individual_forecasts(
            models=forecasting_models, select_models=select_individual_models,
            target=target, covariates=covariates,
            forecast_init_train=forecast_init_train,
            autosarimax_refit_interval=autosarimax_refit_interval,
            csv_export=csv_export, verbose=verbose,
            *args, **kwargs
        )
    )

    # Pipeline step 3: Compute ensemble predictions
    full_predictions = (
        pipe3_ensemble_forecasts(
            individual_predictions=individual_predictions,
            methods=ensemble_methods, select_methods=select_ensemble_methods,
            ensemble_init_train=ensemble_init_train,
            csv_export=csv_export, verbose=verbose,
            *args, **kwargs
        )
    )

    # Pipeline step 4: Ranking by metrics
    metrics_ranking = (
        pipe4_metrics_ranking(
            full_predictions=full_predictions,
            metrics=metrics,
            sort_by=sort_by,
            csv_export=csv_export, verbose=verbose,
            *args, **kwargs
        )
    )

    vprint("\n\nFinished Pipeline!\n"
           "================================================================================="
           )

    # Return results
    return target, covariates, individual_predictions, full_predictions, metrics_ranking
