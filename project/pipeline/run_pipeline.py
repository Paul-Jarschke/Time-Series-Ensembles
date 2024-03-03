from utils.paths import *

from pipeline.pipe1_data_preprocessing import pipe1_data_preprocessing
from pipeline.pipe2_individual_forecasts import pipe2_individual_forecasts
from pipeline.pipe3_ensemble_forecasts import pipe3_ensemble_forecasts
from pipeline.pipe4_metrics_ranking import pipe4_metrics_ranking


def run_pipeline(df, forecasting_models, ensemble_methods, metrics,
                 date_col='infer', date_format=None, target='infer', covariates='infer', exclude=None,
                 agg_method=None, agg_freq=None,
                 autosarimax_refit_interval=None,
                 indiv_init_train_ratio=0.3, ens_init_train_ratio=0.3,
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
    if verbose:
        print("=======================")
        print("== Starting Pipeline ==")
        print("=======================")

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
            models=forecasting_models,
            target=target, covariates=covariates,
            indiv_init_train_ratio=indiv_init_train_ratio,
            autosarimax_refit_interval=autosarimax_refit_interval,
            csv_export=csv_export, verbose=verbose,
            *args, **kwargs
        )
    )

    # Pipeline step 3: Compute ensemble predictions
    full_predictions = (
        pipe3_ensemble_forecasts(
            individual_predictions=individual_predictions,
            methods=ensemble_methods,
            ens_init_train_ratio=ens_init_train_ratio,
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

    if verbose:
        print("Finished pipeline!")

    # Return results
    return target, covariates, individual_predictions, full_predictions, metrics_ranking
