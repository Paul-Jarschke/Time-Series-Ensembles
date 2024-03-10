from math import ceil

import numpy as np
import pandas as pd

from src.utils.helpers import vprint, csv_exporter
from src.utils.predictions.ensemble_predictions_wrapper import ensemble_prediction_wrapper


def pipe3_ensemble_forecasts(
        forecaster_results, ensemblers,
        ensemble_init_train,
        fh=None,
        select_ensemblers='all',
        export_path=None, verbose=False,
        *args, **kwargs):
    """
    Perform ensemble forecasts using the provided individual predictions and ensemble approaches.

    Parameters:
    -----------
    forecaster_results : tuple containing pandas DataFrames
        Tuple containing results from historical and future predictions of the individual forecasters. The
        predictions are each a pandas DataFrames. Each column contains a model's prediction. Index containing dates
        should be of pandas PeriodIndex format.
    ensemblers : dict
        Dictionary containing ensemble methods and their corresponding models for ensemble forecasting. The keys
        represent the ensemble methods (e.g., 'weighted', 'meta') and the values are dictionaries where keys are model
        names and values are dictionaries containing model information like 'model', 'package', and 'options'.
    ensemble_init_train : float
        Initial ensemblers' training set size as a fraction of individual forecasters' predictions.
    fh : int, optional
        When provided, pipeline not only performs historical evaluation of forecasters and ensemblers but also
        provides out-of-sample future predictions along the whole provided forecast horizon.
    select_ensemblers : str or list, optional
        Specify which ensemblers to use (default: 'all').
    export_path : os.PathLike, optional
        Path to export_path the ensemble predictions as a CSV file. If not provided, no CSV file is exported (default: None).
    verbose : bool, optional
        If True, prints detailed information about the individual forecasting process and stores results in log file
        (default: False).
    *args:
            Additional positional arguments.
    **kwargs:
        Additional keyword arguments.

    Returns:
    --------
        full_historical_predictions : pandas.DataFrame
            DataFrame containing both historical individual and ensemble predictions per model.
        full_historical_predictions : pandas.DataFrame
            DataFrame containing both future individual and ensemble predictions per model (if desired, otherwise:
            None).

    Notes:
    ------
    This function performs ensemble forecasting using the provided individual predictions and ensemble methods.
    It iterates over each ensemble approach and model within the ensemblers dictionary, generates historical 
    one-step-ahead predictions using the individual predictions as covariates. Finally, it merges them with the 
    individual predictions. If export_path path is provided, it exports these predictions as CSV file.
    """

    # Verbose print to indicate that the historical ensemble predictions start
    vprint('\n==================================================='
           '\n== Pipeline Step 3: Ensemble Models\' Predictions =='
           '\n===================================================\n')

    # Unpack tuple
    historical_individual_predictions = forecaster_results[0]
    future_individual_predictions = forecaster_results[1]

    # Ensemble Train Split
    # Determine size of the training set for ensemble forecasting
    n_predictions = len(historical_individual_predictions)
    ens_init_trainsize = ceil(ensemble_init_train * n_predictions)
    # At this period ensemble training ends end ensemble forecast is produced for ens_init_trainsize + 1

    # Calculate the forecast horizon for ensemble forecasters
    H_ensemble = n_predictions - ens_init_trainsize

    # Printing information about prediction process
    vprint(f'Splitting individual forecast data (n = {n_predictions}) for training of ensemblers (train/test ratio: '
           f'{int(ensemble_init_train * 100)}/{int(100 - ensemble_init_train * 100)})...',
           f'Initial training set has {ens_init_trainsize} observations and goes from '
           f'{historical_individual_predictions.index[0]} to '
           f'{historical_individual_predictions.index[ens_init_trainsize - 1]}',
           f'\nIn an historical expanding window approach, there are {H_ensemble} periods to be forecasted by the '
           f'ensemble models: '
           f'{historical_individual_predictions.index[ens_init_trainsize]} to'
           f' {historical_individual_predictions.index[-1]}')


    # If out-of-sample prediction is desired, verbose print information about periods to be forecasted
    if fh is None:
        pass
    else:
        last_period = future_individual_predictions.to_timestamp().index[-1]
        if fh == 1:
            vprint(f"Out-of-sample predictions are generated for next period: "
                   f"{(last_period + pd.DateOffset(n=-1))}")
        else:
            vprint(
                f"Out-of-sample predictions are generated for the next {fh} periods: "
                f"{(last_period + pd.DateOffset(n=-1))} "
                f"to {(last_period + pd.DateOffset(n=fh))}")

    # Create empty DataFrames for storing ensemble predictions
    historical_ensemble_predictions = pd.DataFrame()
    if fh is None:
        future_ensemble_predictions = None
    else:
        future_ensemble_predictions = pd.DataFrame()

    # Define the percentage interval for printing forecast updates
    # (e.g., print 0.2 means printing at 0%, 25%, 50%, 75%, and 100%)
    # Include first and last prediction in console output
    printout_percentage_interval = 0.25
    printed_k = [ceil(x) for x in
                 H_ensemble * np.arange(0, 1, printout_percentage_interval)]
    printed_k[0] = 1

    # Iterate over ensemble approaches (e.g., weighted, meta)
    for ensemble_approach, approach_dict in ensemblers.items():
        # Skipping approach when no models are given
        if approach_dict in [None, {}]:
            vprint(f"Skipping {ensemble_approach} since no models are provided.")
            continue
        # Iterate over models in the approach (e.g., weighting scheme or metamodel)
        for ensembler_name, MODEL in approach_dict.items():
            # Skipping approach when model is not properly defined
            if MODEL in [None, {}]:
                vprint(f"Skipping {ensembler_name} since it is not properly by user.")
                continue
            # Consider only selected models in the current ensemble approach dictionary
            if select_ensemblers != 'all' and ensembler_name not in select_ensemblers:
                continue

            # Checks for metamodels input:
            if ensemble_approach == 'meta':
                # If no additional arguments are given, set empty dictionary
                if 'options' not in MODEL.keys():
                    MODEL['options'] = {}
                # Check if package is stated
                if 'package' not in MODEL.keys():
                    raise RuntimeError(f'You need to provide the package for {ensembler_name}.')
            # extract model function, package name, and arguments from the models' dictionary
            model_function, package_name, options = MODEL['model'], MODEL['package'], MODEL['options'],

            # ----------------------------------------------
            # Starting expanding window approach here
            # We are at period t+k and forecast period t+k+1
            # Loop until all H periods are forecasted
            # thus: k = [0, ... , H-1]

            # If verbose, print current state of the process
            printed_model_name = f'{ensembler_name}' + (f' ({package_name})' if package_name else '')
            vprint(
                f'\nNow generating {H_ensemble} one-step ahead historical expanding window predictions from ensemble '
                f'model: \'{ensemble_approach.capitalize()} - {printed_model_name}\'')

            # Set up empty DataFrame for the predictions per model
            historical_predictions_model = pd.DataFrame()

            # Start creating ensemble predictions
            for k in range(H_ensemble):
                # Print forecast updates (console output)
                if k + 1 in printed_k:
                    vprint(f'...Forecast {k + 1} / {H_ensemble}')
                # Current train size = Period at which forecast is made:
                current_trainsize = ens_init_trainsize + k
                current_ensemble_train = historical_individual_predictions.iloc[0:current_trainsize, :]

                next_individual_predictions = historical_individual_predictions.iloc[
                                              current_trainsize:current_trainsize + 1, :]

                # Depending on ensemble approach 'weighted' or 'meta' the ensemble_prediction_wrapper
                # selects the appropriate approach and creates a forecast
                historical_next_individual_prediction = ensemble_prediction_wrapper(
                    past_individual_predictions=current_ensemble_train,
                    next_individual_predictions=next_individual_predictions,
                    approach=ensemble_approach, model_function=model_function,
                    options=options, verbose=verbose,
                )

                # Append to DataFrame
                historical_predictions_model = pd.concat([historical_predictions_model,
                                                          historical_next_individual_prediction])
            vprint("...finished!")

            #  Build ensemble model name for DataFrame
            ensembler_name = f'{ensemble_approach.capitalize()} Ensemble: {ensembler_name}'

            # Store future predictions in a new column of historical ensemble prediction DataFrame
            historical_ensemble_predictions[ensembler_name] = historical_predictions_model

            # Performing out-of-sample predictions for current ensemble model
            if fh is not None:
                vprint(f"Performing out-of-sample predictions...")
                future_ensemble_predictions_model = ensemble_prediction_wrapper(
                    past_individual_predictions=historical_individual_predictions,
                    next_individual_predictions=future_individual_predictions,
                    approach=ensemble_approach, model_function=model_function,
                    options=options, verbose=False
                )

                # Store future predictions in a new column of future ensemble prediction DataFrame
                future_ensemble_predictions[ensembler_name] = future_ensemble_predictions_model

            # Finished predictions of current model
            vprint('...finished!')

    # Finished predictions all ensemble models - provide insights if verbose == True
    vprint("\nFinished predictions of ensemble forecasters!")
    vprint('\nInsights into ensemblers\' historical predictions:',
           historical_ensemble_predictions.head())
    if fh is not None:
        vprint(
            "\nInsights into ensemblers' future predictions:",
            future_ensemble_predictions.head()
        )

    # Merge with historical ensemble predictions with individual predictions
    vprint("\nMerging...")
    full_historical_predictions = historical_individual_predictions.merge(
        historical_ensemble_predictions,
        left_index=True,
        right_index=True,
        how='inner'
    )

    if future_ensemble_predictions is not None:
        full_future_predictions = future_individual_predictions.merge(
            future_ensemble_predictions,
            left_index=True,
            right_index=True,
            how="inner"
        )
    else:
        full_future_predictions = None

    vprint("...finished!\n")

    # If path is specified, export_path results as .csv
    csv_exporter(export_path, full_historical_predictions, full_future_predictions,
                 file_names=["historical_predictions", "future_predictions"])

    return full_historical_predictions, full_future_predictions
