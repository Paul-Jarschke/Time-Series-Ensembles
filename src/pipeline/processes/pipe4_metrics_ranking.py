import pandas as pd

from src.utils.helpers import vprint, csv_exporter


def pipe4_metrics_ranking(
        full_predictions, metrics,
        fh=None,
        export_path=None, sort_by='MAPE',
        verbose=False,
        *args, **kwargs):
    """
    Perform ranking of models (base forecasters and ensemble models!) based on specified performance metrics.

    Parameters:
    -----------
    full_predictions : tuple
        Tuple contains pandas DataFrames containing historical and future predictions. These are obtained by
        the different models specified (forecasters and ensemblers). Index containing dates should be a PeriodIndex.
        The future predictions correspond to None when out-of-sample predicting is not desired by the user.
    metrics : dict
        List of performance measures for model ranking in historical predictions.
        Can be imported from the 'metrics' module of the project. Edit '.yml' files to add/remove metrics.
    fh : int, optional
        When provided, pipeline not only performs historical evaluation of forecasters and ensemblers but also
        provides out-of-sample future predictions along the whole provided forecast horizon.
    export_path : os.PathLike, optional
        Path to export_path the ensemble predictions as a CSV file. If not provided, no CSV file is exported
        (default: None).
    sort_by : str, optional
        Performance measure to sort by for model ranking (default: 'MAPE').
    verbose : bool, optional
        If True, prints detailed information about the individual forecasting process and stores results in 
        log file (default: False).
    *args:
            Additional positional arguments.
    **kwargs:
        Additional keyword arguments.

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the ranking of implemented models based on specified performance metrics.
    """

    # Verbose print to indicate the start of ranking calculation
    vprint('\n=============================================================='
           '\n== Pipeline Step 4: Ranking Models\' Predictive Performance =='
           '\n==============================================================\n')

    # Unpack tuple with historical and future predictions
    full_historical_predictions = full_predictions[0]
    full_future_predictions = full_predictions[1]

    # Verbose print to indicate which metrics are being calculated
    vprint(f"Calculating {', '.join(metrics.keys())} per model...")

    # Extract actual values from predictions
    y_actual = full_historical_predictions.pop('Target')

    # Extract model names and store them in a dictionary
    # The dictionary will be used to create the first column in the metric DataFrame
    model_names = full_historical_predictions.columns
    metrics_dict = {'Model': model_names}

    # Calculate metrics per (individual and ensemble) model
    # Iterate over each model
    for model_name in model_names:

        y_predicted = full_historical_predictions[model_name]  # Predicted values

        # Loop over metrics and corresponding function in METRICS dictionary
        for metric_name, metric_function in metrics.items():
            if metric_name not in metrics_dict:
                metrics_dict[metric_name] = []
            # Calculate performance metric
            model_metrics = metric_function(y_actual, y_predicted)
            # Save to metrics dictionary
            metrics_dict[metric_name].append(model_metrics)

    # Save metrics dictionary as pandas DataFrame
    metrics_df = pd.DataFrame(metrics_dict)

    # Rank the forecasters based on metric columns
    vprint('Ranking models ...')
    # Iterate over each metric column
    for metric_name, metric_values in metrics_df.items():
        if metric_name == 'Model':  # No ranking for 'Model' column
            continue
        # Transform metric name to uppercase for clarity
        metric_name = metric_name
        # Rank metric column (highest metric = lowest rank and vice versa)
        metrics_df[f'{metric_name} Ranking'] = [int(rank) for rank in metric_values.rank()]

    # Sort the DataFrame based on selected metric
    metrics_ranking = metrics_df.sort_values(by=f'{sort_by} Ranking')

    # Reset index
    # metrics_ranking.reset_index(drop=True, inplace=True)
    metrics_ranking.set_index('Model', inplace=True)

    # Verbose print to indicate completion of ranking
    vprint('...finished!')

    # Verbose print to display the resulting ranking DataFrame
    vprint('\nResults:',
           metrics_ranking)

    # If future predictions are to be made, make a recommendation as to which model to choose based on metric ranking
    best_model = metrics_ranking.index[0]
    vprint(f'\nThe \'{best_model}\' is identified as the best model based on the {sort_by} value of its the historical '
           f'predictions.')
    # Show the corresponding future predictions
    if fh is not None:
        vprint('Thus, it is recommended to work with the future predictions coming from this model:')
        vprint(full_future_predictions[best_model])

    # If export_path path is specified, export_path results as .csv
    csv_exporter(export_path, metrics_ranking)

    return metrics_ranking
