import pandas as pd

from utils.helpers import vprint, csv_exporter


# for debugging:
# from paths import *
# full_predictions = pd.read_csv(os.path.join(EXPORT_DIR, 'full_predictions.csv'), index_col=0)


def pipe4_metrics_ranking(
        full_predictions, metrics,
        export_path=None, sort_by='MAPE',
        verbose=False,
        *args, **kwargs):
    """
    Perform ranking of models (base forecasters and ensemble models!) based on specified performance metrics.

    Parameters:
    -----------
    full_predictions : pandas.DataFrame
        DataFrame containing predictions made by different models (forecasters and ensemblers). 
        Index containing dates should be of pandas PeriodIndex format.
    metrics : dict
            List of performance measures for model ranking in historical predictions.
            Can be imported from the 'metrics' module of the project. Edit '.yml' files to add/remove metrics.
    export_path : os.PathLike, optional
        Path to export the ensemble predictions as a CSV file. If not provided, no CSV file is exported 
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

    # Verbose print to indicate which metrics are being calculated
    vprint(f"Calculating {', '.join(metrics.keys())} per model...")

    # Extract actual values from predictions
    y_actual = full_predictions.pop('Target')

    # Extract model names and store them in a dictionary
    # The dictionary will be used to create the first column in the metric DataFrame
    model_names = full_predictions.columns
    metrics_dict = {'Model': model_names}

    # Calculate metrics per (individual and ensemble) model
    # Iterate over each model
    for model_name in model_names:

        y_predicted = full_predictions[model_name]  # Predicted values

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
    # Iterate over each metric column
    vprint('Ranking forecasters ...')
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
    metrics_ranking.reset_index(drop=True, inplace=True)

    # Verbose print to indicate completion of ranking
    vprint('...finished!\n')

    # If export path is specified, export results as .csv
    csv_exporter(export_path, metrics_ranking)

    # Verbose print to display the resulting ranking DataFrame
    vprint('Results:',
           metrics_ranking)

    return metrics_ranking

# pipe4_metrics_ranking(full_predictions, metrics=metrics, export_path=EXPORT_DIR)
