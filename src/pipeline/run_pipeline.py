import logging
import sys
import os
from datetime import datetime

import pandas as pd

from src.utils.paths import *

from src.pipeline.processes.pipe1_data_preprocessing import pipe1_data_preprocessing
from src.pipeline.processes.pipe2_individual_forecasts import pipe2_individual_forecasts
from src.pipeline.processes.pipe3_ensemble_forecasts import pipe3_ensemble_forecasts
from src.pipeline.processes.pipe4_metrics_ranking import pipe4_metrics_ranking
from src.utils.helpers import strfdelta, vprint


# noinspection PyTypeChecker
def run_pipeline(df, models, metrics,
                 fh=None,
                 start=None, end=None,
                 date_col='infer', date_format=None,
                 target='infer', covariates='infer', exclude=None,
                 agg_method=None, agg_freq=None,
                 select_forecasters='all', forecast_init_train=0.3,
                 autosarimax_refit_interval=0.33,
                 select_ensemblers='all', ensemble_init_train=0.25,
                 sort_by='MAPE',
                 export_path=None, errors='raise', verbose=False,
                 paper=False,
                 *args, **kwargs):
    """
    Run pipeline of data preprocessing, individual, and ensemble forecasting, and subsequent model ranking.

    Parameters:
    -----------
        df : pandas.DataFrame, pandas.Series or dict:
            Input DataFrame containing date, targets (and optionally covariates). Can also be a dictionary of
            dataframes with DataFrame names in the keys and DataFrames/Series in the corresponding value.
        models : dict
            Dictionary containing the forecasters and ensemblers models (approach, names, class names, package name,
            and options). This can be imported from the 'models' module of the project.
            Edit the '.yml' file to add/remove models.ch
        metrics : dict
            List of performance measures for model ranking in historical predictions.
            Can be imported from the 'metrics' module of the project. Edit '.yml' files to add/remove metrics.
        fh : int, optional
            When provided, pipeline not only performs historical evaluation of forecasters and ensemblers but also
            provides out-of-sample future predictions along the whole provided forecast horizon.
        start : str, optional
            Filter data to start from date string. Expects ISO DateTimeIndex format "YYYY-MMMM-DDD" (default: None).
        end : str, optional
            Filter data to end on date string. Expects ISO DateTimeIndex format "YYYY-MMMM-DDD" (default: None).
        date_col : str or int, optional
            Name or index of the date column in the input data (default: 'infer', searches for ISO formatted column).
        date_format : str, optional
            Custom format of the date column if date_col is specified (default: None, expects ISO format YYYY-MM-DD).
        target : str, int, optional
            Name or positional index of the target column in the input data
            (default: 'infer', takes first column after the date was set).
        covariates : str, int, or list, optional
            Names of covariates columns in the input data (default: 'infer', takes all columns after date and target
            are inferred.).
        exclude : str, int, or list, optional
            List of columns (string or positional index) to exclude from the input data (default: None).
        agg_method : str, optional
            Aggregation method for preprocessing.
            One of the pandas methods 'first', 'last', 'min', 'max', and 'mean' (default: None).
        agg_freq : str, optional
            DateTimeIndex aggregation frequency for preprocessing (default: None).
        select_forecasters : str or list, optional
            Specify which forecaster classes to use (default: 'all').
        forecast_init_train : float, optional
            Initial forecasters' training set size as a fraction of preprocessed data (default: 0.3, results in a
            30%/80% train-test split of the data).
        autosarimax_refit_interval : float, optional
            Refit interval for AutoSARIMA model (default: 0.33, corresponds to fitting at 0%, 33%, and 66% of ensemblers
            training set).
        select_ensemblers : str or list, optional
            Specify which ensemblers to use (default: 'all').
        ensemble_init_train : float, optional
            Initial ensemblers' training set size as a fraction of individual forecasters' predictions (default: 0.25,
            results in a 25%/75% train-test split of the data).
        sort_by : str, optional
            Performance measure to sort by for model ranking (default: 'MAPE').
        export_path : str or os.PathLike, optional
            Exports results to provided path
        errors : str, optional
            How to handle errors (default: 'raise').
        verbose : bool, optional
            If True, prints progress, intermediate results and steps console and log file (default: False).
        *args:
            Additional positional arguments.
        **kwargs:
            Additional keyword arguments.

    Returns:
    --------
    dict: Dictionary containing the following keys as pandas Series or DataFrames:
        - 'target and covariates': Tuple of preprocessed target and covariates.
        - 'historical_individual_predictions': Individual forecasters' predictions.
        - 'full predictions': Full ensemble predictions.
        - 'metrics ranking': Rankings based on specified metrics.
    If input is a dictionary, then the results will be a dictionary of dictionaries for each DataFrame in the input
    dictionary.
    """

    df = df.copy()

    # Save starting time
    start_pipe = datetime.now()
    start_pipe_formatted = start_pipe.strftime("%Y%m%d_%H%M")

    # For paper pipline execution
    if isinstance(df, (pd.Series, pd.DataFrame)):
        # If df is imported by internal function csv_reader, export_path path can be inferred from file_name in attrs.
        if 'file_name' in df.attrs:
            file_name = df.attrs['file_name']
        else:
            file_name = "output"

        # Outer loop requires DataFrames as dictionary to iterate over
        df_dict = {file_name: df}
    elif isinstance(df, dict):
        # Is already a dict
        df_dict = df

    else:
        raise ValueError("df input must be either a pandas DataFrame or Series or a list of these.")

    # Set up output dict
    output = {}

    # Correct confusing user inputs
    if fh == 0:
        fh = None

    # Set up logging if both exporting and verbosity are enabled
    # Outlook: Also store forecasters' hyperparameters in log
    # Set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Set up console output/print logger
    if verbose:
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logging.Formatter('\x1b[33;1m%(message)s\x1b[0m'))
        logger.addHandler(consoleHandler)

    # Define pipeline processes
    processes = [
        pipe1_data_preprocessing,   # Pipeline step 1: Perform data preprocessing
        pipe2_individual_forecasts, # Pipeline step 2: Compute individual predictions
        pipe3_ensemble_forecasts,   # Pipeline step 3: Compute ensemble predictions
        pipe4_metrics_ranking       # Pipeline step 4: Ranking by metrics
    ]

    # Define expected outputs
    expected_outputs = [
        'target and covariates',
        'historical_individual_predictions',
        'full predictions',
        'metrics ranking'
    ]

    for df_name, df in df_dict.items():

        print_info = f" for {df_name if df_name != 'output' else ''}"

        # Print pipeline start time
        vprint('================================================================================='
               f'\n[{start_pipe.strftime("%Y-%m-%d %H:%M")}] Starting  Pipeline{print_info}...')
        
        # Set up empty output dictionary
        output_dict = {}

        # Determine export_path path if exporting is enabled
        if export_path:
            if isinstance(export_path, (os.PathLike, str)):
                curr_export_path = os.path.join(export_path, start_pipe_formatted, df_name)
            else:
                try:
                    curr_export_path = os.path.join(PIPE_OUTPUT_DIR, start_pipe_formatted, df_name)
                except:
                    raise ValueError("Provide valid path for 'export_path'.")
            # Create non-existing export_path directory
            if not os.path.exists(curr_export_path):
                os.makedirs(curr_export_path)

            # log_file_name = os.path.join(export_path, f""pipe_log_{start_pipe_formatted}.txt")
            log_file_name = os.path.join(curr_export_path, 'pipe.log')
            # Output file
            fileHandler = logging.FileHandler(log_file_name)
            fileHandler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(fileHandler)
        else:
            curr_export_path = None

        # Feed pipeline with pandas DataFrame or Series
        pipe_input = df.copy()

        # Iterate over pipeline processes
        for step, process in enumerate(processes):
            # Execute each process
            pipe_output = process(
                # Provide required input (DataFrame or result from previous pipeline step)
                pipe_input,
                # Provide arguments for Pipe 1
                start=start, end=end,
                date_col=date_col, date_format=date_format,
                target=target, covariates=covariates, exclude=exclude,
                agg_method=agg_method, agg_freq=agg_freq,
                # Provide arguments for Pipe 2
                forecasters=models['FORECASTERS'], select_forecasters=select_forecasters,
                forecast_init_train=forecast_init_train,
                autosarimax_refit_interval=autosarimax_refit_interval,
                # Provide arguments for Pipe 3
                ensemblers=models['ENSEMBLERS'], select_ensemblers=select_ensemblers,
                ensemble_init_train=ensemble_init_train,
                # Provide arguments for Pipe 4
                metrics=metrics,
                sort_by=sort_by,
                # Generic/shared arguments
                fh=fh,
                export_path=curr_export_path,
                errors=errors, verbose=verbose,
                *args, **kwargs
            )

            # Store results to output dictionary
            output_dict[expected_outputs[step]] = pipe_output

            # Previous output is next input
            pipe_input = pipe_output

            # Print time elapsed since start
            if step+1 != len(processes):
                vprint(f'\n[Time elapsed: {strfdelta(datetime.now() - start_pipe)}]\n')

        # Report logfile saving message
        if export_path:
            print(f"Saving logfile to {os.path.join(export_path, 'log_file.log')}")

        # Reporting total time elapsed
        end_pipe = datetime.now()
        vprint(f'\n[{end_pipe.strftime("%Y-%m-%d %H:%M")}] Finished Pipeline{print_info}!\n'
               f'[Total time elapsed: {strfdelta(end_pipe - start_pipe)}]'
               '\n=================================================================================\n'
               )

        if len(df_dict) > 1:
            output[df_name] = output_dict
        else:
            output = output_dict

    # Return results
    return output
