import logging
import sys
from datetime import datetime

from pipeline.processes.pipe1_data_preprocessing import pipe1_data_preprocessing
from pipeline.processes.pipe2_individual_forecasts import pipe2_individual_forecasts
from pipeline.processes.pipe3_ensemble_forecasts import pipe3_ensemble_forecasts
from pipeline.processes.pipe4_metrics_ranking import pipe4_metrics_ranking
from utils.helpers import strfdelta, vprint
from utils.paths import *


# noinspection PyTypeChecker
def run_pipeline(df, models, metrics,
                 date_col='infer', date_format=None, target='infer', covariates='infer', exclude=None,
                 agg_method=None, agg_freq=None,
                 select_forecasters='all', autosarimax_refit_interval=0.33, forecast_init_train=0.3,
                 select_ensemblers='all', ensemble_init_train=0.3,
                 sort_by='MAPE',
                 export=True, errors='raise', verbose=False,
                 *args, **kwargs):
    """
    Run pipeline of data preprocessing, individual, and ensemble forecasting, and subsequent model ranking.

    Args:
    - df: pandas DataFrame or Series, input data containing targets and covariates. 
    - verbose: bool, whether to print intermediate steps (default: False).
    - target:
    - covariates:
    - forecasters:
    - init_splits:
    - verbose:
    
    Returns:
    - target:
    - covariates:
    - individual_predictions:
    - full_predictions:
    - metrics_ranking:
    """

    # Outlook: You could make this pipe more generic by looping over functions.

    # Save starting time
    start_pipe = datetime.now()
    start_pipe_formatted = start_pipe.strftime("%Y%m%d_%H%M")

    if export is True:
        if isinstance(export, os.PathLike):
            export_path = export
        else:
            # If df is imported by internal function csv_reader, export path can be inferred from file_name in attrs.
            if 'file_name' in df.attrs:
                file_name = df.attrs['file_name'].replace('.csv', '')
                export_path = os.path.join(EXPORT_DIR, file_name, start_pipe_formatted)
            else:
                raise ValueError("Provide valid path for 'export' or set 'export = False'")

        # Create non-existing export directory
        if not os.path.exists(export_path):
            os.makedirs(export_path)
    else:
        export_path = None

    # Create log file
    # Outlook: Also store forecasters' hyperparameters in log
    if export and verbose:
        # log_file_name = os.path.join(export_path, f""pipe_log_{start_pipe_formatted}.txt")
        log_file_name = os.path.join(export_path, 'pipe_log.log')

        # Set up logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Output file
        fileHandler = logging.FileHandler(log_file_name)
        fileHandler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(fileHandler)

        # Console output
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logging.Formatter('\x1b[33;1m%(message)s\x1b[0m'))
        logger.addHandler(consoleHandler)

    processes = [
        # Pipeline step 1: Perform data preprocessing
        pipe1_data_preprocessing,
        # Pipeline step 2: Compute individual predictions
        pipe2_individual_forecasts,
        # Pipeline step 3: Compute ensemble predictions
        pipe3_ensemble_forecasts,
        # Pipeline step 4: Ranking by metrics
        pipe4_metrics_ranking
    ]

    # Define expected outputs
    expected_outputs = [
        'target and covariates',
        'individual_predictions',
        'full predictions',
        'metrics ranking'
    ]

    # Set up empty output dictionary
    output_dict = {}

    vprint('================================================================================='
           f'\n[{start_pipe.strftime("%Y-%m-%d %H:%M")}] Starting  Pipeline...')

    # Feed pipeline with pandas DataFrame or Series
    pipe_input = df

    for step, process in enumerate(processes):
        pipe_output = process(
            pipe_input,
            # Pipe 1 arguments
            date_col=date_col, date_format=date_format,
            target=target, covariates=covariates, exclude=exclude,
            agg_method=agg_method, agg_freq=agg_freq,
            # Pipe 2 arguments
            forecasters=models['FORECASTERS'], select_forecasters=select_forecasters,
            forecast_init_train=forecast_init_train,
            autosarimax_refit_interval=autosarimax_refit_interval,
            # Pipe 3 arguments
            ensemblers=models['ENSEMBLERS'], select_ensemblers=select_ensemblers,
            ensemble_init_train=ensemble_init_train,
            # Pipe 4 arguments
            metrics=metrics,
            sort_by=sort_by,
            # Generic/shared arguments
            export_path=export_path,
            errors=errors, verbose=verbose,
            *args, **kwargs
        )
        # Store results to output dictionary
        output_dict[expected_outputs[step]] = pipe_output
        # Previous output is next input
        pipe_input = pipe_output
        # Print time elapsed since start
        if step+1 != len(processes):
            vprint(f'[Time elapsed: {strfdelta(datetime.now() - start_pipe)}]\n')

    # Reporting total time elapsed
    end_pipe = datetime.now()
    vprint(f'\n\n[{end_pipe.strftime("%Y-%m-%d %H:%M")}] Finished Pipeline!\n'
           f'[Total time elapsed: {strfdelta(end_pipe - start_pipe)}]'
           '\n================================================================================='
           )

    # Return results
    return output_dict
