import warnings
from math import ceil

import numpy as np
import pandas as pd
from sktime.split import ExpandingWindowSplitter

from utils.helpers import vprint, csv_exporter
from utils.mappings import SEASONAL_FREQ_MAPPING
from utils.transformers import TRANSFORMERS


def pipe2_individual_forecasts(target_covariates_tuple,
                               forecasters, select_forecasters='all',
                               forecast_init_train=0.3,
                               autosarimax_refit_interval=0.33,
                               export_path=None, verbose=False,
                               *args, **kwargs):

    vprint("\n======================================================="
           "\n== Starting Step 2 in Pipeline: Individual Forecasts =="
           "\n=======================================================\n")

    # Extract tuple elements
    target = target_covariates_tuple[0]
    covariates = target_covariates_tuple[1]

    # Calculate initial train size
    init_trainsize = int(target.shape[0] * forecast_init_train)

    # Define target training variables
    y_train_full = target

    # Infer seasonal frequency
    inferred_seasonal_freq = SEASONAL_FREQ_MAPPING[target.index.freqstr] \
        if target.index.freqstr in SEASONAL_FREQ_MAPPING.keys() else None

    if inferred_seasonal_freq > init_trainsize:
        warnings.warn('Too few observations provided for seasonal forecasters. '
                      'If you provide such forecasters, consider removing them!')

    # Turn off warnings
    warnings.filterwarnings('ignore')

    # Calculate full forecast horizon
    H = y_train_full.shape[0] - init_trainsize

    vprint(
         f"Splitting data (train/test ratio: "
         f"{int(forecast_init_train * 100)}/{int(100 - forecast_init_train * 100)})..."
         f"\nInitial training set has {init_trainsize} observations ",
         f"and goes from {target.index[0].date()} to {target.index[init_trainsize - 1].date()}"
         f"\nThere are {H} periods to be forecasted: ",
         f"{target.index[init_trainsize].date()} to {target.index[-1].date()}\n"
    )

    # Create a DataFrame to store all models' predictions
    individual_predictions = pd.DataFrame()
    individual_predictions.index.name = 'Date'

    # Initialize last model_source and covtreatment_bool objects
    last_model_source = None
    last_covtreatment_bool = None

    # Initialize transformed datasets
    y_train_transformed = None
    X_train_transformed = None

    # Set percentage interval for printing forecast updates
    # (e.g., print 0.2 means printing at 0%, 20%, 40%, 60%, 80%, and 100%)
    # Include first and last prediction in console output
    printout_percentage_interval = 0.2
    printed_k = [ceil(x) for x in H * np.arange(0, 1 + printout_percentage_interval, printout_percentage_interval)]
    printed_k[0] = 1

# Loop over model approach (with or without covariates)
    for approach, approach_dict in forecasters.items():

        # Define boolean object that indicates if it is a covariate model or not
        covtreatment_bool = True if approach == 'with_covariates' else False

        # Skip covariates forecasters when no covariates are specified
        if covtreatment_bool and covariates is None:
            vprint(f"\nSince no covariates are given, skipping covariate forecasters.")
            continue
        # Skip approach when no models are given.
        if len(approach_dict) == 0:
            vprint(f"No models given for approach {approach}.")
            continue

        # Initialize variable last_package_name and last_covtreatment_bool such that data is not transformed twice
        last_package_name = None
        last_covtreatment_bool = None

        # Loop over individual model in the current sub-dictionary (with/without covariates)
        for model_name, MODEL in approach_dict.items():
            # Only consider selected models in the current ensemble approach dictionary
            if select_forecasters not in ['all', ['all']] and model_name not in select_forecasters:
                continue

            # Define X_train_full depending on covtreatment_bool, i.e., depending on whether covariates are specified
            X_train_full = covariates if covtreatment_bool else None

            # Extract model information from dictionary
            model_function, package_name, options = MODEL['model'], MODEL['package'], MODEL['options']

            # Adjust model name depending on the approach (with or without covariates)
            model_name += (' with covariates' if covtreatment_bool else '')

            # Construct model with corresponding arguments
            model = model_function(**options)

            # For the prediction process some packages require transformed data
            if package_name == 'sktime' and covtreatment_bool:
                package_name = package_name + '.lagged'

            # Depending on package_name select corresponding data transformer

            # Do not transform again if model source did not change
            if last_package_name == package_name and last_covtreatment_bool == covtreatment_bool:
                pass
            # Transform with provided TRANSFORMERS
            elif package_name in TRANSFORMERS.keys():
                # Select transformer
                transformer = TRANSFORMERS[package_name]
                # Transform
                y_train_transformed, X_train_transformed = transformer(y_train_full, X_train_full)
            # No transformation possible if package_name not in TRANSFORMERS
            else:
                raise RuntimeError(f'{package_name} is not yet supported')

            # Set up empty DataFrame for predictions and define index named 'Date'
            model_predictions = pd.DataFrame()
            model_predictions.index.name = 'Date'

            # Starting one-step ahead expanding window predictions for current model (fitting, updating, predicting)
            vprint(f'Now generating {H} one-step ahead expanding window predictions from model: '
                   f'{model_name} ({package_name.replace('.lagged', '')})'
                   )
            # Method and data transformer is inferred from package_name

            # darts forecasters use .historical_forecasts() method:
            if 'darts' in package_name:
                model_predictions = model.historical_forecasts(
                    series=y_train_transformed, start=init_trainsize, stride=1,
                    forecast_horizon=1, past_covariates=X_train_transformed,
                    show_warnings=False).pd_dataframe()
                # Transform back to periodIndex (nicer outputs)
                period_freq = 'M' if target.index.freqstr == 'MS' else target.index.freqstr
                model_predictions.set_index(
                    pd.PeriodIndex(
                        pd.to_datetime(model_predictions.index), freq=period_freq),
                    inplace=True)

            # sktime forecasters
            elif 'sktime' in package_name:
                # Adjust sktime specific parameters
                # in particular: Seasonal periodicity
                if 'Naive' not in model_name:  # sNaive performs bad. Does not make sense to use this.
                    model.set_params(**{'sp': inferred_seasonal_freq})

                # all sktime forecasters but ARIMA
                if 'ARIMA' not in model_name:
                    # they use ExpandingWindowSplitter and .update_predict() method for historical forecasts
                    cv = ExpandingWindowSplitter(fh=1, initial_window=init_trainsize, step_length=1)
                    model.fit(y_train_transformed[:init_trainsize])
                    model_predictions = model.update_predict(y_train_transformed, cv=cv)

                # ARIMA
                else:
                    # Extra treatment for ARIMA model
                    # (Updating and Refitting each period with method above would take too much time here)
                    # Outlook:
                    # - Consider implementing UpdateRefitEvery() wrapper from sktime package (threw an error)
                    # - source this piece of code out
                    # - and make own .update_predict method for ARIMA (wrap in class)

                    # Define at what interval ARIMA model is being refitted
                    # autosarimax_refit_interval is between 0 and 1
                    # Default: 0, thus: refitting each period
                    # 1 would mean: no refitting at all
                    # Example 0.33, means: only refitting at 33% and 66% of predictions made
                    # + Fitting at the beginning
                    # Fitting works with the AutoArima approach from Hyndman, RJ and Khandakar, Y (2008)
                    if autosarimax_refit_interval in [0, None]:
                        refit_freq = 1
                    else:
                        refit_freq = ceil(H / (1 / autosarimax_refit_interval))

                    # Print information about refitting
                    if refit_freq == 2:
                        vprint(f'Auto-fitting model. Refitting every {refit_freq}nd period.')
                    elif refit_freq == 1:
                        vprint('Auto-fitting model. Refitting every period.')
                    else:
                        vprint(f'Auto-fitting model. Refitting every {refit_freq}th period.')

                    # sktime.lagged transformer removes the first period due to NaNs => positional indices change
                    lag_indicator = 1 if 'lagged' in package_name else 0

                    # We are at period t+k and forecast period t+k+1
                    # Loop until until all H periods are forecasted
                    # thus: k = [0, ... , H-1]
                    for k in range(H):
                        current_trainsize = init_trainsize - lag_indicator + k
                        current_y_train_arima = y_train_transformed[:current_trainsize]

                        current_X_train_arima = X_train_transformed[:current_trainsize] \
                            if X_train_transformed is not None else None

                        # Refit and Update AutoSARIMA(X) Model

                        # Refit AutoArima model:
                        # at period 0 and every 'refit_freq'th period
                        if k % refit_freq == 0:
                            # When refitting, update model with previous parameters (potentially speeds up fitting)
                            if k != 0:
                                sarima_fitted_params = model.get_fitted_params(deep=True)
                                p, d, q = sarima_fitted_params['order']
                                P, D, Q, sp = sarima_fitted_params['seasonal_order']

                                updated_params = {
                                    'start_p': p,
                                    'd': d,
                                    'start_q': q,
                                    'start_P': P,
                                    'D': D,
                                    'start_Q': Q,
                                    'sp': sp,
                                    'maxiter': 15
                                }
                                model.set_params(**updated_params)
                                # vprint('...automatic refitting...')

                            # Fit model first time or refit model
                            model.fit(y=current_y_train_arima, X=current_X_train_arima)

                        # Update:
                        else:
                            # In all other periods just update parameters/coefficients
                            model.update(y=current_y_train_arima, X=current_X_train_arima)

                        # Print forecast update
                        if k + 1 in printed_k:
                            vprint(f'{model_name} forecast {k + 1} / {H}')

                        # Predict:
                        # Select last known X as predictor if using a covariate model
                        X_pred_sarimax = (X_train_transformed[current_trainsize:
                                                              current_trainsize + 1]
                                          if covtreatment_bool else None)
                        # Perform prediction
                        prediction = model.predict(fh=1, X=X_pred_sarimax)
                        model_predictions = pd.concat([model_predictions, prediction], axis=0)

            # After finishing historical forecasts per model: store predictions in a new column
            vprint('...finished!\n')
            individual_predictions[model_name] = model_predictions

            # Save model information to avoid double transforming when no change in model source
            last_covtreatment_bool = covtreatment_bool
            last_package_name = package_name

    vprint('\nIndividual forecasters\' predictions finished!\n'
           '\nInsights into forecasters\' predictions:',
           individual_predictions.head(), '\n'
           )

    target_output = y_train_full[init_trainsize:]
    period_freq = 'M' if y_train_full.index.freqstr == 'MS' else y_train_full.index.freqstr
    target_output.index = pd.PeriodIndex(target_output.index, freq=period_freq)
    individual_predictions.insert(0, 'Target', value=target_output)

    # If path is specified, export results as .csv
    csv_exporter(export_path, individual_predictions)

    return individual_predictions


# For debugging:
# from forecasters.forecasting import FC_MODELS
# from pipe1_data_preprocessing import pipe1_data_preprocessing
# from paths import *
# import os
#
# FILE_PATH = os.path.join(SIMDATA_DIR, 'noisy_simdata.csv')
# df = pd.read_csv(FILE_PATH)
# target, covariates = pipe1_data_preprocessing(df, verbose=True)
# indiv_fc = pipe2_individual_forecasts(target=target, covariates=covariates, forecasters=FC_MODELS, verbose=True)
