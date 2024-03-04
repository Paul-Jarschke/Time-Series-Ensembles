import pandas as pd
import warnings
import numpy as np
from math import ceil
from utils.helpers import vprint, csv_exporter
from utils.transformers import TRANSFORMERS
from utils.mappings import SEASONAL_FREQ_MAPPING
from sktime.split import ExpandingWindowSplitter


def pipe2_individual_forecasts(target_covariates_tuple,
                               models, select_models='all',
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
        warnings.warn('Too few observations provided for seasonal models. '
                      'If you provide such models, consider removing them!')

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

    # Initialize last model_source and covmodel_bool objects
    last_model_source = None
    last_covmodel_bool = None

    # Initialize transformed datasets
    y_train_transformed = None
    X_train_transformed = None

    # Set percentage interval for printing forecast updates
    # (e.g., print 0.2 means printing at 0%, 20%, 40%, 60%, 80%, and 100%)
    # Include first and last prediction in console output
    printout_percentage_interval = 0.2
    printed_k = [ceil(x) for x in np.arange(0, 1 + printout_percentage_interval, printout_percentage_interval)] * H

    # Loop over type of model (with or without covariates)
    for covariates_indicator, models_dict in models.items():

        # Remove models specified by user from current models dictionary
        models_to_remove = [model for model in models_dict.keys() if model not in select_models]
        for model in models_to_remove:
            models_dict.pop(model)

        # Set object that indicates if it is a covariate model or not
        covmodel_bool = True if covariates_indicator == 'covariates' else False
        # Define X_train_full depending on covmodel_bool
        X_train_full = covariates if covmodel_bool else None

        # Loop over individual model in each sub-dictionary
        # Skip covariates models when no covariates are specified
        if covmodel_bool and covariates is None:
            vprint(f"Since no covariates are given, skipping covariate models {', '.join(models_dict.keys())}")
            continue

        for model_name, model in models_dict.items():

            # Find out model source
            # everything before first point and remove the '<class ' part of the string
            model_source = str(type(model)).split('.')[0][8:]
            # Add covariate information to model name
            model_name = model_name + (' with covariates' if covmodel_bool else '')
            vprint(f'Now generating {H} one-step ahead expanding window predictions from model: '
                   f'{model_name} ({model_source})'
                   )

            if model_source == 'sktime' and covmodel_bool:
                model_source = model_source + '.lagged'

            # Depending on model_source select corresponding data transformer
            # Do not transform again if model source did not change
            if last_model_source == model_source and last_covmodel_bool == covmodel_bool:
                pass
            # Transform with provided transformer
            elif model_source in TRANSFORMERS.keys():
                # Select transformer
                transformer = TRANSFORMERS[model_source]
                # Transform
                y_train_transformed, X_train_transformed = transformer(y_train_full, X_train_full)
            # No transformation if model_source not in TRANSFORMERS
            else:
                y_train_transformed = y_train_full.copy()
                X_train_transformed = X_train_full.copy()

            # Fit the model and make historical expanding window one-step ahead predictions
            # Method and data transformer is inferred from model_type

            # Set up empty DataFrame
            model_predictions = pd.DataFrame()
            model_predictions.index.name = 'Date'

            # darts models:
            if 'darts' in model_source:
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

            # sktime models
            elif 'sktime' in model_source:
                # Adjust sktime specific parameters
                # Seasonal periodicity
                if model_name != 'Naive':  # sNaive performs bad. Does not make sense to use this.
                    model.set_params(**{'sp': inferred_seasonal_freq})

                # all sktime models but ARIMA
                if 'ARIMA' not in model_name:
                    cv = ExpandingWindowSplitter(fh=1, initial_window=init_trainsize, step_length=1)
                    model.fit(y_train_transformed[:init_trainsize])
                    model_predictions = model.update_predict(y_train_transformed, cv)

                # ARIMA
                else:
                    # Extra treatment for ARIMA model (Updating and Refitting each period would take too much time here)
                    # Outlook:
                    # - Consider implementing UpdateRefitEvery() wrapper from sktime package (threw an error)
                    # - source this piece of code out
                    # - and make own .update_predict method for ARIMA (wrap in class)

                    # Define at what frequency ARIMA model is being refitted
                    autosarimax_refit_interval = 1 if autosarimax_refit_interval is None else autosarimax_refit_interval
                    refit_freq = (H // (
                            1 / autosarimax_refit_interval) + 1)  # 33 % intervals => consider lowering to 20% or 10%

                    vprint('Auto-fitting model...')

                    # sktime.lagged transformer removes the first period due to NaNs => positional indices change
                    lag_indicator = 1 if 'lagged' in model_source else 0

                    # We are at period t+k and forecast period t+k+1
                    # Loop until until all H periods are forecasted
                    # thus: k = [0, ... , H-1]
                    for k in range(H):
                        current_trainsize = init_trainsize - lag_indicator + k
                        current_y_train_arima = y_train_transformed[:current_trainsize]

                        current_X_train_arima = X_train_transformed[:current_trainsize] \
                            if X_train_transformed is not None else None

                        # Refit and Update AutoSARIMA(X) Model

                        # Refit:
                        if k % refit_freq == 0:
                            if k != 0:  # refit model at period 0 and every 'refit_freq'th period
                                # Initialize model with previous parameters (speed up fitting)
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
                                vprint('...automatic refitting...')
                            model.fit(y=current_y_train_arima, X=current_X_train_arima)

                        # Update:
                        else:
                            # In all other periods just update parameters/coefficients
                            model.update(y=current_y_train_arima, X=current_X_train_arima)

                        # Print forecast update
                        if k in printed_k:
                            vprint(f'{model_name} forecast {k + 1} / {H}')

                        # Predict:
                        # Select last known X as predictor if using a covariate model
                        X_pred_sarimax = (X_train_transformed[current_trainsize:
                                                              current_trainsize + 1]
                                          if covmodel_bool else None)
                        # Perform prediction
                        prediction = model.predict(fh=1, X=X_pred_sarimax)
                        model_predictions = pd.concat([model_predictions, prediction], axis=0)

            # Store predictions in a new column
            vprint('...finished!\n')
            individual_predictions[model_name] = model_predictions

            # Save model information to avoid double transforming when no change in model source
            last_model_source = model_source
            last_covmodel_bool = covmodel_bool

    vprint('\nIndividual predictions finished!\n'
           '\nInsights into models\' predictions:',
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
# from models.forecasting import FC_MODELS
# from pipe1_data_preprocessing import pipe1_data_preprocessing
# from paths import *
# import os
#
# FILE_PATH = os.path.join(SIMDATA_DIR, 'noisy_simdata.csv')
# df = pd.read_csv(FILE_PATH)
# target, covariates = pipe1_data_preprocessing(df, verbose=True)
# indiv_fc = pipe2_individual_forecasts(target=target, covariates=covariates, models=FC_MODELS, verbose=True)
