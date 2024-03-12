import warnings
from math import ceil

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from sktime.split import ExpandingWindowSplitter

from src.utils.helpers import vprint
from src.utils.mappings import SEASONAL_FREQ_MAPPING
from src.utils.transformers import TRANSFORMERS
from src.utils import csv_exporter


def pipe2_individual_forecasts(
        target_covariates_tuple,
        forecasters,
        forecast_init_train,
        fh=None,
        select_forecasters="all",
        autosarimax_refit_interval=0.15,
        export_path=None,
        verbose=False,
        *args, **kwargs):
    """
    Generates individual forecasts for each model specified in the forecasters' dictionary.

    Parameters:
    -----------
        target_covariates_tuple : tuple
            Tuple containing (target, covariates) as pandas DataFrames. Index containing dates 
            should be of pandas DateTimeIndex format.
        forecasters (dict)
            Dictionary containing forecasters' class names, package and options.
        forecast_init_train : float
            Initial forecasters' training set size as a fraction of preprocessed data.
        fh : int, optional
            When provided, pipeline not only performs historical evaluation of forecasters and ensemblers but also
            provides out-of-sample future predictions along the whole provided forecast horizon.
        select_forecasters : str or list, optional
            Specify which forecaster classes to use (default: 'all').
        autosarimax_refit_interval : float, optional
            Refit interval for AutoSARIMA model (default: 0.33, corresponds to fitting at 0%, 33%, and 66% of ensemblers
            training set).
        export_path : os.PathLike, optional
            Path to export_path the ensemble predictions as a CSV file. If not provided, no CSV file is exported
            (default: None).
        verbose : bool, optional
            If True, prints detailed information about the individual forecasting process and stores results in log file
            (default: False).
        *args:
            Additional positional arguments.
        **kwargs:
            Additional keyword arguments.

    Returns:
    --------
        historical_predictions : pandas.DataFrame
            DataFrame containing historical individual predictions per forecasting model.
        future_predictions : pandas.DataFrame
            DataFrame containing future individual predictions per forecasting model (if desired, otherwise:
            None).
        
    Notes:
    ------
    This function performs forecasting using the provided individual forecasting models.
    It iterates over each approach and model in the models dictionary and generates historical one-step-ahead
    predictions. Finally, if an export_path path is provided, it exports the ensemble predictions as a CSV file.
    """

    # Verbose print to indicate that individual forecasters start historical forecasts
    vprint(
        "\n====================================================="
        "\n== Pipeline Step 2: Individual Models\' Predictions =="
        "\n=====================================================\n"
    )

    # Extracting target and covariates data from input tuple
    target = target_covariates_tuple[0]
    covariates = target_covariates_tuple[1]

    # Calculating the initial training set size based on specified fraction
    init_trainsize = ceil(target.shape[0] * forecast_init_train)

    # Defining target variable for training as the target data
    y_train_full = target

    # Inferring the seasonal frequency of the target data
    freq = target.index.freqstr
    inferred_seasonal_freq = (
        SEASONAL_FREQ_MAPPING[freq]
        if target.index.freqstr in SEASONAL_FREQ_MAPPING.keys()
        else None
    )

    # Issuing a warning if the available data is insufficient for seasonal forecasters
    if inferred_seasonal_freq > init_trainsize:
        warnings.warn(
            "Too few observations provided for seasonal forecasters. "
            "If you provide such forecasters, consider removing them!"
        )

    # Turning off warnings (temporarily) for cleaner output
    warnings.filterwarnings("ignore")

    # Calculating full forecast horizon
    H = y_train_full.shape[0] - init_trainsize

    # Providing detailed information about the data splitting process
    vprint(
        f"Splitting data for training of forecasters (train/test ratio: "
        f"{int(forecast_init_train * 100)}/{int(100 - forecast_init_train * 100)})...",
        f"Initial training set has {init_trainsize} observations " f"and goes from {target.index[0]} to"
        f" {target.index[init_trainsize - 1]}.",
        f"\nIn a historical expanding window approach, there are {H} periods to be forecasted by the individual "
        f"models: "
        f"{target.index[init_trainsize]} to {target.index[-1]}",
    )

    # If out-of-sample prediction is desired, verbose print information about periods to be forecasted
    if fh is None:
        pass
    else:
        last_period = target.index[-1]
        offset_next = to_offset(freq)
        offset_end = to_offset(str(fh) + freq)

        if fh == 1:
            vprint(f"Out-of-sample predictions are generated for next period: "
                   f"{(last_period + offset_next)}")
        else:
            offset1 = to_offset(target.index.freq)
            vprint(
                f"Out-of-sample-predictions are generated for the next {fh} periods: "
                f"{(last_period + offset_next)} "
                f"to {(last_period + offset_end)}")

    # Creating a DataFrame to store all models' predictions
    historical_predictions = pd.DataFrame()
    historical_predictions.index.name = "Date"
    if fh is None:
        future_predictions = None
    else:
        future_predictions = pd.DataFrame()
        future_predictions.index.name = "Date"

    # Initialize transformed datasets
    y_train_transformed = None
    X_train_transformed = None

    # Setting percentage interval for printing forecast updates
    # (e.g., print 0.25 means printing at 0%, 25%, 50%, 75%, and 100%)
    # Include first and last prediction in console output
    printout_percentage_interval = 0.25
    printed_k = [
        ceil(x)
        for x in H * np.arange(0, 1, printout_percentage_interval)
    ]
    printed_k[0] = 1

    # Looping over different modeling approaches (with or without covariates)
    for individual_approach, approach_dict in forecasters.items():

        # Determining if the current approach involves covariates
        covtreatment_bool = True if individual_approach == "with_covariates" else False

        # Skipping covariates forecasters if no covariates are specified
        if covtreatment_bool and covariates is None:
            vprint(f"\nSkipping covariate forecasters since no covariates are given.")
            continue

        # Skipping approach when no models are given
        if approach_dict in [None, {}]:
            vprint(f"Skipping {individual_approach} since no models are provided.")
            continue

        # Initializing variable last_package_name and last_covtreatment_bool such that data is not transformed twice
        last_package_name = None
        last_covtreatment_bool = None

        # Loop over individual model in the current sub-dictionary (with/without covariates)
        for forecaster_name, MODEL in approach_dict.items():
            # Skipping approach when model is not properly defined
            if MODEL in [None, {}]:
                vprint(f"Skipping {forecaster_name} since it is not properly by user.")
                continue
            # Only considering selected models in the current ensemble approach dictionary
            if (
                    select_forecasters not in ["all", ["all"]]
                    and forecaster_name not in select_forecasters
            ):
                continue

            # Defining X_train_full depending on covtreatment_bool, i.e., depending on whether covariates are specified
            X_train_full = covariates if covtreatment_bool else None

            # Extracting model information from the dictionary
            if "options" not in MODEL.keys():
                MODEL["options"] = {}
            if "package" not in MODEL.keys():
                raise RuntimeError(f"You need to provide the package for {forecaster_name}.")
            model_function, package_name, options = (
                MODEL["model"],
                MODEL["package"],
                MODEL["options"],
            )

            # Adjusting model name depending on the approach (with/without covariates)
            forecaster_name += " with covariates" if covtreatment_bool else ""

            # Constructing model with corresponding arguments
            model = model_function(**options)

            # Handling transformation for specific packages and covariate models
            # For the prediction process some packages require transformed data
            if package_name == "sktime" and covtreatment_bool:
                # For using covariates in sktime, we must lag the data and remove one row.
                # For this we implemented an extra transformer
                package_name = package_name + ".covariates"

            # Determining whether/which transformation is needed based on package and covariate treatment
            # Do not transform again if model source did not change
            if (
                    last_package_name == package_name
                    and last_covtreatment_bool == covtreatment_bool
            ):
                pass

            # Transform with provided TRANSFORMERS
            elif package_name in TRANSFORMERS.keys():
                # Select transformer
                transformer = TRANSFORMERS[package_name]
                # Transform for historical predictions
                y_train_transformed, X_train_transformed = transformer(
                    y_train_full, X_train_full
                )
                # Transform for future predictions (full set)
                if fh is not None:
                    target_transformed, covariates_transformed = transformer(
                        target, covariates
                    )
                # Replace covariates_transformed with None if it is not a covariate model
                if not covtreatment_bool:
                    covariates_transformed = None
            # No transformation possible if package_name not in TRANSFORMERS
            else:
                raise RuntimeError(f"{package_name} is not yet supported")

            # Set up empty DataFrame for predictions and define index named 'Date'
            historical_predictions_model = pd.DataFrame()
            historical_predictions_model.index.name = "Date"

            # Set up empty DataFrame for predictions and define index named 'Date'
            if fh is None:
                future_predictions_model = None
            else:
                future_predictions_model = pd.DataFrame()
                future_predictions_model.index.name = "Date"

            # Generating one-step ahead expanding window predictions for the current model
            # (fitting, updating, predicting)
            vprint(
                f"\nNow generating {H} one-step ahead historical expanding window predictions from model: "
                f'{forecaster_name} ({package_name.split(".")[0]})'
            )

            # Handling different prediction methods based on the package used
            # Method and data transformer are inferred from package_name
            # darts forecasters use .historical_forecasts() method:
            if "darts" in package_name:

                if forecaster_name in ["TiDE", "LSTM", "RNN"]:
                    # For monthly simulated data we took 12 and for business days 126 (half business year)
                    if freq in ["M", "MS"]:
                        retrain_freq = 12
                    elif freq == "B":
                        retrain_freq = 126
                    elif freq == "D":
                        retrain_freq = 183
                    else:
                        retrain_freq = autosarimax_refit_interval * H

                    retrain = retrain_freq

                    model.fit(series=y_train_transformed[:init_trainsize], verbose=False)
                    historical_predictions_model = model.historical_forecasts(
                        series=y_train_transformed,
                        start=init_trainsize,
                        stride=1,
                        retrain=retrain, # retrain = 12 for monthly data and retrain =
                        forecast_horizon=1,
                        past_covariates=X_train_transformed,  # Provide covariates
                        show_warnings=False,
                        verbose=False
                    ).pd_dataframe()
                else:
                    historical_predictions_model = model.historical_forecasts(
                        series=y_train_transformed,
                        start=init_trainsize,
                        stride=1,
                        forecast_horizon=1,
                        past_covariates=X_train_transformed,  # Provide covariates
                        show_warnings=False,
                    ).pd_dataframe()

                # Transforming back to PeriodIndex for better readability
                period_freq = (
                    "M" if target.index.freqstr == "MS" else target.index.freqstr
                )
                historical_predictions_model.set_index(
                    pd.PeriodIndex(
                        pd.to_datetime(historical_predictions_model.index), freq=period_freq
                    ),
                    inplace=True,
                )

                # out-of-sample predictions:
                if fh is not None:
                    vprint(f"Now performing corresponding out-of-sample predictions...")
                    if covtreatment_bool:
                        model.fit(series=target_transformed, past_covariates=covariates_transformed)
                        future_predictions_model = model.predict(
                            n=fh,
                            #series=target_transformed,
                            past_covariates=covariates_transformed,  # Provide covariates
                            #verbose=verbose,
                            show_warnings=False,
                        ).pd_dataframe()
                    else:
                        model.fit(series=target_transformed)
                        future_predictions_model = model.predict(
                            n=fh,
                            #series=target_transformed,
                            verbose=False,
                            show_warnings=False,
                        ).pd_dataframe()

                    future_predictions_model.set_index(
                    pd.PeriodIndex(
                        pd.to_datetime(future_predictions_model.index), freq=period_freq
                    ),
                        inplace=True,
                    )


            # sktime forecasters
            elif "sktime" in package_name:
                # Adjust sktime specific parameters, in particular: Seasonal periodicity
                if (
                        "Naive" not in forecaster_name and "sp" in model.get_params().keys()
                ):  # sNaive performs bad. Does not make sense to use this.
                    model.set_params(**{"sp": inferred_seasonal_freq})

                # all sktime forecasters but ARIMA
                if "ARIMA" not in forecaster_name:
                    # sktime uses ExpandingWindowSplitter and .update_predict() method for historical forecasts
                    cv = ExpandingWindowSplitter(
                        fh=1, initial_window=init_trainsize, step_length=1
                    )
                    model.fit(y_train_transformed[:init_trainsize])
                    historical_predictions_model = model.update_predict(
                        y=y_train_transformed,
                        X=X_train_transformed,
                        cv=cv
                    )

                # Extra treatment for ARIMA model
                else:
                    """
                    Updating and Refitting each period with method above would take too much time here.
                    Outlook:
                    - Consider implementing UpdateRefitEvery() wrapper from sktime package (threw an error)
                    - source this piece of code out
                    - and make own .update_predict method for ARIMA (wrap in class)

                    Define at what interval ARIMA model is being refitted.
                    autosarimax_refit_interval is between 0 and 1.
                    Default: 0, thus: refitting each period.
                    1 would mean: no refitting at all.
                    Example 0.33, means: only refitting at 33% and 66% of predictions made.
                    + Fitting at the beginning.
                    Fitting works with the AutoArima approach from Hyndman, RJ and Khandakar, Y (2008).
                    """

                    # Determine the refit frequency for the ARIMA model based on autosarimax_refit_interval
                    # If autosarimax_refit_interval is 0 or None, set refit_freq to 1, indicating refitting every period
                    if autosarimax_refit_interval in [0, None]:
                        refit_freq = 1
                    else:
                        # Calculate the refit frequency based on the provided refitting interval
                        refit_freq = ceil(H / (1 / autosarimax_refit_interval))

                    # Printing refitting information
                    if refit_freq == 2:
                        vprint(
                            f"Auto-fitting model. Refitting every {refit_freq}nd period."
                        )
                    elif refit_freq == 1:
                        vprint("Auto-fitting model. Refitting every period.")
                    else:
                        vprint(
                            f"Auto-fitting model. Refitting every {refit_freq}th period."
                        )

                    # Determine if lagged transformation affects positional indices
                    # sktime.covariates transformer removes the first period due to NaNs => positional indices change
                    lag_indicator = 1 if "covariates" in package_name else 0

                    # We are at period t+k and forecast period t+k+1
                    # Loop until all H periods are forecasted
                    # thus: k = [0, ... , H-1]
                    # Iterate over forecast periods to refit and update the model
                    for k in range(H):
                        # Calculate the size of the current training set
                        current_trainsize = init_trainsize - lag_indicator + k
                        current_y_train_arima = y_train_transformed[:current_trainsize]

                        current_X_train_arima = (
                            X_train_transformed[:current_trainsize]
                            if X_train_transformed is not None
                            else None
                        )

                        # Refit or update AutoSARIMA(X) Model
                        if k % refit_freq == 0:
                            # Refit the model at the start and every 'refit_freq' period thereafter
                            # (potentially speeds up fitting)
                            if k != 0:
                                # Update model with previous parameters for efficiency
                                sarima_fitted_params = model.get_fitted_params(
                                    deep=True
                                )
                                p, d, q = sarima_fitted_params["order"]
                                P, D, Q, sp = sarima_fitted_params["seasonal_order"]

                                updated_params = {
                                    "start_p": p,
                                    "d": d,
                                    "start_q": q,
                                    "start_P": P,
                                    "D": D,
                                    "start_Q": Q,
                                    "sp": sp,
                                    "maxiter": 15,
                                }
                                model.set_params(**updated_params)
                                ### vprint('...automatic refitting...')

                            # Fit (first time) or refit the model
                            model.fit(y=current_y_train_arima, X=current_X_train_arima)

                        # Update:
                        else:
                            # Update model parameters/coefficients in all other periods
                            model.update(
                                y=current_y_train_arima, X=current_X_train_arima
                            )

                        # Print forecast update
                        if k + 1 in printed_k:
                            vprint(f"...forecast {k + 1} / {H} done")

                        # Predict:
                        # Select last known X as predictor if using a covariate model
                        X_pred_sarimax = (
                            X_train_transformed[
                                current_trainsize: current_trainsize + 1
                            ]
                            if covtreatment_bool
                            else None
                        )
                        # Perform historical one step ahead prediction
                        prediction = model.predict(fh=1, X=X_pred_sarimax)
                        historical_predictions_model = pd.concat(
                            [historical_predictions_model, prediction], axis=0
                        )

                # out-of-sample predictions for sktime model:
                if fh is not None:
                    # Perform predictions
                    vprint(f"Performing out-of-sample predictions...")
                    model.fit(fh=list(range(1, fh+1)), y=target_transformed, X=covariates_transformed)
                    # Note: do not take the transformed X for predictions; we require the "real" next observation
                    if covtreatment_bool and fh > 1:
                        raise ValueError("For now it is only supported to make one-step-ahead future predictions when "
                                         "using covariate models using past observations (lags).")

                    X_predict_sk = X_train_full
                    if covtreatment_bool:
                        X_predict_sk = X_predict_sk.copy() # Don't know if this is necessary
                        X_predict_sk.index = X_train_full.index.shift(1)

                    future_predictions_model = model.predict(X=X_predict_sk)

            # Now finished historical (and future) forecasts per model
            vprint("...finished!")

            # Store historical predictions in a new column
            historical_predictions[forecaster_name] = historical_predictions_model

            # Store future predictions in a new column
            if fh is not None:
                future_predictions[forecaster_name] = future_predictions_model

            # Save model information to avoid redundant transformations when the model source hasn't changed
            last_covtreatment_bool = covtreatment_bool
            last_package_name = package_name

    vprint("\nFinished predictions of individual forecasters!")

    # Return information about end of process and get predictions
    vprint("\nInsights into forecasters' historical predictions:",
           historical_predictions.head()
    )

    if fh is not None:
        vprint(
            "\nInsights into forecasters' future predictions:",
            future_predictions.head()
        )

    # Prepare target output, adjust its index frequency and transform to PeriodIndex
    target_output = y_train_full[init_trainsize:]
    period_freq = (
        "M" if y_train_full.index.freqstr == "MS" else y_train_full.index.freqstr
    )
    target_output.index = pd.PeriodIndex(target_output.index, freq=period_freq)
    historical_predictions.insert(0, "Target", value=target_output)

    # # Export results as .csv if a path is specified
    csv_exporter(export_path, historical_predictions)

    # Note: Consider working with DateTimeIndex frequencies again

    # Return individual predictions
    return historical_predictions, future_predictions
