import warnings
from math import ceil

import numpy as np
import pandas as pd
from sktime.split import ExpandingWindowSplitter

from utils.helpers import vprint, csv_exporter
from utils.mappings import SEASONAL_FREQ_MAPPING
from utils.transformers import TRANSFORMERS


def pipe2_individual_forecasts(
        target_covariates_tuple,
        forecasters,
        forecast_init_train,
        select_forecasters="all",
        autosarimax_refit_interval=0.33,
        export_path=None,
        verbose=False,
        *args, **kwargs):
    """
    Generates individual forecasts for each model specified in the forecasters' dictionary.

    Parameters:
    -----------
        target_covariates_tuple : tuple
            Tuple containing (target, covariates) as pandas DataFrames. Index containing dates 
            should be of pandas PeriodIndex format.
        forecasters (dict)
            Dictionary containing forecasters' class names, package and options.
        forecast_init_train : float
            Initial forecasters' training set size as a fraction of preprocessed data.
        select_forecasters : str or list, optional
            Specify which forecaster classes to use (default: 'all').
        autosarimax_refit_interval : float, optional
            Refit interval for AutoSARIMA model (default: 0.33, corresponds to fitting at 0%, 33%, and 66% of ensemblers
            training set).
        export_path : os.PathLike, optional
            Path to export the ensemble predictions as a CSV file. If not provided, no CSV file is exported
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
        pd.DataFrame: DataFrame containing individual predictions per forecaster model (columns).
        
    Notes:
    ------
    This function performs forecasting using the provided individual forecasting models.
    It iterates over each approach and model in the models dictionary and generates historical one-step-ahead
    predictions. Finally, if an export path is provided, it exports the ensemble predictions as a CSV file.
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
    init_trainsize = int(target.shape[0] * forecast_init_train)

    # Defining target variable for training as the target data
    y_train_full = target

    # Inferring the seasonal frequency of the target data
    inferred_seasonal_freq = (
        SEASONAL_FREQ_MAPPING[target.index.freqstr]
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
        f"Splitting data (train/test ratio: "
        f"{int(forecast_init_train * 100)}/{int(100 - forecast_init_train * 100)})..."
        f"\nInitial training set has {init_trainsize} observations " f"and goes from {target.index[0].date()} to"
        f" {target.index[init_trainsize - 1].date()}."
        f"\nThere are {H} periods to be forecasted: " f"{target.index[init_trainsize].date()} to"
        f" {target.index[-1].date()}\n",
    )

    # Creating a DataFrame to store all models' predictions
    individual_predictions = pd.DataFrame()
    individual_predictions.index.name = "Date"

    # Initializing variables for tracking the last model (model_source) source and covariate treatment (covtreatment_bool)
    last_model_source = None
    last_covtreatment_bool = None

    # Initialize transformed datasets
    y_train_transformed = None
    X_train_transformed = None

    # Setting percentage interval for printing forecast updates
    # (e.g., print 0.2 means printing at 0%, 20%, 40%, 60%, 80%, and 100%)
    # Include first and last prediction in console output
    printout_percentage_interval = 0.2
    printed_k = [
        ceil(x)
        for x in H
                 * np.arange(0, 1 + printout_percentage_interval, printout_percentage_interval)
    ]
    printed_k[0] = 1

    # Looping over different modeling approaches (with or without covariates)
    for approach, approach_dict in forecasters.items():

        # Determining if the current approach involves covariates
        covtreatment_bool = True if approach == "with_covariates" else False

        # Skipping covariates forecasters if no covariates are specified
        if covtreatment_bool and covariates is None:
            vprint(f"\nSince no covariates are given, skipping covariate forecasters.")
            continue

        # Skipping approach when no models are given
        if len(approach_dict) == 0:
            vprint(f"No models given for approach {approach}.")
            continue

        # Initializing variable last_package_name and last_covtreatment_bool such that data is not transformed twice
        last_package_name = None
        last_covtreatment_bool = None

        # Loop over individual model in the current sub-dictionary (with/without covariates)
        for model_name, MODEL in approach_dict.items():
            # Only considering selected models in the current ensemble approach dictionary
            if (
                    select_forecasters not in ["all", ["all"]]
                    and model_name not in select_forecasters
            ):
                continue

            # Defining X_train_full depending on covtreatment_bool, i.e., depending on whether covariates are specified
            X_train_full = covariates if covtreatment_bool else None

            # Extracting model information from the dictionary
            if "options" not in MODEL.keys():
                MODEL["options"] = {}
            if "package" not in MODEL.keys():
                raise RuntimeError(f"You need to provide the package for {model_name}.")
            model_function, package_name, options = (
                MODEL["model"],
                MODEL["package"],
                MODEL["options"],
            )

            # Adjusting model name depending on the approach (with/without covariates)
            model_name += " with covariates" if covtreatment_bool else ""

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
                # Transform
                y_train_transformed, X_train_transformed = transformer(
                    y_train_full, X_train_full
                )
            # No transformation possible if package_name not in TRANSFORMERS
            else:
                raise RuntimeError(f"{package_name} is not yet supported")

            # Set up empty DataFrame for predictions and define index named 'Date'
            model_predictions = pd.DataFrame()
            model_predictions.index.name = "Date"

            # Generating one-step ahead expanding window predictions for the current model (fitting, updating, predicting)
            vprint(
                f"Now generating {H} one-step ahead expanding window predictions from model: "
                f'{model_name} ({package_name.split(".")[0]})'
            )

            # Handling different prediction methods based on the package used
            # Method and data transformer are inferred from package_name
            # darts forecasters use .historical_forecasts() method:
            if "darts" in package_name:
                model_predictions = model.historical_forecasts(
                    series=y_train_transformed,
                    start=init_trainsize,
                    stride=1,
                    forecast_horizon=1,
                    past_covariates=X_train_transformed,
                    show_warnings=False,
                ).pd_dataframe()

                # Transforming back to PeriodIndex for better readability
                period_freq = (
                    "M" if target.index.freqstr == "MS" else target.index.freqstr
                )
                model_predictions.set_index(
                    pd.PeriodIndex(
                        pd.to_datetime(model_predictions.index), freq=period_freq
                    ),
                    inplace=True,
                )

            # sktime forecasters
            elif "sktime" in package_name:
                # Adjust sktime specific parameters, in particular: Seasonal periodicity
                if (
                        "Naive" not in model_name and "sp" in model.get_params().keys()
                ):  # sNaive performs bad. Does not make sense to use this.
                    model.set_params(**{"sp": inferred_seasonal_freq})

                # all sktime forecasters but ARIMA
                if "ARIMA" not in model_name:
                    # sktime uses ExpandingWindowSplitter and .update_predict() method for historical forecasts
                    cv = ExpandingWindowSplitter(
                        fh=1, initial_window=init_trainsize, step_length=1
                    )
                    model.fit(y_train_transformed[:init_trainsize])
                    model_predictions = model.update_predict(y_train_transformed, cv=cv)

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
                            # Refit the model at the start and every 'refit_freq' period thereafter(potentially speeds up fitting)
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
                            vprint(f"{model_name} forecast {k + 1} / {H}")

                        # Predict:
                        # Select last known X as predictor if using a covariate model
                        X_pred_sarimax = (
                            X_train_transformed[
                            current_trainsize: current_trainsize + 1
                            ]
                            if covtreatment_bool
                            else None
                        )
                        # Perform prediction
                        prediction = model.predict(fh=1, X=X_pred_sarimax)
                        model_predictions = pd.concat(
                            [model_predictions, prediction], axis=0
                        )

            # Store predictions in a new column after finishing historical forecasts per model
            vprint("...finished!\n")
            individual_predictions[model_name] = model_predictions

            # Save model information to avoid redundant transformations when the model source hasn't changed
            last_covtreatment_bool = covtreatment_bool
            last_package_name = package_name

    # Retunr information about end of process and get predictions
    vprint(
        "\nIndividual forecasters' predictions finished!\n"
        "\nInsights into forecasters' predictions:",
        individual_predictions.head(),
        "\n",
    )

    # Prepare target output and adjust its index frequency
    target_output = y_train_full[init_trainsize:]
    period_freq = (
        "M" if y_train_full.index.freqstr == "MS" else y_train_full.index.freqstr
    )
    target_output.index = pd.PeriodIndex(target_output.index, freq=period_freq)
    individual_predictions.insert(0, "Target", value=target_output)

    # Export results as .csv if a path is specified
    csv_exporter(export_path, individual_predictions)

    # Return individual predictions
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
