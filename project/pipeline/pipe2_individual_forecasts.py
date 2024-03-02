import pandas as pd
import os
import warnings
from utils.helpers import transform_to_darts_format
from sktime.split import ExpandingWindowSplitter

# Turn off warnings
warnings.filterwarnings('ignore')


###############################################################
# inputs:
# preprocessed data (generic format):  target, covariates
###############################################################

def pipe2_individual_forecasts(models, target, covariates=None, indiv_init_train_ratio=0.3, csv_export=False,
                               autosarimax_refit_interval=0.33, verbose=False):
    if verbose:
        print("\n=======================================================")
        print("== Starting Step 2 in Pipeline: Individual Forecasts ==")
        print("=======================================================")

    # Calculate initial train size
    init_train_size = int(target.shape[0] * indiv_init_train_ratio)

    # Define training variables
    y_train_full = target

    X_train_full = covariates

    # Infer frequency
    inferred_sp = 12 if target.index.freq == "M" else NotImplementedError("Implement me for daily etc")

    # Define full forecast horizon
    H = y_train_full.shape[0] - init_train_size

    if verbose:
        print(
            f"Splitting data for individual forecasts (train/test ratio: {int(indiv_init_train_ratio * 100)}/{int(100 - indiv_init_train_ratio * 100)})...")
        print(
            f"Initial training set has {init_train_size} observations and goes from {target.index[0]} to {target.index[init_train_size - 1]}")
        print(
            f"There are {H} periods to be forecasted by the individual models {target.index[init_train_size]} to {target.index[-1]}\n")

    # Create a DataFrame to store all models' predictions
    individual_predictions = pd.DataFrame()
    individual_predictions.index.name = "Date"

    for model_name, model in models.items():
        # Skip covariate models if no covariates are provided
        if (model_name == "AutoSARIMAX" or model_name == "XGBoost (+ X)") and covariates is None:
            continue

        # Fit the model and make historical expanding window one-step ahead predictions   
        if verbose:
            print(f'Now generating {H} expanding window predictions for individual model: {model_name}')

        # Darts models need different input format
        if "XGBoost" in model_name:
            y_train_darts = transform_to_darts_format(y_train_full)

            X_train_darts = transform_to_darts_format(X_train_full) if model_name != "XGBoost" else None

            model_predictions = model.historical_forecasts(series=y_train_darts, start=init_train_size, stride=1,
                                                           forecast_horizon=1, past_covariates=X_train_darts,
                                                           show_warnings=False).pd_dataframe()
            model_predictions.set_index(pd.PeriodIndex(pd.to_datetime(model_predictions.index), freq=target.index.freq),
                                        inplace=True)

        elif "AutoSARIMA" in model_name:

            X_train_full_lagged = None

            if model_name == "AutoSARIMAX":
                lag_indicator = True
                # Lag X by one period (we only know value at time of prediction)
                X_train_full_lagged = X_train_full.shift(1).dropna()
                init_train_size_X = init_train_size - 1
            else:
                lag_indicator = False
                init_train_size_X = init_train_size

            model.set_params(**{'sp': inferred_sp})
            model.set_tags(**{"X-y-must-have-same-index": False, 'handles-missing-data': True})

            model_predictions = pd.DataFrame()
            model_predictions.index.name = "Date"

            # Define at what frequency ARIMA model is refitted

            refit_freq = (H // (
                        1 / autosarimax_refit_interval) + 1)  # 33 % intervals => consider deacreasing this to 20% or 10%

            if verbose:
                print("Auto-fitting model...")

            # In loop we forecast are at period t+k and forecast period t+k+1 until all H periods are forecasted
            for k in range(H):

                current_y_train_arima = y_train_full[int(lag_indicator):(init_train_size + k)]

                current_X_train_arima = X_train_full_lagged[:(init_train_size_X + k)] if (
                        model_name == "AutoSARIMAX") else None

                # Refit ARIMA model (including order) at period 0 and each "refit_freq"th period
                if k % refit_freq == 0:
                    if k != 0:
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
                        if verbose:
                            print("...automatic refitting...")
                    model.fit(y=current_y_train_arima, X=current_X_train_arima)
                else:
                    # In all other periods just update parameters/coefficients
                    model.update(y=current_y_train_arima, X=current_X_train_arima)

                if model_name == "AutoSARIMAX":
                    # last known X as predictor
                    X_pred_sarimax = X_train_full_lagged[init_train_size_X + k:init_train_size_X + k + 1]
                else:
                    X_pred_sarimax = None

                if k == 0 or (k + 1) == H or ((k + 1) % 10) == 0:
                    if verbose:
                        print(f"{model_name} forecast {k + 1} / {H}")
                prediction = model.predict(1, X=X_pred_sarimax)
                model_predictions = pd.concat([model_predictions, prediction], axis=0)

        else:
            cv = ExpandingWindowSplitter(fh=1, initial_window=init_train_size, step_length=1)
            model.fit(y_train_full[:init_train_size])
            model_predictions = model.update_predict(y_train_full, cv)

        # Store predictions in a new column
        if verbose:
            print("...finished!\n")
        individual_predictions[model_name] = model_predictions

    if verbose:
        print("Individual predictions finished!")
    individual_predictions.insert(0, "Target", value=y_train_full[init_train_size:])

    if isinstance(csv_export, (os.PathLike, str)):
        individual_predictions.to_csv(os.path.join(csv_export, f"individual_predictions.csv"), index=True)
        if verbose:
            print("Exporting individual forecasts as csv...")
            print("...finished!\n")

    if verbose:
        print(individual_predictions.head(), "\n")

    return individual_predictions
