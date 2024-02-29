from models.forecasting_models import models
from pipeline.pipe1_data_preprocessing import target, covariates



# todo: paths generisch definieren anhand von project libray
export_path = r'C:/Users/Work/OneDrive/GAU/3. Semester/Statistisches Praktikum/Git/NEW_Ensemble_Techniques_TS_FC/project/interim_results/'


# Funktion ab hier. Das oben muss noch ins preprocessing

# comments fehlen noch

###############################################################
# inputs:
# preprocessed data (generic format):  target, covariates
###############################################################

def pipe2_individual_forecasts(models, target, covariates=None, init_train_ratio=0.3, csv_export=False, autosarimax_refit_interval=0.25):
    
    print("=======================================================")
    print("== Starting Step 2 in Pipeline: Individual Forecasts ==")
    print("=======================================================")
    
    import pandas as pd  
    import os
    import warnings
    from utils.helpers import transform_to_darts_format
    from sktime.split import ExpandingWindowSplitter
    
    # Turn off warnings
    warnings.filterwarnings('ignore')
    
    # Training Subset
    print("Splitting data for individual forecasts")
    init_train_size = int(target.shape[0] * init_train_ratio)
    y_train_full = target

    X_train_full = covariates
    
    # Create a DataFrame to store all models' predictions
    individual_predictions = pd.DataFrame()
    individual_predictions.index.name = "Date" 

    # Define full forecast horizon
    H = y_train_full.shape[0] - init_train_size

    for model_name, model in models.items():

        # Skip covariate models if no covariates are provided
        if (model_name == "AutoSARIMAX" or model_name == "XGBoost (+ X)") and covariates is None:
            continue
        
        # Fit the model and make historical expanding window one-step ahead predictions   
        print(f'Now generating {H} expanding window predictions for individual model: {model_name}')
        
        # Darts models need different input format
        if "XGBoost" in model_name:
            y_train_darts = transform_to_darts_format(y_train_full)
            
            X_train_darts = transform_to_darts_format(X_train_full) if model_name != "XGBoost" else None
            
            model_predictions = model.historical_forecasts(series=y_train_darts, start=init_train_size, stride=1, forecast_horizon=1, past_covariates=X_train_darts, show_warnings=False).pd_dataframe()
            model_predictions.set_index(pd.PeriodIndex(pd.to_datetime(model_predictions.index), freq=target.index.freq), inplace=True)
            
        elif "AutoSARIMA" in model_name:
            
            if model_name == "AutoSARIMAX":
                lag_indicator = True
                # Lag X by one period (we only know value at time of prediction)
                X_train_full_lagged = X_train_full.shift(1).dropna()
                init_train_size_X = init_train_size - 1
            else:
                lag_indicator = False
                init_train_size_X = init_train_size
                    
        
            infered_sp = 12 if target.index.freq=="M" else NotImplementedError("Implement me for daily etc")
            model.set_params(**{'sp':infered_sp})
            model.set_tags(**{"X-y-must-have-same-index": False, 'handles-missing-data': True})
            
            model_predictions = pd.DataFrame()
            model_predictions.index.name = "Date" 
            
            # Define at what frequency ARIMA model is refitted
            
            refit_freq = H // (1/autosarimax_refit_interval) # 25 % intervals => consider deacreasing this to 20% or 10%
            
            print("Auto-fitting model...")
            
            # In loop we forecast are at period t+k and forecast period t+k+1 until all H periods are forecasted
            for k in range(H):
                
                current_y_train_ARIMA = y_train_full[int(lag_indicator):(init_train_size+k)]

                current_X_train_ARIMA = X_train_full_lagged[:(init_train_size_X+k)] if model_name == "AutoSARIMAX" else None
                
                # Refit ARIMA model (including order) at period 0 and each "refit_freq"th period
                if k % refit_freq == 0:
                    if k != 0:
                        # Initialize model with previous parameters (speed up fitting)
                        sarima_fitted_params = model.get_fitted_params(deep=True)
                        p, d, q = sarima_fitted_params['order']
                        P, D, Q, sp = sarima_fitted_params['seasonal_order']

                        # Todo: check if it takes updated params (trace = True)
                        updated_params = params = {
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
                        print("...automatic refitting...")
                    model.fit(y=current_y_train_ARIMA, X=current_X_train_ARIMA) 
                else:
                # In all other periods just update parameters/coefficients
                    model.update(y=current_y_train_ARIMA, X=current_X_train_ARIMA)
                
                if model_name == "AutoSARIMAX":
                    # last known X as predictor
                    X_pred_SARIMAX = X_train_full_lagged[init_train_size_X+k:init_train_size_X+k+1]
                else:
                    X_pred_SARIMAX = None
                
                if k == 0 or (k+1) == H or ((k+1) % 5) == 0:
                    print(f"{model_name} forecast {k+1} / {H}")
                prediction = model.predict(1, X=X_pred_SARIMAX)
                model_predictions = pd.concat([model_predictions, prediction], axis = 0)
                
        else:
            cv = ExpandingWindowSplitter(fh=1, initial_window=init_train_size, step_length=1)
            model.fit(y_train_full[:init_train_size])  
            model_predictions= model.update_predict(y_train_full, cv)
        
        # Store predictions in a new column
        print("...finished!\n")
        individual_predictions[model_name] = model_predictions

    individual_predictions.insert(0, "Target", value=y_train_full[init_train_size:])    

    if isinstance(csv_export, (os.PathLike, str)):
        # todo:if path not defined export in working directory
        print("Exporting individual forecasts as csv...")
        individual_predictions.to_csv(os.path.join(csv_export, f"historical_forecasts.csv"), index=True)
        print("...finished!\n")
    
    return individual_predictions


individual_predictions = pipe2_individual_forecasts(models=models, target=target, covariates=covariates, init_train_ratio=0.3)
print(individual_predictions)