from ..models.forecasting_models import models
import pandas as pd
from utils.helpers import transform_to_darts_format
from sktime.split import ExpandingWindowSplitter
import warnings
import os

# Turn off warnings
warnings.filterwarnings('ignore')


# vorübergehend ist Datenimport + Preprocessing noch in diesem File, wird dann aber irgendwann outgesourced
# für die Modelle/Pipeline müssen preprocessede Daten folgendes Format haben:
# Target heißt "y"
# Covariates heißen "x1", "x2", "x.."
# "Date Spalte beinhaltet Zeitindex im Format DD-MM-YYYY"



# Define the directory path and file name
sim_dir = r'C:/Users/Work/OneDrive/GAU/3. Semester/Statistisches Praktikum/Git/NEW_Ensemble_Techniques_TS_FC/project/data/simulations/'
file_name = 'noisy_simdata.csv'

# Combine the directory path and file name
file_path = sim_dir + file_name

# Read and preprocess Dataset
df = pd.read_csv(file_path)
df['Date'] = pd.PeriodIndex(pd.to_datetime(df['Date'], format='%d-%m-%Y'), freq='M')
df.set_index('Date', inplace=True)


# Convert DataFrame to TimeSeries and split target and predictors
predictors_names = ["x" + str(x + 1) for x in range(df.shape[1]-1)] # allows that the number of predictors can be generic; ensure in data preprocessing that target is named 'y' and predictors 'x1', 'x2' etc.
# Indicator if DataFrame includes covariates
contains_covariates = bool(predictors_names)

# Training Subset
fc_train_split = 0.3
init_train_size = int(df.shape[0] * fc_train_split)
y_train_full = df['y']
y_train_init = df['y'][:init_train_size]

if contains_covariates:
    X_train_full = df[predictors_names]

# Create a DataFrame to store predictions
predictions = pd.DataFrame()
predictions.index.name = "Date" 

for model_name, model in models.items():
    # Skip covariate models if dataset does not include covariates
    if not contains_covariates and (model_name == "AutoSARIMAX" or model_name == "XGBoost (+ covs)"):
        continue
      
    # Fit the model and make historical expanding window one-step ahead predictions   
    print(f'Now generating expanding window predictions for Model: {model_name}')
    
    
    # Darts models need different input format
    if "XGBoost" in model_name:
        y_train_darts = transform_to_darts_format(y_train_full)
        if "covs" in model_name: 
            X_train_darts = transform_to_darts_format(X_train_full)
            pred = model.historical_forecasts(series=y_train_darts, start=init_train_size, stride=1, forecast_horizon=1, future_covariates=X_train_darts, show_warnings=False).pd_dataframe()
        else:
            pred = model.historical_forecasts(series=y_train_darts, start=init_train_size, stride=1, forecast_horizon=1, show_warnings=False).pd_dataframe()
        pred.set_index(pd.PeriodIndex(pd.to_datetime(pred.index, format='%d-%m-%Y'), freq='M'), inplace=True)
        
    elif "AutoSARIMA" in model_name:
        if model_name == "AutoSARIMAX":
            X_train_full_SARIMAX = X_train_full.copy().shift(1)
            X_train_full_SARIMAX.dropna(inplace=True)
            y_train_full_SARIMAX = y_train_full[1:].copy()
            cov_bool = True
        else:
            X_train_full_SARIMAX = None
            y_train_full_SARIMAX = y_train_full.copy()
            cov_bool = False
        
        model_sarima = model.clone()
        sarima_predictions = pd.DataFrame()
        sarima_predictions.index.name = "Date" 
        
        H = y_train_full.shape[0] - init_train_size
        H_range = range(y_train_full.shape[0] - init_train_size)
        for h in H_range:
            # note: Implementation of UpdateRefitEvery class... (consider later)
                # this will probably speed up process
                # e.g., refit model every 6th or 12th period
                # I will do that romorrow
            if h == 0 or (h+1) == H or ((h+1) % 5) == 0:
                print(f"{model_name} forecast {h+1} / {H}")
            model_sarima.fit(y_train_full_SARIMAX[:(init_train_size+h-cov_bool)], X=X_train_full_SARIMAX) 
            if cov_bool is True:
                X_pred_SARIMAX = X_train_full_SARIMAX[init_train_size+h-1:init_train_size+h]
            else:
                X_pred_SARIMAX = None
            pr = model_sarima.predict(1, X=X_pred_SARIMAX)
            # model update params (new model uses old params as initialization => faster autofit)
            sarima_predictions = pd.concat([sarima_predictions, pr], axis = 0)
            p, d, q = model_sarima.get_fitted_params()['order']
            P, D, Q, sp = model_sarima.get_fitted_params()['seasonal_order']
            model_sarima = model.clone()
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
            model_sarima.set_params(**updated_params)
        pred = sarima_predictions
    else:
        cv = ExpandingWindowSplitter(fh=1, initial_window=init_train_size, step_length=1)
        model.fit(y_train_init)  
        pred = model.update_predict(y_train_full, cv)
    
    # Store predictions in a new column
    print("...finished!\n")
    predictions[model_name] = pred

predictions.insert(0, "Actual", value=y_train_full[init_train_size:])    
print(predictions)

# Because training takes so long, interim results are (for now) exported as csv to work on ensembles for pipeline
export_path = sim_dir = r'C:/Users/Work/OneDrive/GAU/3. Semester/Statistisches Praktikum/Git/NEW_Ensemble_Techniques_TS_FC/project/interim_results/'
predictions.to_csv(os.path.join(export_path, f"historical_forecasts.csv"), index=True)



# ToDos:
# - implement faster SARIMA(X) updater class => UpdateRefitEvery()
# - Decide which forecasts model to choose
# - continue with ensemble methods in pipeline

