# External Modules
from darts import TimeSeries
from darts.models import XGBModel, NaiveDrift, NaiveMovingAverage, AutoARIMA
import pandas as pd



# Naive models
model_naive = NaiveMovingAverage(input_chunk_length=1)
model_naive_drift = NaiveDrift()

# Not-so naive models
model_XGB = XGBModel(lags=12)
model_AutoARIMA = AutoARIMA()


models = [model_naive, model_naive_drift, model_XGB, model_AutoARIMA]





def hfc_models(models, target_series):

    for model in models:
        historical_fc = model.historical_forecasts(
            target_series,
            start = 0.3,
            forecast_horizon = 1,
            verbose = True,
            )
    
    