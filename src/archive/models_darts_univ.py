print("Loading Univariate darts forecasters...")


# External Modules
import pandas as pd
from darts.models import (
    XGBModel,
    NaiveDrift,
    NaiveMovingAverage,
    AutoARIMA,
    RandomForest,
)

# Naive forecasters
model_naive = NaiveMovingAverage(input_chunk_length=1)
model_naive_drift = NaiveDrift()

# Not-so naive forecasters
model_XGB = XGBModel(lags=12)
model_RF = RandomForest(lags=12)
model_AutoARIMA = AutoARIMA()

# Create list of forecasters
univariate_models = [model_naive, model_naive_drift, model_XGB]


def hfc_models(models, target_series):
    """
    Generate historical forecasts for a given list of forecasters and target time series.

    Args:
    - forecasters (list): List of forecasters for generating historical forecasts.
    - target_series (pandas.Series): Target time series for forecasting.

    Returns:
    - pandas.DataFrame: DataFrame containing historical forecast predictions for each model.
    """

    # Initiate dictionary for model predictions
    predictions = {}

    for model in models:

        # Extract Model name
        model_name = type(model).__name__
        print(f"Training {model_name}:")

        # Generate historical forecasts using the model
        historical_fc = model.historical_forecasts(
            target_series,
            start=0.3,
            forecast_horizon=1,
            verbose=True,
            show_warnings=False,
        )

        # Transform darts' TimeSeries to list of values
        prediction_values = list(historical_fc.values().flatten())

        # Remove last element of historical forecast, since there is no target value for comparison
        prediction_values.pop()

        # Using the model's name as the column label
        predictions[model_name] = prediction_values

    # Creating a DataFrame from the predictions dictionary
    predictions_df = pd.DataFrame(predictions)

    return predictions_df
