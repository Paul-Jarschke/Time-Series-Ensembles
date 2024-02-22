# External Modules
from darts import TimeSeries
from darts.models import XGBModel
import pandas as pd


# Imports and path setting
import sys
import os

# Get the current directory of the notebook
notebook_dir = os.path.abspath('')
# Get the parent directory of the notebook directory
parent_dir = os.path.dirname(notebook_dir)
# Append the parent directory to the system path
sys.path.append(parent_dir)

# Now you can import modules from the 'data' folder
from project.data.clean_data import df


model = XGBModel(lags=12)

historical_fcast_XGB = model.historical_forecasts(
    target_series,
    start=0.3,
    forecast_horizon=1,
    verbose=True
)

target_series.plot()
historical_fcast_XGB.plot()

from darts.models import NaiveDrift

model_naive_drift = NaiveDrift()

historical_fcast_naive_drift = model_naive_drift.historical_forecasts(
    target_series,
    start=0.3,
    forecast_horizon=1,
    verbose=True,
    show_warnings = False
)

target_series.plot()
historical_fcast_naive_drift.plot()


hfc_XGB = timeseries_to_array(historical_fcast_XGB)
hfc_Naive_drift = timeseries_to_array(historical_fcast_naive_drift)
targets = timeseries_to_array(val, pop = False)


# Create a dictionary with keys as column names and values as arrays
data = {'Target': targets,
        'XGB': hfc_XGB,
        'Naive_drift': hfc_Naive_drift}

# Create DataFrame
hfc_data = pd.DataFrame(data)

# Display the DataFrame
print(hfc_data)