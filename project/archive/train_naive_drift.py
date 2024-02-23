import pandas as pd
import numpy as np
from darts import TimeSeries
from sktime.forecasting.naive import NaiveForecaster


# Define the directory path and file name
sim_dir = r'C:/Users/Work/OneDrive/GAU/3. Semester/Statistisches Praktikum/Git/NEW_Ensemble_Techniques_TS_FC/project/data/simulations/'
file_name = 'noisy_simdata.csv'

# Combine the directory path and file name
file_path = sim_dir + file_name

# Read and preprocess Dataset
df = pd.read_csv(file_path)
df['Date'] = pd.PeriodIndex(pd.to_datetime(df['Date'], format='%d-%m-%Y'), freq='M')
df.set_index('Date', inplace=True)

# Now you can work with the data DataFrame
# For example, you can print the first few rows

# The following data preprocessing should be above the final wrapper function


# Convert DataFrame to TimeSeries and split target and predictors
predictors_names = ["x" + str(x + 1) for x in range(df.shape[1]-1)] # allows that the number of predictors can be generic; ensure in data preprocessing that target is named 'y' and predictors 'x1', 'x2' etc.
df[predictors_names] = df[predictors_names].shift(1)
df.dropna(inplace=True)


# Split the data into training and validation sets
train_split = 0.3
train_size = int(df.shape[0] * train_split)
train = df[:train_size]

fc_horizon = 1
current_predictors = None

# Setting up Auto-SARIMAX Model without covariates
model = NaiveForecaster(
    strategy="drift"
)

# Train Model on Training Data
model.fit(train['y'])

prediction = model.predict(fc_horizon)

