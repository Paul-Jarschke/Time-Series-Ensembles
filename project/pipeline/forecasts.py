from forecasting_models import models
import pandas as pd
import numpy as np

print(models)


# vorübergehend ist Datenimport + Preprocessing noch in diesem Fail, wird dann aber irgendwann outgesourced
# für die Modelle müssen preprocessede Daten folgendes Format haben:
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
df[predictors_names] = df[predictors_names].shift(1)
df.dropna(inplace=True)


# Training Subset
train_split = 0.3
train_size = int(df.shape[0] * train_split)
y_train = df['y'][:train_size]
X_train = df[predictors_names][:train_size]

#current_predictors = df[predictors_names][:(train_size+fc_horizon)]

# Create a DataFrame to store predictions
predictions = pd.DataFrame()
predictions.index.name = "Date" 

# Make one-step-ahead prediction for each model
fc_horizon = 1

for model_name, model in models.items():
    # Fit the model
    model.fit(y_train)  # Assuming you have training data X_train, y_train
    
    # Make predictions
    pred = model.predict(fc_horizon)
    
    # Store predictions in a new column
    predictions[model_name] = pred
    

print(predictions)

# ToDos:
# - add XGBoost (with/without covariates to forecasting_models.py)
# - Add SARIMA with covariates to forecasting_models.py
# - Implement expanding window approach (either manually with loop or using historical_forecast respectively ExpandingWindowSplitter and update_prpedict in sktime)
# - add Cubic Splines to forecasting models (maybe not)

