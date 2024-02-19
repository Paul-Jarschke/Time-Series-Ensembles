import pandas as pd

# Read data
df = pd.read_csv('eurusd_df.csv')

# Drop NaN values for when trading is closed
data = df.dropna()

# Validation for nan removal
if data.isnull().sum().sum() == 0:
    print("There are no NaN or null values in the DataFrame after cleaning.")
else:
    print("There are still NaN or null values in the DataFrame after cleaning.")