import pandas as pd
import os

# Get the current directory of the script
script_dir = os.path.dirname(__file__)

# Path to the CSV file
csv_file_path = os.path.join(script_dir, 'eurusd_df.csv')

# Read data
df = pd.read_csv(csv_file_path)

# Drop NaN values for when trading is closed
data = df.dropna()

# Validation for nan removal
if data.isnull().sum().sum() == 0:
    print("There are no NaN or null values in the DataFrame after cleaning.")
else:
    print("There are still NaN or null values in the DataFrame after cleaning.")