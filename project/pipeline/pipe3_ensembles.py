#from pipeline.pipe2_forecasts import predictions

# for now CSV import; later: uncomment import from pipe2 (for debugging it takes to long to run full pipe for now)

import pandas as pd
from utils.weigthing_methods import *
from utils.metrics import mape, rmse

# Define the file path
file_path = r'C:\Users\Work\OneDrive\GAU\3. Semester\Statistisches Praktikum\Git\NEW_Ensemble_Techniques_TS_FC\project\interim_results\historical_forecasts.csv'

# Import the CSV file into a DataFrame
df = pd.read_csv(file_path, index_col='Date', delimiter=';').iloc[:,:-1]

# Display the DataFrame
print(df)

################
## Ensembling ##
################


