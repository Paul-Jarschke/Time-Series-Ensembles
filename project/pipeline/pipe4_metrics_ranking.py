# later wrapper function here will need Input full_predictions from pipe3
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error # later use Paul's functions

# Define the file path
file_path = r'C:\Users\Work\OneDrive\GAU\3. Semester\Statistisches Praktikum\Git\NEW_Ensemble_Techniques_TS_FC\project\interim_results\full_predictions.csv'

# Import the CSV file into a DataFrame
full_predictions = pd.read_csv(file_path, index_col='Date') # later obsolete

########################################
# Wrapper function content ab hier:

# Extracting the 'Target' column
Y_actual = full_predictions.pop('Target')

# Initialize an empty DataFrame to store the metrics
metrics_ranking = pd.DataFrame(columns=['Model', 'MAPE', 'RMSE', 'MAPE_Ranking'])



# Loop over each prediction column in the appended_data DataFrame
for model_name in full_predictions.columns:
    Y_predicted = full_predictions[model_name]  # Predicted values
    # Calculate MAPE and RMSE
    mape = mean_absolute_percentage_error(Y_actual, Y_predicted)
    rmse = np.sqrt(mean_squared_error(Y_actual, Y_predicted))
    # Append the metrics to the metrics_ranking DataFrame
    temp_df = pd.DataFrame({'Model': [model_name], 'MAPE': [mape], 'RMSE': [rmse], 'MAPE_Ranking': [np.nan]})
    metrics_ranking = pd.concat([metrics_ranking, temp_df], ignore_index=True)

# Rank the models based on MAPE values
metrics_ranking['MAPE_Ranking'] = [int(element) for element in metrics_ranking['MAPE'].rank()]

# Rank the models based on RMSE values
metrics_ranking['RMSE_Ranking'] = [int(element) for element in metrics_ranking['RMSE'].rank()]

# Sort the DataFrame based on MAPE values
metrics_ranking = metrics_ranking.sort_values(by='MAPE')

# Reset the index
metrics_ranking.reset_index(drop=True, inplace=True)

print(metrics_ranking)
#display(metrics_ranking.style.hide())