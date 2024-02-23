#from pipeline.pipe2_forecasts import predictions

# for now CSV import; later: uncomment import from pipe2 (for debugging it takes to long to run full pipe for now)

import pandas as pd
import numpy as np
from utils.weigthing_methods import *

# Define the file path
file_path = r'C:\Users\Work\OneDrive\GAU\3. Semester\Statistisches Praktikum\Git\NEW_Ensemble_Techniques_TS_FC\project\interim_results\historical_forecasts.csv'

# Import the CSV file into a DataFrame
predictions = pd.read_csv(file_path, index_col='Date', delimiter=';').iloc[:,:-1]
predictions.rename(columns={'Actual': 'Target'}, inplace=True)

# Display the DataFrame
print(predictions)

#####################################
## Historical Ensemble Predictions ##
#####################################

# Ensemble Train Split
n_predictions = predictions.shape[0]
ens_train_split = 0.3
end_ens_training = int(ens_train_split*n_predictions) # At this period ensemble training ends end ensemble forecast is produced for end_ens_training + 1

# Set Up Ensemble Forecast Dataset
ensemble_predictions = pd.DataFrame(columns = ["Date", "fc_Ensemble_Simple", "fc_Ensemble_RSME", "fc_Ensemble_Variance", "fc_Ensemble_ErrorCorrelation", "fc_Ensemble_Metamodel_SVR", "fc_Ensemble_RandomForest"])


for i, fc_period in enumerate(range(end_ens_training, n_predictions)):
    if i+1 == 1 or i==(n_predictions-end_ens_training) or (i+1) % 10 == 0:
        print(f'Ensemble forecast {i+1} / {n_predictions-end_ens_training}')
    # Periode an der vorgecastet wird = fc_period
    current_ensemble_predictions = []
    current_train = predictions.iloc[0:fc_period,]

    individual_preds_next = predictions.iloc[fc_period:fc_period+1,]

    current_ensemble_predictions.append(predictions.index[fc_period])

    current_ensemble_predictions.append(float(simple_average(individual_preds_next)))
    current_ensemble_predictions.append(float(ensemble_predictions_given_weights(individual_preds_next, compute_rmse_weights(current_train)))) 
    current_ensemble_predictions.append(float(ensemble_predictions_given_weights(individual_preds_next, compute_variance_weights(current_train))))
    current_ensemble_predictions.append(float(ensemble_predictions_given_weights(individual_preds_next, compute_error_correlation_weights(current_train,verbose = False))))
    current_ensemble_predictions.append(float(metamodel_svr(current_train, individual_preds_next)))      
    current_ensemble_predictions.append(float(metamodel_random_forest(current_train, individual_preds_next)))

    ensemble_predictions.loc[len(ensemble_predictions)] = current_ensemble_predictions
        
print(ensemble_predictions)
print("...finished!")

# Missing:
# - append to normal forecasts