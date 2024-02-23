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
end_training = int(ens_train_split*n_predictions) - 1 # At this period (index!) ensemble training ends end ensemble forecast is produced for end_training + 1

# Set Up Ensemble Forecast Dataset
ensemble_fc_data = pd.DataFrame(columns = ["Date", "fc_Ensemble_Simple", "fc_Ensemble_RSME", "fc_Ensemble_Variance", "fc_Ensemble_ErrorCorrelation", "fc_Ensemble_Metamodel_SVR", "fc_Ensemble_RandomForest"])



for fc_period in range(end_training+1, n_predictions):
    print(fc_period)
    # Periode die vorgecastet wird = fc_period
    ensemble_forecasts = []
    current_train = predictions.iloc[0:fc_period,]
    print(len(current_train))

    individual_fc_next = predictions.iloc[fc_period:fc_period+1,]
    #print(individual_fc_next)
    ensemble_forecasts.append(predictions.index[fc_period])
    #print(simple_average(individual_fc_next))
    ensemble_forecasts.append(float(simple_average(individual_fc_next)))
    ensemble_forecasts.append(float(ensemble_predictions_given_weights(individual_fc_next, compute_rmse_weights(current_train)))) 
    ensemble_forecasts.append(float(ensemble_predictions_given_weights(individual_fc_next, compute_variance_weights(current_train))))
    ensemble_forecasts.append(float(ensemble_predictions_given_weights(individual_fc_next, compute_error_correlation_weights(current_train,verbose = False))))
    ensemble_forecasts.append(float(metamodel_svr(current_train, individual_fc_next)))      
    ensemble_forecasts.append(float(metamodel_random_forest(current_train, individual_fc_next)))
    #print(ensemble_forecasts)
    #print(type(ensemble_forecasts))
    #test = pd.DataFrame(ensemble_forecasts)
    ensemble_fc_data.loc[len(ensemble_fc_data)] = ensemble_forecasts
    #ensemble_fc_data = pd.concat([ensemble_fc_data, test], axis = 0)
    
print(ensemble_fc_data)