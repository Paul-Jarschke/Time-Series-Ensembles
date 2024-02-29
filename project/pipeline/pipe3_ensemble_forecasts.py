from utils.ensembling_methods import simple_average, compute_rmse_weights, compute_variance_weights, compute_error_correlation_weights, ensemble_predictions_given_weights, metamodel_random_forest, metamodel_svr
import pandas as pd
import os


def pipe3_ensemble_forecasts(individual_predictions, ens_init_train_ratio=0.3, csv_export=False, verbose=False):
    print("\n#############################################")
    print("## Step 3: Historical Ensemble Predictions ##")
    print("#############################################")
    
    # Ensemble Train Split
    n_predictions = individual_predictions.shape[0]
    ens_init_train_size = int(ens_init_train_ratio*n_predictions) # At this period ensemble training ends end ensemble forecast is produced for ens_init_train_size + 1
    
    H_ens = n_predictions - ens_init_train_size
    
    if verbose:
        print(f"Splitting forecast data (n = {n_predictions}) for ensemble forecasts (train/test ratio: {int(ens_init_train_ratio*100)}/{int(100-ens_init_train_ratio*100)})...")
        print(f"Initial training set has {ens_init_train_size} observations and goes from {individual_predictions.index[0]} to {individual_predictions.index[ens_init_train_size-1]}")
        print(f"There are {H_ens} periods to be forecasted by the individual models {individual_predictions.index[ens_init_train_size]} to {individual_predictions.index[-1]}")

    # Set Up Ensemble Forecast Dataset
    ensemble_predictions = pd.DataFrame(columns = ["Date", "Ensemble_Simple", "Ensemble_RSME", "Ensemble_Variance", "Ensemble_ErrorCorrelation", "Ensemble_Metamodel_SVR", "Ensemble_RandomForest"])


    for i, fc_period in enumerate(range(ens_init_train_size, n_predictions)):
        if verbose:
            if i+1 == 1 or i+1==(n_predictions-ens_init_train_size) or (i+1) % 10 == 0:
                print(f'Ensemble forecast {i+1} / {n_predictions-ens_init_train_size}')
        # Periode an der vorgecastet wird = fc_period
        current_ensemble_predictions = []
        current_train = individual_predictions.iloc[0:fc_period,]

        individual_preds_next = individual_predictions.iloc[fc_period:fc_period+1,]

        current_ensemble_predictions.append(individual_predictions.index[fc_period])

        current_ensemble_predictions.append(float(simple_average(individual_preds_next)))
        current_ensemble_predictions.append(float(ensemble_predictions_given_weights(individual_preds_next, compute_rmse_weights(current_train)))) 
        current_ensemble_predictions.append(float(ensemble_predictions_given_weights(individual_preds_next, compute_variance_weights(current_train))))
        current_ensemble_predictions.append(float(ensemble_predictions_given_weights(individual_preds_next, compute_error_correlation_weights(current_train,verbose = False))))
        current_ensemble_predictions.append(float(metamodel_svr(current_train, individual_preds_next)))      
        current_ensemble_predictions.append(float(metamodel_random_forest(current_train, individual_preds_next)))

        ensemble_predictions.loc[len(ensemble_predictions)] = current_ensemble_predictions
        
    # Set "Date" column as index and drop it
    ensemble_predictions.set_index("Date", inplace=True)
    if verbose:            
        print(ensemble_predictions)
        print("...ensemble predictions finished!")
    # Append to individual predictions:
        print("...merging...")
    full_predictions = ensemble_predictions.merge(individual_predictions, left_index=True, right_index=True, how='left')
    print(full_predictions)

    if isinstance(csv_export, (os.PathLike, str)):
        if verbose:
            print("Exporting ensemble forecasts as csv...")
        full_predictions.to_csv(os.path.join(csv_export, f"full_predictions.csv"), index=True)
        
    if verbose:  
        print("...finished!")
        print(full_predictions, "\n")
    
    return full_predictions


