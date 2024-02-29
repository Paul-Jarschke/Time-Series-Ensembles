from pipeline.pipe2_individual_forecasts import individual_predictions

print(individual_predictions)

export_path = r'C:/Users/Work/OneDrive/GAU/3. Semester/Statistisches Praktikum/Git/NEW_Ensemble_Techniques_TS_FC/project/interim_results/'

# todo: comment and streamlining

def pipe3_ensemble_forecasts(individual_predictions, ens_init_train_ratio=0.3, csv_export=False):
    
    import pandas as pd
    from utils.ensembling_methods import simple_average, compute_rmse_weights, compute_variance_weights, compute_error_correlation_weights, ensemble_predictions_given_weights, metamodel_random_forest, metamodel_svr
    import os
    
    print("#############################################")
    print("## Step 3: Historical Ensemble Predictions ##")
    print("#############################################")
    
    # Ensemble Train Split
    n_predictions = individual_predictions.shape[0]
    end_ens_training = int(ens_init_train_ratio*n_predictions) # At this period ensemble training ends end ensemble forecast is produced for end_ens_training + 1

    # Set Up Ensemble Forecast Dataset
    ensemble_predictions = pd.DataFrame(columns = ["Date", "Ensemble_Simple", "Ensemble_RSME", "Ensemble_Variance", "Ensemble_ErrorCorrelation", "Ensemble_Metamodel_SVR", "Ensemble_RandomForest"])


    for i, fc_period in enumerate(range(end_ens_training, n_predictions)):
        if i+1 == 1 or i+1==(n_predictions-end_ens_training) or (i+1) % 10 == 0:
            print(f'Ensemble forecast {i+1} / {n_predictions-end_ens_training}')
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
    print(ensemble_predictions)
    print("...ensemble predictions finished!")

    # Append to individual predictions:
    print("...merging...")
    full_predictions = ensemble_predictions.merge(individual_predictions, left_index=True, right_index=True, how='left')
    print(full_predictions)
    print("...finished!")

    if isinstance(csv_export, (os.PathLike, str)):
        # todo:if path not defined export in working directory
        print("Exporting individual forecasts as csv...")
        full_predictions.to_csv(os.path.join(csv_export, f"full_predictions.csv"), index=True)
        print("...finished!")
    
    return full_predictions


full_predictions = pipe3_ensemble_forecasts(individual_predictions=individual_predictions, ens_init_train_ratio=0.3, csv_export=export_path)