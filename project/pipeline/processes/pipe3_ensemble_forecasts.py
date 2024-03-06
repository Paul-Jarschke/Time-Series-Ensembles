from math import ceil

import numpy as np
import pandas as pd

from utils.helpers import vprint, csv_exporter
from utils.predictions.ensemble_predictions_wrapper import ensemble_prediction_wrapper


# For debugging
# from paths import *
# individual_predictions = pd.read_csv(os.path.join(EXPORT_DIR, 'individual_predictions.csv'), sep=';', index_col=0)
# individual_predictions = individual_predictions.drop(columns=['Theta']) # remove me later!


def pipe3_ensemble_forecasts(individual_predictions, ensemblers,
                             select_ensemblers='all',
                             ensemble_init_train=0.3,
                             export_path=None, verbose=False,
                             *args, **kwargs):

    vprint('\n============================================='
           '\n== Step 3: Historical Ensemble Predictions =='
           '\n=============================================\n')

    # ensemblers is a dictionary of weighting schemes and metamodels (docstring follows)

    # Ensemble Train Split
    n_predictions = len(individual_predictions)
    ens_init_trainsize = int(ensemble_init_train * n_predictions)
    # At this period ensemble training ends end ensemble forecast is produced for ens_init_trainsize + 1

    # Ensemble forecasters' forecast horizon
    H_ensemble = n_predictions - ens_init_trainsize

    # Printing information about prediction process
    vprint(f'Splitting forecast data (n = {n_predictions}) for ensemble forecasts (train/test ratio: '
           f'{int(ensemble_init_train * 100)}/{int(100 - ensemble_init_train * 100)})...')
    vprint(f'Initial training set has {ens_init_trainsize} observations and goes from '
           f'{individual_predictions.index[0]} to {individual_predictions.index[ens_init_trainsize-1]}')
    vprint(f'There are {H_ensemble} periods to be forecasted by the individual models '
           f'{individual_predictions.index[ens_init_trainsize]} to {individual_predictions.index[-1]}')

    # Create empty pandas data objects for results
    ensemble_predictions = pd.DataFrame()  # Resulting DataFrame containing all ensemble models' predictions

    # Set percentage interval for printing forecast updates
    # (e.g., print 0.2 means printing at 0%, 20%, 40%, 60%, 80%, and 100%)
    # Include first and last prediction in console output
    printout_percentage_interval = 0.2
    printed_k = [ceil(x) for x in
                 H_ensemble * np.arange(0, 1 + printout_percentage_interval, printout_percentage_interval)]
    printed_k[0] = 1

    # Loop over ensemble approach: weighted or meta
    for ensemble_approach, approach_dict in ensemblers.items():
        # Loop over models in the approach: e.g., weighting scheme or metamodel
        for model_name, MODEL in approach_dict.items():
            # Only consider selected models in the current ensemble approach dictionary
            if select_ensemblers != 'all' and model_name not in select_ensemblers:
                continue

            # Define column name in resulting pandas DataFramae
            ens_col_name = f'{ensemble_approach.capitalize()} Ensemble: {model_name}'

            # Extract model function, package name, and arguments from the models' dictionary
            # Thus, overwrite model dictionary with actual model
            model_function, package_name, options = MODEL['model'], MODEL['package'], MODEL['options'],

            # Expanding window approach starts here
            # We are at period t+k and forecast period t+k+1
            # Loop until all H periods are forecasted
            # thus: k = [0, ... , H-1]

            # if verbose: print current state to process
            printed_model_name = f'{model_name}' + f' ({package_name})'
            vprint(
                f'\nNow generating {H_ensemble} one-step ahead expanding window predictions from ensemble model: '
                f'\'{ensemble_approach.capitalize()} - {printed_model_name}\'')

            # Set up empty DataFrame for the predictions per model
            model_ens_predictions = pd.DataFrame()

            # Start creating ensemble predictions
            for k in range(H_ensemble):
                # Print forecast updates (console output)
                if k + 1 in printed_k:
                    vprint(f'Ensemble forecast {k + 1} / {H_ensemble}')
                # Current train size = Period at which forecast is made:
                current_trainsize = ens_init_trainsize + k
                past_individual_predictions = individual_predictions.iloc[0:current_trainsize, :]

                next_individual_predictions = individual_predictions.iloc[current_trainsize:
                                                                          current_trainsize+1, :]

                # Depending on ensemble approach 'weighted' or 'meta' the ensemble_prediction_wrapper
                # selects the appropriate approach and creates a forecast
                next_ens_prediction = ensemble_prediction_wrapper(
                    past_individual_predictions=past_individual_predictions,
                    next_indiv_predictions=next_individual_predictions,
                    approach=ensemble_approach, model_function=model_function, options=options,
                    verbose=verbose,
                )

                # Append to DataFrame
                model_ens_predictions = pd.concat([model_ens_predictions, next_ens_prediction])

            vprint('...finished!')
            ensemble_predictions[ens_col_name] = model_ens_predictions

    vprint('\nEnsemble predictions finished!\n'
           '\nInsights into ensembles\' predictions:',
           ensemble_predictions.head())

    # Merge with individual predictions
    vprint('\nMerging...')
    full_predictions = individual_predictions.merge(ensemble_predictions,
                                                    left_index=True, right_index=True, how='inner')
    vprint('...finished!\n')

    # If path is specified, export results as .csv
    csv_exporter(export_path, ensemble_predictions, full_predictions)

    return full_predictions

# For debugging
# pipe3_ensemble_forecasts(individual_predictions, export_path=EXPORT_DIR)
