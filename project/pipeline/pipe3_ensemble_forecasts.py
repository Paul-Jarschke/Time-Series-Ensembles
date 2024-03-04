import pandas as pd
import numpy as np
from math import ceil
from utils.helpers import vprint, csv_exporter
from utils.ensembling import get_ensemble_prediction

# For debugging
# from paths import *
# individual_predictions = pd.read_csv(os.path.join(EXPORT_DIR, 'individual_predictions.csv'), sep=';', index_col=0)
# individual_predictions = individual_predictions.drop(columns=['Theta']) # remove me later!


def pipe3_ensemble_forecasts(individual_predictions, methods,
                             select_methods='all',
                             ensemble_init_train=0.3,
                             export_path=None, verbose=False,
                             *args, **kwargs):

    vprint('\n============================================='
           '\n== Step 3: Historical Ensemble Predictions =='
           '\n=============================================\n')

    # methods is a dictionary of weighting schemes and metamodels (docstring follows)

    # Ensemble Train Split
    n_predictions = len(individual_predictions)
    ens_init_trainsize = int(ensemble_init_train * n_predictions)
    # At this period ensemble training ends end ensemble forecast is produced for ens_init_trainsize + 1

    # Ensemble models' forecast horizon
    H_ensemble = n_predictions - ens_init_trainsize

    vprint(f'Splitting forecast data (n = {n_predictions}) for ensemble forecasts (train/test ratio: '
           f'{int(ensemble_init_train * 100)}/{int(100 - ensemble_init_train * 100)})...')
    vprint(f'Initial training set has {ens_init_trainsize} observations and goes from '
           f'{individual_predictions.index[0]} to {individual_predictions.index[ens_init_trainsize-1]}')
    vprint(f'There are {H_ensemble} periods to be forecasted by the individual models '
           f'{individual_predictions.index[ens_init_trainsize]} to {individual_predictions.index[-1]}')

    model_ens_predictions = pd.Series()
    ensemble_predictions = pd.DataFrame()

    # Set percentage interval for printing forecast updates
    # (e.g., print 0.2 means printing at 0%, 20%, 40%, 60%, 80%, and 100%)
    # Include first and last prediction in console output
    printout_percentage_interval = 0.2
    printed_k = ([ceil(x) for x in np.arange(0, 1 + printout_percentage_interval, printout_percentage_interval)]
                 * H_ensemble)

    # Ensemble approach: weighted and meta
    for ensemble_approach, methods_dict in methods.items():

        # Remove methods specified by user from current methods dictionary
        methods_to_remove = [method for method in methods_dict.keys() if method not in select_methods]
        for method in methods_to_remove:
            methods_dict.pop(method)

        # Methods:
        for method_name, method in methods_dict.items():

            ens_col_name = f'{ensemble_approach.capitalize()} Ensemble: {method_name}'
            if isinstance(method, dict):
                args = method.get('args')
                method = method.get('model')
            else:
                args = {}

            vprint(f'\nNow generating {H_ensemble} one-step ahead expanding window predictions from ensemble model: '
                   f'\'{ensemble_approach.capitalize()} - {method_name}\'')

            # Expanding window approach starts here
            # We are at period t+k and forecast period t+k+1
            # Loop until until all H periods are forecasted
            # thus: k = [0, ... , H-1]
            for k in range(H_ensemble):
                # Print forecast updates (console output)
                if k in printed_k:
                    vprint(f'Ensemble forecast {k + 1} / {H_ensemble}')
                # Current train size = Period at which forecast is made:
                current_trainsize = ens_init_trainsize + k
                past_individual_predictions = individual_predictions.iloc[0:current_trainsize, :]

                next_individual_predictions = individual_predictions.iloc[current_trainsize:
                                                                          current_trainsize+1, :]

                next_ens_prediction = get_ensemble_prediction(
                    past_individual_predictions, next_individual_predictions,
                    method=ensemble_approach, model=method, verbose=verbose, **args
                )

                # Append to DataFrame

                if k == 0:
                    model_ens_predictions = next_ens_prediction
                else:
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
