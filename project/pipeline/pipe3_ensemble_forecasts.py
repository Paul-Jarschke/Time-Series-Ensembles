import pandas as pd
from utils.helpers import vprint
from utils.ensembling import get_ensemble_prediction
import os

# For debugging
# from paths import *
# individual_predictions = pd.read_csv(os.path.join(EXPORT_DIR, "individual_predictions.csv"), sep=";", index_col=0)
# individual_predictions = individual_predictions.drop(columns=['Theta']) # remove me later!


def pipe3_ensemble_forecasts(individual_predictions, methods,
                             select_methods='all',
                             ensemble_init_train=0.3,
                             csv_export=False, verbose=False,
                             *args, **kwargs):

    vprint("\n============================================="
           "\n== Step 3: Historical Ensemble Predictions =="
           "\n=============================================\n")

    # methods is a dictionary of weighting schemes and metamodels (docstring follows)

    # Ensemble Train Split
    n_predictions = len(individual_predictions)
    ens_init_train_size = int(ensemble_init_train * n_predictions)
    # At this period ensemble training ends end ensemble forecast is produced for ens_init_train_size + 1

    H_ens = n_predictions - ens_init_train_size

    vprint(f"Splitting forecast data (n = {n_predictions}) for ensemble forecasts (train/test ratio: "
          f"{int(ensemble_init_train * 100)}/{int(100 - ensemble_init_train * 100)})...")
    vprint(f"Initial training set has {ens_init_train_size} observations and goes from "
          f"{individual_predictions.index[0]} to {individual_predictions.index[ens_init_train_size-1]}")
    vprint(f"There are {H_ens} periods to be forecasted by the individual models "
          f"{individual_predictions.index[ens_init_train_size]} to {individual_predictions.index[-1]}")

    model_ens_predictions = pd.Series()
    ens_predictions_df = pd.DataFrame()

    for ensemble_approach, methods_dict in methods.items():

        # Remove methods specified by user from current methods dictionary
        methods_to_remove = [method for method in methods_dict.keys() if method not in select_methods]
        for method in methods_to_remove:
            methods_dict.pop(method)

        for method_name, method in methods_dict.items():

            ens_col_name = f"{ensemble_approach.capitalize()} Ensemble: {method_name}"
            if isinstance(method, dict):
                args = method.get('args')
                method = method.get('model')
            else:
                args = {}

            vprint(f'\nNow generating {H_ens} one-step ahead expanding window predictions from ensemble model: '
                   f'\'{ensemble_approach.capitalize()} - {method_name}\'')

            for i, fc_period in enumerate(range(ens_init_train_size, n_predictions)):

                if (i + 1) == 1 or (i + 1) == (n_predictions - ens_init_train_size) or (i + 1) % 20 == 0:
                    vprint(f'Ensemble forecast {i + 1} / {n_predictions - ens_init_train_size}')
                # fc_period: Period at which forecast is made
                past_individual_predictions = individual_predictions.iloc[0:fc_period, :]

                next_individual_predictions = individual_predictions.iloc[fc_period:fc_period + 1, :]

                next_ens_prediction = get_ensemble_prediction(
                    past_individual_predictions, next_individual_predictions,
                    method=ensemble_approach, model=method, verbose=verbose, **args
                )

                if i == 0:
                    model_ens_predictions = next_ens_prediction
                else:
                    model_ens_predictions = pd.concat([model_ens_predictions, next_ens_prediction])

            vprint("...finished!")
            ens_predictions_df[ens_col_name] = model_ens_predictions

    vprint("\nEnsemble predictions finished!\n"
           "\nInsights into ensembles' predictions:\n", ens_predictions_df.head(),
           "\n\nMerging...")

    # Merge with individual predictions
    full_predictions = ens_predictions_df.merge(individual_predictions, left_index=True, right_index=True, how='left')

    if isinstance(csv_export, (os.PathLike, str)):
        vprint("\nExporting ensemble forecasts as csv...")
        ens_predictions_df.to_csv(os.path.join(csv_export, f"ensemble_predictions.csv"), index=True)
        full_predictions.to_csv(os.path.join(csv_export, f"full_predictions.csv"), index=True)

    vprint("...finished!\n")
    # vprint(full_predictions.head(), "\n")

    return full_predictions

# For debugging
# pipe3_ensemble_forecasts(individual_predictions, csv_export=EXPORT_DIR)
