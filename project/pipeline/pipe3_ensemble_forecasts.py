import pandas as pd
from utils.ensembling import get_ensemble_prediction
import os

# For debugging
# from paths import *
# individual_predictions = pd.read_csv(os.path.join(EXPORT_DIR, "historical_predictions.csv"), sep=";", index_col=0)
# individual_predictions = individual_predictions.drop(columns=['Theta']) # remove me later!


def pipe3_ensemble_forecasts(individual_predictions, methods,
                             ens_init_train_ratio=0.3, csv_export=False, verbose=False):
    if verbose:
        print("\n#############################################")
        print("## Step 3: Historical Ensemble Predictions ##")
        print("#############################################")

    # methods is a dictionary of weighting schemes and metamodels (docstring folgt)

    # Ensemble Train Split
    n_predictions = len(individual_predictions)
    ens_init_train_size = int(ens_init_train_ratio*n_predictions)
    # At this period ensemble training ends end ensemble forecast is produced for ens_init_train_size + 1

    H_ens = n_predictions - ens_init_train_size

    if verbose:
        print(f"Splitting forecast data (n = {n_predictions}) for ensemble forecasts (train/test ratio: "
              f"{int(ens_init_train_ratio*100)}/{int(100-ens_init_train_ratio*100)})...")
        print(f"Initial training set has {ens_init_train_size} observations and goes from "
              f"{individual_predictions.index[0]} to {individual_predictions.index[ens_init_train_size-1]}")
        print(f"There are {H_ens} periods to be forecasted by the individual models "
              f"{individual_predictions.index[ens_init_train_size]} to {individual_predictions.index[-1]}")

    model_ens_predictions = pd.Series()
    ens_predictions_df = pd.DataFrame()

    for method, models in methods.items():
        for model_name, model in models.items():
            ens_col_name = f"Ens_{method}_{model_name}"
            if isinstance(model, dict):
                options = model.get('options')
                model = model.get('model')
            else:
                options = {}

            if verbose:
                print(f"\nPerforming {model_name} {method} expanding window...")

            for i, fc_period in enumerate(range(ens_init_train_size, n_predictions)):
                if verbose:
                    if (i + 1) == 1 or (i + 1) == (n_predictions - ens_init_train_size) or (i + 1) % 10 == 0:
                        print(f'Ensemble forecast {i + 1} / {n_predictions - ens_init_train_size}')
                # Periode an der vorgecastet wird = fc_period
                past_individual_predictions = individual_predictions.iloc[0:fc_period, :]

                next_individual_predictions = individual_predictions.iloc[fc_period:fc_period + 1, :]

                next_ens_prediction = get_ensemble_prediction(
                    past_individual_predictions, next_individual_predictions,
                    method=method, model=model, verbose=verbose, **options
                )

                if i == 0:
                    model_ens_predictions = next_ens_prediction
                else:
                    model_ens_predictions = pd.concat([model_ens_predictions, next_ens_prediction])

            if verbose:
                print("...finished!")
            ens_predictions_df[ens_col_name] = model_ens_predictions

    if verbose:
        print("Ensemble predictions finished!")
        print(ens_predictions_df.head())
        print("Merging...")

    # Merge with individual predictions
    full_predictions = ens_predictions_df.merge(individual_predictions, left_index=True, right_index=True, how='left')

    if isinstance(csv_export, (os.PathLike, str)):
        if verbose:
            print("Exporting ensemble forecasts as csv...")
        ens_predictions_df.to_csv(os.path.join(csv_export, f"ensemble_predictions.csv"), index=True)
        full_predictions.to_csv(os.path.join(csv_export, f"full_predictions.csv"), index=True)

    if verbose:
        print("...finished!")
        # print(full_predictions.head(), "\n")

    return full_predictions

# For debugging
# pipe3_ensemble_forecasts(individual_predictions, csv_export=EXPORT_DIR)
