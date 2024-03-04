from utils.helpers import vprint
import pandas as pd
import os

# for debugging:
# from paths import *
# full_predictions = pd.read_csv(os.path.join(EXPORT_DIR, "full_predictions.csv"), index_col=0)


def pipe4_metrics_ranking(full_predictions, metrics,
                          csv_export=False, sort_by="MAPE", verbose=False,
                          *args, **kwargs):

    vprint("\n============================================"
           "\n== Step 4: Creating Metrics Ranking Table =="
           "\n============================================\n")

    vprint(f'Calculating {", ".join(metrics.keys())} per model...')
    
    # Extract actual values
    Y_actual = full_predictions.pop('Target')

    # Calculate metrics per (individual and ensemble) model
    model_names = full_predictions.columns
    metrics_dict = {'Model': model_names}
    for model_name in model_names:

        Y_predicted = full_predictions[model_name]  # Predicted values

        # Calculate metrics
        for metric_name, metric_function in metrics.items():
            if metric_name not in metrics_dict:
                metrics_dict[metric_name] = []
            model_metrics = metric_function(Y_predicted, Y_actual)
            metrics_dict[metric_name].append(model_metrics)

    # Save as pandas DataFrame
    metrics_df = pd.DataFrame(metrics_dict)

    # Rank the models based on metric columns
    vprint("Ranking models ...")
    for metric_name, metric_values in metrics_df.items():
        if 'Model' in metric_name:  # No ranking for 'Model' column
            continue
        metrics_df[f'{metric_name} Ranking'] = [int(rank) for rank in metric_values.rank()]

    # Sort the DataFrame based on selected metric
    metrics_ranking = metrics_df.sort_values(by=f'{sort_by} Ranking')

    # Reset the index
    metrics_ranking.reset_index(drop=True, inplace=True)
    
    # If desired, export results as csv
    if isinstance(csv_export, (os.PathLike, str)):
        metrics_ranking.to_csv(os.path.join(csv_export, f"metrics_ranking.csv"), index=True)
        vprint("\nExporting metrics ranking as csv...")

    vprint('...finished!\n'
           '\nResults:\n',
           metrics_ranking)

    return metrics_ranking

# pipe4_metrics_ranking(full_predictions, metrics=metrics, csv_export=EXPORT_DIR)