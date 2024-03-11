import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.colors as colors
import os

from src.utils.paths import *


def plot_metrics(metric, data_dir=PAPERDATA_DIR, remove_models=None, sort_labels=None, export=True):
    """
    Plot comparison of a specified metric across different datasets for various models.

    Parameters:
        metric (str):                           The metric to be plotted.
        data_dir (str, os.PathLike):            The directory where currently analyzed files are stored.
        remove_models (list, optional):         Models to be removed from plot.
        sort_labels (list, optional):           Provide the order for the labels on the x-axis.
        export (bool, optional):                Export plots to given PLOT_DIR.

    Raises:
        ValueError: If an invalid metric is provided.

    Example:
        # Example usage
        labels = ['Data1', 'Data2', 'Data3']
        plot_metrics('sMAPE', data_dir=PAPERDATA_DIR, labels)
    """

    # Empty DataFrame to store data for each model
    models_data = pd.DataFrame()

    # Column for best model names
    best_model_names = []

    # List for benchmarks
    benchmarks = ["Naive", "Best Forecaster"]

    # Setting up empty lists for None inputs
    if remove_models is None:
        remove_models = []

    # List all file paths in currently analyzed data
    current_data_paths = os.listdir(data_dir)
    relevant_files = [file for file in current_data_paths if file.endswith('_metrics_ranking.csv')]

    # Sort files:
    if sort_labels:
        relevant_files = [file + "_metrics_ranking.csv" for file in sort_labels]

    for file in relevant_files:
        # Read in file as dataframe
        df = pd.read_csv(os.path.join(data_dir, file), index_col='Model')
        current_label = file.replace("_metrics_ranking.csv", "")

        # Throw error if metric is not in input data and thus not supported
        if metric not in df.columns:
            valid_metrics = df.columns[~df.columns.contains('Ranking')]
            raise ValueError(f"Invalid metric '{metric}'. Please choose from: {', '.join(valid_metrics)}")
        # Only select that column
        else:
            df = df[metric]

        # Identify the best individual forecaster for current DataFrame
        forecasters_names = df.index[[not model.startswith(("Weighted", "Meta")) for model in df.index]]
        df_forecasters_sorted = df.loc[forecasters_names].sort_values(ascending=True)
        df_best_forecaster_name = df_forecasters_sorted.index[0]
        df_best_forecaster_value = df_forecasters_sorted[0]
        best_model_names.append(df_best_forecaster_name)

        # # Indicator that there are forecasters better than Naive
        # better_then_naive = True if best_forecaster != "Naive" else False

        # benchmarks += [best_forecaster] if better_then_naive else []

        # Remove

        # Filter for ensemble and naive (not drift!) models only
        filtered_df = df.loc[(df.index.str.contains('Ensemble') | (df.index.isin(benchmarks))) &
                             ~df.index.isin(remove_models)].to_frame()

        # Add Best Forecaster column to filtered df
        filtered_df.loc["Best Forecaster"] = df_best_forecaster_value

        # Transpose and change index to filename
        filtered_df = filtered_df.T
        filtered_df.index = [current_label]

        models_data = pd.concat([models_data, filtered_df], axis=0)

        print("Dataset:", file,
              "\nBest Model:", df_best_forecaster_name,
              f"\n{metric} Value:",
              round(df_best_forecaster_value, 3), "\n"
              )

    # Customize the x-axis tick marks based on whether custom data labels are provided
    # if data_labels:
    #     # Use custom data labels
    #     try:
    #         models_data.index = data_labels
    #     except:
    #         raise ValueError('Length of provided data labels does not match number of given datasets.')
    # else:
    #     # Default to 1, 2, 3
    #     models_data.index = [f'{i + 1}' for i in range(len(models_data.index))]

    # Redundant:
    # if sort_labels:
    #     models_data = models_data.loc[sort_labels]

    # Sort data such that first comes naive, then weighted, then meta
    model_names = models_data.columns
    models_data = models_data[
        sorted(model_names, key=lambda x: (x.startswith('Meta'), x.startswith('Weighted'),
                                           x.startswith("Best Forecaster"), x.startswith('Naive'))
               )
    ]

    # Extract names of weighted and metamodels
    weighted_model_names = [model for model in model_names if model.startswith('Weighted')]
    meta_model_names = [model for model in model_names if model.startswith('Meta')]

    # Get number of weighted and metamodels
    n_weighted = len(weighted_model_names)
    n_meta = len(meta_model_names)
    n_benchmarks = len(benchmarks)

    # Set colormaps
    def truncate_colormap(cmap, n, minval=0.3, maxval=1.6):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    # Plot data
    cmap = plt.get_cmap("Blues_r")
    cmap = truncate_colormap(cmap, n_benchmarks, 0.4, 0.6)
    models_data[benchmarks].copy().plot(
        marker='D', linestyle='-', colormap=cmap)  #, linewidth=3)
    if n_weighted > 0:
        cmap = plt.get_cmap("YlGn")
        if n_weighted > 1:
            cmap = truncate_colormap(cmap, n_weighted, 0.3, 0.8)
        models_data[weighted_model_names].copy().plot(
            marker='s', linestyle='-', ax=plt.gca(), colormap=cmap, alpha=0.95)
    if n_meta > 0:
        cmap = plt.get_cmap("RdPu")
        if n_weighted > 1:
            cmap = truncate_colormap(cmap, n_meta, 0.3, 0.8)
        models_data[meta_model_names].copy().plot(
            marker='o', linestyle='-', ax=plt.gca(), colormap=cmap, alpha=0.95)

    # Labeling
    plt.xlabel('Complexity Level', fontsize=12, labelpad=10)
    plt.ylabel(metric, fontsize=12)
    # plt.yscale('log')
    plt.title(f'Model Performance vs. Data Complexity', fontsize=18)

    # Add footnote for best individual forecasters
    note = "Best individual forecasters:"
    for i, model_name in enumerate(best_model_names):
        label = sort_labels[i]
        if i == len(best_model_names) - 1:
            note += f" and {model_name} ({label})."
        elif i == 2: # new line
            note += f" {model_name} ({label}),\n"
        else:
            note += f" {model_name} ({label}),"
    plt.annotate(text=note,
                 xy=(0.5, -0.2),
                 xycoords='axes fraction',
                 ha='center',
                 va="center",
                 fontsize=10)

    # Cropping and other formatting
    plt.legend()
    plt.grid(True)
    plt.gcf().set_size_inches(8.57,6)
    plt.subplots_adjust(right=0.7)  # # makes space for legend
    #plt.tight_layout(rect=(0, 0, 0.6, 1))  # makes space for legend

    # Set up pretty legend
    lines, labels = plt.gca().get_legend_handles_labels()

    # Shorten ensembles names
    labels = [label.split(": ")[1] for label in labels[n_benchmarks:]]
    labels = benchmarks + labels

    # Set up LegendTitle class for handling legend titles
    class LegendTitle(object):
        def __init__(self, text_props=None):
            self.text_props = text_props or {}
            super(LegendTitle, self).__init__()

        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            title = mtext.Text(x0, y0, orig_handle, **self.text_props)
            handlebox.add_artist(title)
            return title

    # Naive:
    legend_lines = [
        "Benchmark",
        lines[0],
        lines[1]
    ]
    legend_labels = [
        '',
        labels[0],
        labels[1]
    ]

    # Weighted:
    if n_weighted > 0:
        legend_lines += ['', 'Ensemble: Weighted'] \
                         + lines[n_benchmarks : n_benchmarks + n_weighted]
        legend_labels += ['', ''] \
                        + labels[n_benchmarks:n_benchmarks + n_weighted]

    # Meta
    if n_meta > 0:
        legend_lines += ['', 'Ensemble: Meta'] \
                         + lines[-n_meta:]
        legend_labels += ['', ''] \
                         + labels[-n_meta:]

    plt.legend(
        # Labels:
        legend_lines,
        # Lines:
        legend_labels,
        # Settings:
        handler_map={str: LegendTitle({'fontsize': 12})},
        bbox_to_anchor=(1.03, 1),
        loc='upper left',
        fontsize=10
    )
    # Export
    if export:
        plt.savefig(os.path.join(PLOT_DIR, "metrics_ranking.png"))
        print("Export successful!")

    plt.show()
