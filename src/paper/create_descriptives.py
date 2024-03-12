import pandas as pd
import os
from src.utils.paths import DATA_DIR


def create_descriptives(file_names, filter_variables, directory=DATA_DIR):
    descriptives = pd.DataFrame()
    dataset_csv = []
    for i, data in enumerate(file_names):
        if not data.endswith(".csv"):
            dataset_csv.append(data + ".csv")

    for dirpath, dirnames, files in os.walk(directory):
        for file in files:
            if file in dataset_csv:
                df = pd.read_csv(str(os.path.join(dirpath, file)), index_col=0)
                df = df.loc[:, [col in filter_variables for col in df.columns]]
                df.index = pd.DatetimeIndex(df.index)
                inferred_freq = pd.infer_freq(df.index)
                df.index.freq = inferred_freq
                df.index.to_period()
                if inferred_freq in ['M', 'MS']:
                    fmt = "%Y-%m"
                elif inferred_freq in ['D', 'B']:
                    fmt = "%Y-%m-%d"
                else:
                    fmt = None
                start = df.index.strftime(fmt)[0]
                end = df.index.strftime(fmt)[-1]
                start_end_df = pd.DataFrame({"start": [start] * len(df.columns), "end": [end] * len(df.columns)},
                                            index=df.columns)
                missing = df.isna().sum(axis=0)
                missing = missing.rename("NaN")
                inferred_freq_df = pd.DataFrame({"freq": [inferred_freq] * len(df.columns)},
                                            index=df.columns)
                transposed_summary = df.describe().T
                transposed_summary = pd.concat([start_end_df, transposed_summary, missing, inferred_freq_df], axis=1)

                #transposed_summary = pd.concat([transposed_summary, missing_count, freq], axis=1)

                descriptives = pd.concat([descriptives, transposed_summary], axis=0)

                descriptives.index.rename("Data", inplace=True)

    # Sort
    descriptives = descriptives.reindex(filter_variables)

    return descriptives