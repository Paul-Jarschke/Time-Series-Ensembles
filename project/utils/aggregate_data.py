import pandas as pd

print('Loading aggregation function...')


def aggregate_data(data, agg_freq: str, date_col='index', columns='all', method='mean', drop_nan=True):
    """
    Aggregate data to specified frequency by specified method.

    Parameters:
    - data (pandas DataFrame or Series): DataFrame containing data to aggregate.
    - date_col (str, optional): Name of the date column. Needs to be specified only if date is not yet in the index.
    - columns (str or list of str, optional): Name of the aggregated columns to be returned by the function.\
    Takes all columns by default.
    - agg_freq (str, optional): Desired aggregation frequency. (E.g., 'B', 'D', 'M', 'Y', ...)
    - method (str, optional): Pandas aggregation method (E.g., 'last', 'mean', 'min', 'max'). Defaults to 'mean'.
    - drop_nan (bool, optional): Whether to drop NaN values after aggregation. Defaults to True.

    Returns:
    - aggregated_df (DataFrame): Aggregated DataFrame.
    """

    # Input validation
    assert isinstance(data, (pd.DataFrame, pd.Series)), 'Data must be a pandas DataFrame or Series.'
    assert isinstance(agg_freq, str), 'Aggregation frequency must be provided as str.'
    assert method in ['last', 'mean', 'min', 'max'], 'Aggregation method must be one of ["last", "mean", "min", "max"].'

    # Ensure that index contains dates
    if date_col != 'index':
        data.set_index(date_col, inplace=True)

    # Check if pandas object has valid time index (DatetimeIndex, PeriodIndex, TimedeltaIndex), if not transfer index
    if not isinstance(data.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        data.index = pd.DatetimeIndex(data.index, freq='infer')

    # Take subset
    if columns != 'all':
        data = data[columns]

    # Aggregate based on method
    aggregated_df = data.resample(agg_freq).apply(method)

    # Optionally drop rows with NaN values
    if drop_nan:
        aggregated_df.dropna(inplace=True)

    return aggregated_df

#%%
# For debugging:
# import os
# from paths import *
# FILE = os.path.join(DATA_DIR, 'testing', 'eurusd_df.csv')
# df = pd.read_csv(FILE, index_col='datetime')
#
# print(df)
# print(df.index)
# agg_df = aggregate_data(data=df, columns='bid_close', agg_freq='B', method='last', drop_nan=True)
# print(agg_df)
# print(agg_df.head(15))