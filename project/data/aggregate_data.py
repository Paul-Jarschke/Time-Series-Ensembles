import pandas as pd

print('Loading aggregation function...')

def aggregate_data_daily(path, date=None, value_column='bid_open', method='mean', drop_nan=True):
    """
    Aggregate data by day.

    Parameters:
    - path (str): Path to the CSV file.
    - date (str): Name of the column containing dates. If None, the function will infer the column.
    - value_column (str): Name of the column containing the values to aggregate.
    - method (str): Aggregation method ('mean', 'min', 'max').
    - drop_nan (bool): Whether to drop NaN values after aggregation.

    Returns:
    - aggregated_df (DataFrame): Aggregated DataFrame.
    """

    # Load csv data and transform to pandas DataFrame
    df = pd.read_csv(path)

    # Infer the date column if not specified
    if date is None:
        date = df.columns[0]  # Assuming the first column is the date column

    # Ensure that date column is datetime object
    df[date] = pd.to_datetime(df[date])

    # Aggregate based on method
    aggregated_df = df.groupby(df[date].dt.date)[value_column].agg(method).reset_index()

    # Optionally drop rows with NaN values
    if drop_nan:
        aggregated_df.dropna(subset=[value_column], inplace=True)

    return aggregated_df

