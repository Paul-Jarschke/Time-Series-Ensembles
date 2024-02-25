from darts import TimeSeries
import pandas as pd

# Run Check
print('Loading helper functions...')


def timeseries_to_array(timeseries, pop=False):
    """
    Convert a dictionary of time series data into a flattened array.

    Parameters:
    - timeseries (dict): A dictionary containing time series data.
    - pop (bool): If True, pop the last element to match the length of the validation set.

    Returns:
    - array (list): A flattened list representing the time series data.
    """
    
    # Convert to list
    values = list(timeseries.values().flatten())
    
    # Pop last element to match length of validation/test set
    if pop:
        values.pop()
    
    return values


def prepare_eurusd_data(data, n=1000):
    """
    Prepare EURUSD data for time series analysis.

    Args:
        data (pd.DataFrame): DataFrame containing time series data.
        n (int, optional): Number of rows to select from the beginning of the DataFrame. Defaults to 1000.

    Returns:
        Tuple[darts.timeseries.TimeSeries, darts.timeseries.TimeSeries, darts.timeseries.TimeSeries]:
            A tuple containing the prepared time series data:
            - series: Time series object representing the entire dataset.
            - target_series: Time series object representing the target variable (bid_open).
            - cov_series: Time series object representing the covariate variables.

    Note:
        This function prepares the input DataFrame by selecting the first 'n' rows,
        removing the 'datetime' column, and separating the target variable (bid_open)
        and covariates for further analysis.
    """
    # Select the first n rows
    df = data.head(n)

    # Save datetime column
    dates = df['datetime']

    # Reset index to a range of integers
    df.reset_index(drop=True, inplace=True)
    df.drop('datetime', axis=1, inplace=True)

    # Extract target and covariate columns
    target_col = df['bid_open']
    covariate_cols = df[['bid_high', 'bid_low', 'bid_close', 'ask_open', 'ask_high', 'ask_low', 'ask_close']]

    # Create Darts TimeSeries objects
    series = TimeSeries.from_dataframe(df)
    target_series = TimeSeries.from_series(target_col)
    cov_series = TimeSeries.from_series(covariate_cols)

    return series, target_series, cov_series, dates



def train_test_split(series, train_split=0.3):
    """
    Split a time series into training and validation sets.

    Args:
        series (darts.timeseries.TimeSeries): Time series data to split.
        train_split (float, optional): Proportion of the series to include in the training set.
            Defaults to 0.3 (30% training, 70% validation).

    Returns:
        Tuple[darts.timeseries.TimeSeries, darts.timeseries.TimeSeries]:
            A tuple containing the training and validation sets:
            - train: Time series object representing the training set.
            - val: Time series object representing the validation set.

    Note:
        This function splits the input time series into two sets: training and validation.
        The 'train_split' parameter determines the proportion of the series allocated to training.
        The remaining portion is assigned to the validation set.
    """
    # Calculate the size of the training set based on the specified split ratio
    train_size = int(len(series) * train_split)

    # Split the series into training and validation sets based on the calculated size
    train, val = series[:train_size], series[train_size:]

    return train, val


def transform_to_darts_format(pandas_object):
    """
    Transforms a pandas DataFrame or Series object into a darts TimeSeries format.

    Parameters:
    pandas_object (DataFrame or Series): The pandas object to be transformed.

    Returns:
    TimeSeries: The transformed darts TimeSeries object.

    Note:
    If a DataFrame is passed, the index of the DataFrame is converted to timestamps
    before creating the TimeSeries object.
    """
    # Create a copy of the input pandas object to avoid changing the index globally
    pandas_object_copy = pandas_object.copy()
    
    # Convert the index of the pandas object to timestamps
    pandas_object_copy.index = pandas_object_copy.index.to_timestamp()

    # Check if the pandas object is a DataFrame
    if isinstance(pandas_object_copy, pd.DataFrame):
        # If Treu create a darts TimeSeries object
        darts_ts = TimeSeries.from_dataframe(pandas_object_copy)

    # Check if the pandas object is a Series
    elif isinstance(pandas_object_copy, pd.Series):
        # If True, create a darts TimeSeries object
        darts_ts = TimeSeries.from_series(pandas_object_copy)
        
    return darts_ts
