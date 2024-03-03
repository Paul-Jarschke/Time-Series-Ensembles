from darts import TimeSeries
import pandas as pd
import os

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
    darts_object: The transformed darts TimeSeries object.

    Note:
    The index of the DataFrame is converted to DatetimeIndex before creating the TimeSeries object.
    """
    # Create a copy of the input pandas object to avoid changing the index globally
    pandas_object_copy = pandas_object.copy()
    
    # Ensure that index of the pandas object is DatetimeIndex
    if not isinstance(pandas_object_copy.index, pd.DatetimeIndex):
        pandas_object_copy.index = pandas_object_copy.index.to_timestamp()

    # Check if the pandas object is a DataFrame
    if isinstance(pandas_object_copy, pd.DataFrame):
        # If True create a darts TimeSeries object
        darts_object = TimeSeries.from_dataframe(pandas_object_copy)

    # Check if the pandas object is a Series
    elif isinstance(pandas_object_copy, pd.Series):
        # If True, create a darts TimeSeries object
        darts_object = TimeSeries.from_series(pandas_object_copy)
    else:
        raise ValueError("Input must be pandas DataFrame or Series object.")
    return darts_object


def identify_date_column(df, date_format):
    # identifies where the date is stored (index or column), transforms it to timestamp,
    # infers frequency, set it as pandas index
    # input is pandas df
    for column_name, column_content in df.items():
        # Search in columns

        if column_content.dtype == 'object':
            try:
                pd.to_datetime(column_content, format=date_format)
                identified_location = column_name
                return identified_location
            except ValueError:
                pass
        # Look in index
        try:
            pd.to_datetime(df.index, format=date_format)
            identified_location = 'index'
            return identified_location
        # Raise error if date could neither be found in index nor in columns
        except ValueError:
            raise ValueError(
                'Date information can not be inferred. ',
                'Please specify name or position using \'date_col\' argument.'
            )


def target_covariate_split(df, target='infer', covariates='infer', exclude=None):
    """
    Separates a pandas DataFrame into target and covariates.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target (int or str, optional): The column index or name of the target variable. Infers first column as target
        by default.
        covariates (None, int, str, or list of int and/or str, optional): The column indices or names of the covariates.
        Infers all existing columns after the first column as covariates by default.
        exclude (int, str, or list of int and/or str, optional): Indices or names of the columns to be excluded.

    Returns:
        tuple: A tuple containing the target variable DataFrame and the covariates DataFrame.
    """
    # Exclude columns if specified
    if exclude is not None:

        # if not isinstance(exclude, list):
        #     exclude = [exclude]
        # elif isinstance(exclude, list):
        #     exclude = [var if isinstance(var, str) else df.index[var] for var in exclude]
        # else:
        #     raise ValueError('Excluded columns must be provided either as int, str, or list of int/str.')
        # it is already ensured in pipeline that exclude is of correct for dropping (str or list of str)
        df = df.drop(exclude, axis=1)
    
    # Infer covariates if option is chosen
    if target == 'infer':
        target = df.columns[0]
    # elif isinstance(target, int):
    #     target = df.iloc[:, target].copy()
    # it is already ensured in pipeline that target is of correct type (str)
    target = df[target].copy()
    
    if covariates == 'infer':
        n_covariates = len(df.columns) - 1  # reduce by 1 because of target
        # Infer covariates if option is chosen
        if n_covariates > 0:
            covariates = list(range(1, n_covariates + 1))
            covariates = df.columns[covariates]
        else:
            covariates = None
        # Select covariates based on input
        # if isinstance(covariates, int):
        #    covariates = df.iloc[:, covariates].copy()
    # it is already ensured in pipeline that covariates is of correct type (str or list of str)
    if covariates is not None:
        covariates = df[covariates].copy()
        # elif isinstance(covariates, list):
            # covariates = [cov if isinstance(cov, int) else df.columns.get_loc(cov) for cov in covariates]
        # else:
        #     raise ValueError("Provided covariates must be provided either as int, str, or list of int/str.")

    return target, covariates


def csv_reader(PATH, file_name, date_col=0, columns='all', *args, **kwargs):
    """
        Read a CSV file from the specified directory path and return it as a pandas DataFrame.

        Parameters:
        - PATH (str): The directory path where the CSV file is located.
        - file_name (str): The name of the CSV file.
        - date_col (int, optional): The column index to be used as the index for the DataFrame. Default is first column.
        - columns (list of int/str, optional): Subset of columns to select, denoted either \
        by column labels or column indices stored in a list-like object.
        - *args: Additional positional arguments to be passed to pandas.read_csv().
        - **kwargs: Additional keyword arguments to be passed to pandas.read_csv().

        Returns:
        - df (pandas DataFrame): The DataFrame containing the data from the CSV file.
        """

    # Ensure that file has csv ending
    if not file_name.endswith('.csv'):
        file_name += '.csv'

    # Combine the directory path and file name
    FILE = os.path.join(PATH, file_name)

    # Read data, set time index end select columns
    columns = None if columns == 'all' else columns
    df = pd.read_csv(FILE, index_col=date_col, usecols=columns, *args, **kwargs)

    return df
