import pandas as pd
from darts.timeseries import TimeSeries


def transform_to_darts(*args):
    """
    Transforms a pandas DataFrame or Series object into a darts TimeSeries format.

    Parameters:
    pandas_object (DataFrame or Series): The pandas object to be transformed.

    Returns:
    darts_object: The transformed darts TimeSeries object.

    Note:
    The index of the DataFrame is converted to DatetimeIndex before creating the TimeSeries object.
    """
    return_tuple = ()
    for pandas_object in args:
        if pandas_object is None:
            darts_object = pandas_object
            pass
    # Check if the pandas object is a pandas object
        elif isinstance(pandas_object, (pd.Series, pd.DataFrame)):
            # Create a copy of the input pandas object to avoid changing the index globally
            darts_object = pandas_object.copy()

            # Ensure that index of the pandas object is DatetimeIndex
            if not isinstance(darts_object.index, pd.DatetimeIndex):
                darts_object.index = darts_object.index.to_timestamp()

            # Transform to TimeSeries object
            # If pandas object is a DataFrame
            if isinstance(darts_object, pd.DataFrame):
                # If True create a darts TimeSeries object
                darts_object = TimeSeries.from_dataframe(darts_object)
            # If pandas object is a Series
            elif isinstance(darts_object, pd.Series):
                # If True, create a darts TimeSeries object
                darts_object = TimeSeries.from_series(darts_object)
        else:
            raise ValueError("Input must be pandas DataFrame or Series object.")
        return_tuple = return_tuple + (darts_object,)
    return return_tuple


def transform_to_sktime(*args, freq='infer'):
    """
    Transforms a pandas DataFrame or Series object into a sktime compatible format (PeriodIndex).

    Parameters:
    pandas_object (DataFrame or Series): The pandas object to be transformed.

    Returns:
    sktime_object: The transformed darts TimeSeries object.

    Note:
    The index of the DataFrame is converted to DatetimeIndex before creating the TimeSeries object.
    """
    # Check if the object is a pandas object
    return_tuple = ()
    for pandas_object in args:
        if pandas_object is None:
            sktime_object = pandas_object
            pass
        elif isinstance(pandas_object, (pd.Series, pd.DataFrame)):
            # Create a copy of the input pandas object to avoid changing the index globally
            sktime_object = pandas_object.copy()
            # Ensure that object has DatetimeIndex
            if not isinstance(sktime_object.index, pd.DatetimeIndex):
                sktime_object.index = sktime_object.index.to_timestamp()
            # Infer frequency if not specified
            if freq == 'infer':
                freq = pd.infer_freq(sktime_object.index)

            # Transform to PeriodIndex
            freq = 'M' if freq == 'MS' else freq # Somehow PeriodIndex only takes 'M' and not 'MS'
            sktime_object.index = pd.PeriodIndex(sktime_object.index, freq=freq)
        else:
            raise ValueError("Input must be pandas DataFrame or Series object.")
        return_tuple = return_tuple + (sktime_object,)
    if len(return_tuple) == 1:
        return_tuple = return_tuple[0]
    return return_tuple


def transform_sktime_lagged_covariates(*args):
    # When using covariate model in sktime, lag covariates by one period
    # Known values for forecast in period t+k+1 are all values up to t+k
    # only relevant for covariate models in sktime (since there is no past_covariates argument and fitting
    # needs all covariates up to t+k+1)
    # ensure that data has PeriodIndex, if not transform first
    return_tuple = ()
    for i, pandas_object in enumerate(args):
        if not isinstance(pandas_object.index, pd.PeriodIndex):
            sktime_object = transform_to_sktime(pandas_object)
        else:
            sktime_object = pandas_object

        # target
        if i == 0:
            sktime_object = sktime_object.iloc[1:].copy()  # drop first observation
        # covariates
        elif sktime_object is not None:
            sktime_object = sktime_object.shift(1).dropna().copy()
        return_tuple = return_tuple + (sktime_object, )
    return return_tuple


TRANSFORMERS = {
    'sktime': transform_to_sktime,
    'darts': transform_to_darts,
    'sktime.lagged': transform_sktime_lagged_covariates
}