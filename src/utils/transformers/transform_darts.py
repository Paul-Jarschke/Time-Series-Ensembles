import pandas as pd
from darts import TimeSeries


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
