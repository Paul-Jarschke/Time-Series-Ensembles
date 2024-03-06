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
