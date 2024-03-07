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
