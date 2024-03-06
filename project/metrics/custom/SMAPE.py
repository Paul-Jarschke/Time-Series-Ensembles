import numpy as np


def symmetric_mean_absolute_percentage_error(targets, predictions):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE) between predicted values and targets.

    Parameters:
        targets: Array or Series containing target values.
        predictions: Array, Series, or DataFrame containing the predicted values.

    Returns:
        SMAPE: float, SMAPE value.
    """
    # Convert input arrays to numpy arrays to ensure compatibility
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Calculate Symmetric mean absolute percentage error
    n = len(targets)
    SMAPE = (100 / n) * np.sum(np.abs(targets - predictions) / (np.abs(targets) + np.abs(predictions) / 2))

    return SMAPE
