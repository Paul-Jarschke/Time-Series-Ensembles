import numpy as np


def root_mean_squared_error(targets, predictions):
    """
    Calculate the Root Mean Squared Error (RMSE) between predicted values and targets.

    Parameters:
        targets: Array or Series containing target values.
        predictions: Array, Series or DataFrame containing predicted values.

    Returns:
        RMSE: float, RMSE value.
    """
    # Convert input arrays to numpy arrays to ensure compatibility
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Calculate the root mean squared error
    RMSE = np.sqrt(((targets - predictions) ** 2).mean())

    return RMSE
