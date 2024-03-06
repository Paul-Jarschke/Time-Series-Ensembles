import numpy as np


def rmse(predictions, targets):
    """
    Calculate the Root Mean Squared Error (RMSE) between predicted values and targets.

    Parameters:
        predictions: Array, Series or DataFrame containing predicted values.
        targets: Array or Series containing target values.

    Returns:
        RMSE: float, RMSE value.
    """
    # Convert input arrays to numpy arrays to ensure compatibility
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Calculate the root mean squared error
    RMSE = np.sqrt(((targets - predictions) ** 2).mean())

    return RMSE
