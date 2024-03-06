import numpy as np


def mape(predictions, targets):
    """
    Calculates the Mean Absolute Percentage Error (MAPE) between predicted values and targets.

    Parameters:
        predictions: Array, Series or DataFrame containing the predicted values.
        targets: Array or Series containing target values.

    Returns:
        MAPE: float, MAPE value.
    """
    # Convert input arrays to numpy arrays to ensure compatibility
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Calculate absolute percentage error
    abs_percentage_error = np.abs((targets - predictions) / targets)

    # Replace infinite values with NaNs to handle cases where target is zero
    abs_percentage_error[np.isinf(abs_percentage_error)] = np.nan

    # Calculate mean absolute percentage error (ignoring nan values)
    MAPE = np.nanmean(abs_percentage_error) * 100

    return MAPE
