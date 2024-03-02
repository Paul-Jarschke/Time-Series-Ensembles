import numpy as np

print('Loading metrics...')


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


def smape(predictions, targets):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE) between predicted values and targets.

    Parameters:
        predictions: Array, Series, or DataFrame containing the predicted values.
        targets: Array or Series containing target values.

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


metrics = {
    'MAPE': mape,
    'RMSE': rmse,
    'SMAPE': smape
}