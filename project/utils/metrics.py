import numpy as np

print('Loading metrics...')

def rmse(predictions, targets):
    """
    Calculate the Root Mean Squared Error (RMSE) between predicted values and targets.

    Parameters:
        predictions (array-like): Array containing predicted values.
        targets (array-like): Array containing target values.

    Returns:
        float: RMSE value.
    """
    # Convert input arrays to numpy arrays to ensure compatibility
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Calculate the root mean squared error
    rmse = np.sqrt(((predictions - targets) ** 2).mean())

    return rmse


def mape(predictions, targets):
    """
    Calculates the Mean Absolute Percentage Error (MAPE) between predicted values and targets.

    Parameters:
        predictions (array-like): Array containing the predicted values.
        targets (array-like): Array containing target values.

    Returns:
        float: MAPE value.
    """
    # Convert input arrays to numpy arrays to ensure compatibility
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate absolute percentage error
    abs_percentage_error = np.abs((targets - predictions) / targets)
    
    # Replace infinite values with NaNs to handle cases where target is zero
    abs_percentage_error[np.isinf(abs_percentage_error)] = np.nan
    
    # Calculate mean absolute percentage error (ignoring nan values)
    mape = np.nanmean(abs_percentage_error) * 100
    
    return mape