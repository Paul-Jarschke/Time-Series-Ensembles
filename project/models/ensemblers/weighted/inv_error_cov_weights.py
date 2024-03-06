import numpy as np


def inv_error_cov_weights(predictions, verbose=True):
    """
    Computes weights based on the inverse of prediction error covariances.

    This function computes the error correlation weights for a set of forecasters based on the error matrix C,
    where C_ij represents the correlation between the errors of model i and model j.

    Parameters:
        predictions (DataFrame): DataFrame containing target values ('Target') and predictions from individual forecasters.
        verbose (bool, optional): Whether to print information about the computed weights. Defaults to True.

    Returns:
       weights (dict): Dictionary containing model names as keys and respective error correlation weights as items.
    """
    predictions = predictions.copy()
    # Extract target
    target = predictions.pop('Target')

    # Compute errors for each model
    errors = predictions.apply(lambda x: target - x)

    # Compute the number of forecasters
    n_models = len(errors.columns)

    # Calculate Error Covariance Matrix
    C = errors.cov()
    # Compute the inverse matrix
    C_inv = np.linalg.inv(C)

    # Sum the elements in each row of the inverse error correlation matrix to get model weights
    row_sums = np.sum(C_inv, axis=0)

    # Calculate weights for each model based on the proportion of its summed up inverse error correlations
    # to the total sum of error correlations across all forecasters.
    total_sum = np.sum(C_inv)
    model_names = predictions.columns  # Extract model names

    # Compute model weights
    weights = {model: row_sum / total_sum for model, row_sum in zip(model_names, row_sums)}

    return weights
