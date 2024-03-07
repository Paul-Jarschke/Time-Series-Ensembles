import numpy as np


def inv_variance_weights(predictions):
    """
    Computes weights based on the inverse of prediction variances.

    Parameters:
        predictions (DataFrame): DataFrame containing target values ('Target') and predictions from individual forecasters.

    Returns:
        weights (dict): Dictionary containing model names as keys and respective inverse variance weights as items.
    """
    # Remove target
    predictions = predictions.drop(columns=["Target"])

    # Compute prediction variance for each model
    prediction_variances = {}
    for model in predictions.columns:
        prediction_variances[model] = np.var(predictions[model])

    # Compute the sum of the inverse prediction variances
    sum_inverse_variances = sum(
        1 / variance for variance in prediction_variances.values()
    )

    # Compute weights based on the inverse of prediction variances
    weights = {
        model: (1 / variance) / sum_inverse_variances
        for model, variance in prediction_variances.items()
    }

    return weights
