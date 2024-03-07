from src.metrics import METRICS

rmse = METRICS["RMSE"]


def inv_rmse_weights(predictions):
    """
    Computes weights based on inverse RMSE

    Parameters:
        predictions (DataFrame): DataFrame containing target values ('Target') and predictions from individual forecasters.

    Returns:
        weights (dict): Dictionary containing model names as keys and respective inverse RMSE weights as items.
    """

    predictions = predictions.copy()
    # Extract target
    target = predictions.pop("Target")

    # Compute RMSE for each model
    inv_rmse_values = {}
    for model in predictions.columns:
        inv_rmse_values[model] = 1 / rmse(
            targets=target, predictions=predictions[model]
        )

    # Compute inverse RMSE per model as accuracy measure
    total_inverse_rmse = sum(inv_rmse_values.values())

    # Compute inverse RMSE weights
    weights = {
        model: inv_rmse_val / total_inverse_rmse
        for model, inv_rmse_val in inv_rmse_values.items()
    }

    # Transform weights to sum up to 1
    # sum_weights = sum(weights.values())
    # transformed_weights = {model: weight / sum_weights for model, weight in weights.items()}

    return weights  # transformed_weights
