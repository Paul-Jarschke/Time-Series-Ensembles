def equal_weights(predictions):
    """
    Compute equal weights.

    Parameters:
        predictions (DataFrame): DataFrame containing target values ('Target') and predictions from individual forecasters.

    Returns:
        weights (dict): Dictionary containing model names as keys and equal weights as items.
    """
    # Remove target
    predictions = predictions.drop(columns=['Target'])

    # Get number of forecasters
    n_models = len(predictions.columns)

    # Compute equal weights
    weights = {model: 1 / n_models for model in predictions.keys()}

    return weights
