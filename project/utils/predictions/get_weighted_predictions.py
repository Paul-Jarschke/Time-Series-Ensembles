def get_weighted_predictions(next_individual_predictions, weights):
    """
    Compute weighted ensemble predictions using provided weights

    Parameters:
        next_individual_predictions (DataFrame):    DataFrame containing predictions from individual forecasters
                                                    that should be ensembled.
        weights (dict):                             Dictionary containing model names as keys and their respective weights as items.

    Returns:
        numpy.ndarray: Ensemble predictions given weights
    """

    # Exclude target column if present
    if 'Target' in next_individual_predictions.columns:
        next_individual_predictions = next_individual_predictions.drop(columns=['Target'])

    # Compute row-wise weighted average predictions
    weighted_predictions = sum(next_individual_predictions[model] * weights[model] for model in
                               next_individual_predictions.columns)

    return weighted_predictions
