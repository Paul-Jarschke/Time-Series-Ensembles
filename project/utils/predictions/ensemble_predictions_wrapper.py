from utils.predictions.get_metamodel_predictions import get_metamodel_prediction
from utils.predictions.get_weighted_predictions import get_weighted_predictions


def ensemble_prediction_wrapper(past_individual_predictions, next_indiv_predictions,
                                approach, model_function, options,
                                verbose=False,
                                *args, **kwargs):
    
    """
    Function to generate ensemble predictions based on individual predictions and a chosen approach.

    Args:
    - past_individual_predictions (DataFrame):  DataFrame containing individual predictions for past periods.
    - next_indiv_predictions (DataFrame):       DataFrame containing individual predictions for the next period.
    - approach (str):                           Approach for generating ensemble predictions. Should be one of 'meta' or 'weighted'.
    - model_function (callable):                Function to create either a meta model or a weighting scheme.
    - options (dict):                           Options to be passed to the model_function.
    - verbose (bool, optional):                 If True, additional information will be printed. Defaults to False.

    - *args: Additional positional arguments to be passed to model_function.
    - **kwargs: Additional keyword arguments to be passed to model_function.

    Returns:
    - DataFrame: Ensemble predictions for the next period.

    Notes:
    - The 'model_function' should either create a meta model with '.fit' and '.predict' methods
      that returns predictions, or a weighting scheme that returns a dictionary with model names (keys)
      and corresponding weights (values).

    - For the 'meta' approach:
      - The 'model_function' should create a meta model using 'options'.
      - The function 'get_metamodel_prediction' is used to make predictions using the meta model.

    - For the 'weighted' approach:
      - The 'model_function' should return a dictionary containing model names and their weights,
        based on the 'past_individual_predictions'.
      - The function 'get_weighted_predictions' is used to perform ensemble prediction based on weights.

    Raises:
    - ValueError: If the provided 'approach' is neither 'meta' nor 'weighted'.

    Example:
    ```
    ensemble = ensemble_prediction_wrapper(past_preds, next_preds, approach='meta',
                                           model_function=create_meta_model, options=model_options)
    ```
    """

    # Meta approach for generating ensemble predictions
    if approach == "meta":
        # Construct meta model
        metamodel = model_function(**options)
        # Make metamodel prediction using a model function
        next_ensemble_prediction = get_metamodel_prediction(
            train_data=past_individual_predictions,
            next_indiv_predictions=next_indiv_predictions, metamodel=metamodel, options=options)

    # Weighted approach (as 'ensembler' model) for generating ensemble predictions
    elif approach == "weighted":
        # Calculate weights via provided 'model', i.e., a weighting scheme
        # 'Model' must return a dictionary with items model_name and model_weight
        weights = model_function(predictions=past_individual_predictions)

        # Validation: Check if weighting scheme returns a dictionary of correct length
        expected_length_weights = len(past_individual_predictions.columns)
        expected_length_weights -= 1 if ('Target' in past_individual_predictions.columns) else 0
        assert len(weights) == expected_length_weights and isinstance(weights, dict), (
            f"Weighting scheme must return dictionary of length {expected_length_weights}")

        # Perform ensemble prediction for the next period given weights
        next_ensemble_prediction = get_weighted_predictions(next_indiv_predictions, weights=weights)
    else:
        raise ValueError("Method must be one of 'weighted' or 'meta'.")

    return next_ensemble_prediction
