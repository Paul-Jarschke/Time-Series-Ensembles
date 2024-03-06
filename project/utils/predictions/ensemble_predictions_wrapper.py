from utils.predictions.get_metamodel_predictions import get_metamodel_prediction
from utils.predictions.get_weighted_predictions import get_weighted_predictions


def ensemble_prediction_wrapper(past_individual_predictions, next_indiv_predictions,
                                approach, model_function, options,
                                verbose=False,
                                *args, **kwargs):

    # scheme: either 'meta' or 'weighted'
    # model:
    # - either a metamodel with .fit and .predict method that already returns predictions
    # - or a weighting scheme that returns a dictionary with model names (keys) and corresponding weights (values)

    # Meta forecasters
    if approach == "meta":
        # Make metamodel prediction using a model function
        next_ensemble_prediction = get_metamodel_prediction(
            train_data=past_individual_predictions,
            next_indiv_predictions=next_indiv_predictions, metamodel=model_function, options=options)

    # Weighted 'forecasters'
    elif approach == "weighted":
        # Calculate weights via provided 'model', i.e., a weighting scheme
        # 'Model' must return a dictionary with items model_name and model_weight
        weights = model_function(predictions=past_individual_predictions)

        # For validation check if weighting scheme returns a dictionary of correct length
        expected_length_weights = len(past_individual_predictions.columns)
        expected_length_weights -= 1 if ('Target' in past_individual_predictions.columns) else 0
        assert len(weights) == expected_length_weights and isinstance(weights, dict), (
            f"Weighting scheme must return dictionary of length {expected_length_weights}")

        # Perform ensemble prediction for the next period given weights
        next_ensemble_prediction = get_weighted_predictions(next_indiv_predictions, weights=weights)
    else:
        raise ValueError("Method must be one of 'weighted' or 'meta'.")

    return next_ensemble_prediction
