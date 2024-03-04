import numpy as np
import pandas as pd
from utils.metrics import rmse
# from utils.helpers import vprint

def equal_weights(predictions):
    """
    Compute equal weights.

    Parameters:
        predictions (DataFrame): DataFrame containing target values ('Target') and predictions from individual models.

    Returns:
        weights (dict): Dictionary containing model names as keys and equal weights as items.
    """
    # Remove target
    predictions = predictions.drop(columns=['Target'])

    # Get number of models
    n_models = len(predictions.columns)

    # Compute equal weights
    weights = {model: 1 / n_models for model in predictions.keys()}

    return weights


def inv_rmse_weights(predictions):
    """
    Computes weights based on inverse RMSE

    Parameters:
        predictions (DataFrame): DataFrame containing target values ('Target') and predictions from individual models.

    Returns:
        weights (dict): Dictionary containing model names as keys and respective inverse RMSE weights as items.
    """

    predictions = predictions.copy()
    # Extract target
    target = predictions.pop('Target')

    # Compute RMSE for each model
    inv_rmse_values = {}
    for model in predictions.columns:
        inv_rmse_values[model] = 1 / rmse(predictions[model], target)

    # Compute inverse RMSE per model as accuracy measure
    total_inverse_rmse = sum(inv_rmse_values.values())

    # Compute inverse RMSE weights
    weights = {model: inv_rmse_val / total_inverse_rmse for model, inv_rmse_val in inv_rmse_values.items()}

    # Transform weights to sum up to 1
    # sum_weights = sum(weights.values())
    # transformed_weights = {model: weight / sum_weights for model, weight in weights.items()}

    return weights  # transformed_weights


def inv_variance_weights(predictions):
    """
    Computes weights based on the inverse of prediction variances.

    Parameters:
        predictions (DataFrame): DataFrame containing target values ('Target') and predictions from individual models.

    Returns:
        weights (dict): Dictionary containing model names as keys and respective inverse variance weights as items.
    """
    # Remove target
    predictions = predictions.drop(columns=['Target'])

    # Compute prediction variance for each model
    prediction_variances = {}
    for model in predictions.columns:
        prediction_variances[model] = np.var(predictions[model])

    # Compute the sum of the inverse prediction variances
    sum_inverse_variances = sum(1 / variance for variance in prediction_variances.values())

    # Compute weights based on the inverse of prediction variances
    weights = {model: (1 / variance) / sum_inverse_variances for model, variance in prediction_variances.items()}

    return weights


def inv_error_cov_weights(predictions, verbose=True):
    """
    Computes weights based on the inverse of prediction error covariances.

    This function computes the error correlation weights for a set of models based on the error matrix C,
    where C_ij represents the correlation between the errors of model i and model j.

    Parameters:
        predictions (DataFrame): DataFrame containing target values ('Target') and predictions from individual models.
        verbose (bool, optional): Whether to print information about the computed weights. Defaults to True.

    Returns:
       weights (dict): Dictionary containing model names as keys and respective error correlation weights as items.
    """
    predictions = predictions.copy()
    # Extract target
    target = predictions.pop('Target')

    # Compute errors for each model
    errors = predictions.apply(lambda x: target - x)

    # Compute the number of models
    n_models = len(errors.columns)

    # Initialize the error correlation matrix C with zeros
    # C = np.zeros((n_models, n_models))

    # Compute the element-wise product of errors between each pair of models
    # for i, model_i in enumerate(errors.columns):
    #     for j, model_j in enumerate(errors.columns):
    #         C[i, j] = np.sum(errors[model_i] * errors[model_j])
    # Slightly different because formula above assumes E[error] = 0
    # => not the real covariance

    # Divide by n to obtain the error covariance matrix
    # C /= len(errors)
    # Calculate Error Covariance Matrix
    C = errors.cov()
    # Compute the inverse matrix
    C_inv = np.linalg.inv(C)

    # Sum the elements in each row of the inverse error correlation matrix to get model weights
    row_sums = np.sum(C_inv, axis=0)

    # Calculate weights for each model based on the proportion of its summed up inverse error correlations
    # to the total sum of error correlations across all models.
    total_sum = np.sum(C_inv)
    model_names = predictions.columns  # Extract model names

    # Compute model weights
    weights = {model: row_sum / total_sum for model, row_sum in zip(model_names, row_sums)}

    return weights


def compute_weighted_predictions(next_individual_predictions, weights, verbose=False):
    """
    Compute weighted ensemble predictions using provided weights

    Parameters:
        next_individual_predictions (DataFrame): DataFrame containing predictions from individual models \
        that should be ensembled.
        weights (dict): Dictionary containing model names as keys and their respective weights as items.

    Returns:
        numpy.ndarray: Ensemble predictions given weights
    """

    # Print information about the computed weights if verbose is True
    # vsum_weights = sum(weights.values())
    # vmin_weight = min(weights.values())
    # vmax_weight = max(weights.values())

    # vprint(f'Checking weights...')
    # vprint(f'Sum of weights: {sum_weights}')
    # vprint(f'Range of weights: [{min_weight}, {max_weight}]\n')
    # vprint(f'Weights:')
    # vprint(weights)

    # Exclude target
    if 'Target' in next_individual_predictions.columns:
        next_individual_predictions = next_individual_predictions.drop(columns=['Target'])

    # Compute row-wise weighted average predictions
    weighted_predictions = sum(next_individual_predictions[model] * weights[model] for model in
                               next_individual_predictions.columns)

    return weighted_predictions


def compute_metamodel_prediction(train_data, next_indiv_predictions, metamodel, *args, **kwargs):
    # Prepare features (predictions of forecasting models) and target variable (actual values) for training
    train_data = train_data.copy()
    target = train_data.pop('Target')
    predictions = train_data

    # Set up Meta-Model with arguments
    model = metamodel(*args, **kwargs) # models.ensembling

    # Train Meta-Model
    model.fit(predictions, target)

    # Prepare features for prediction
    next_indiv_predictions = next_indiv_predictions.drop(columns=['Target'])

    # Make ensemble predictions for next given period
    ensemble_prediction = model.predict(next_indiv_predictions)

    if not isinstance(ensemble_prediction, pd.Series):
        ensemble_prediction = pd.Series(data=ensemble_prediction, index=next_indiv_predictions.index)

    return ensemble_prediction


def get_ensemble_prediction(past_individual_predictions, next_indiv_predictions, method, model=None, verbose=False,
                            *args, **kwargs):
    # scheme is either a metamodel with fit and predict method or a weighting scheme that returns a dictionary with
    # model names (keys) and corresponding weights (values)

    # Meta models
    if method == "meta":
        # Check if metamodel is provided
        if model is not None:
            # Check if provided metamodel is indeed a model with fit and predict methods
            if (hasattr(model, 'fit') and callable(getattr(model, 'fit')) and
                    hasattr(model, 'predict') and callable(getattr(model, 'predict'))):
                next_ensemble_prediction = compute_metamodel_prediction(
                    train_data=past_individual_predictions,
                    next_indiv_predictions=next_indiv_predictions, metamodel=model, *args, **kwargs)
            else:
                raise ValueError('Meta model must have \'fit\' and \'predict\' methods.')
        else:
            raise ValueError("meta_model must be defined")

    # Weighted 'models'
    elif method == "weighted":
        # Calculate weights
        weights = model(predictions=past_individual_predictions, next_prdictions=next_indiv_predictions)

        expected_length_weights = len(past_individual_predictions.columns)
        expected_length_weights -= 1 if ('Target' in past_individual_predictions.columns) else 0
        if len(weights) != expected_length_weights:
            raise ValueError(f"Weighting scheme must return dictionary of length {expected_length_weights}")
        else:
            next_ensemble_prediction = compute_weighted_predictions(next_indiv_predictions, weights, verbose=verbose)
    else:
        raise ValueError("Method must be one of 'weighted' or 'meta'.")

    return next_ensemble_prediction
