import numpy as np
from metrics import rmse


def simple_average(data):
    """
    Averages predictions of all models.

    Parameters:
        data (DataFrame): DataFrame containing actual values and predictions from different models.

    Returns:
        numpy.ndarray: Averaged predictions.
    """
    # Select columns containing predictions (excluding the first two columns)
    prediction_columns = data.columns[2:]

    # Average predictions across all models
    averaged_predictions = np.mean(data[prediction_columns], axis=1)

    return averaged_predictions


def compute_rmse_weights(data):
    """
    Computes predictions using the weighted average ensemble method with weights based on RMSE shares.

    Parameters:
        data (DataFrame): DataFrame containing actual values and predictions from different models.

    Returns:
        numpy.ndarray: Weighted average predictions.
    """
    
    # Extract target 
    target = data['Actual']
    
    # Select columns containing predictions (excluding the first two columns)
    prediction_columns = data.columns[2:]

    # Compute RMSE for each model
    rmse_values = {}
    for model in prediction_columns:
        rmse_values[model] = rmse(data[model], target)

    # Compute share of RMSE for each model compared to the sum of all RMSE values
    total_rmse = sum(rmse_values.values())
    rmse_shares = {model: rmse_val / total_rmse for model, rmse_val in rmse_values.items()}

    # Compute weights inversely proportional to RMSE shares
    weights = {model: 1 / share for model, share in rmse_shares.items()}

    # Normalize weights to sum up to 1
    total_weight = sum(weights.values())
    normalized_weights = {model: weight / total_weight for model, weight in weights.items()}

    return normalized_weights


def compute_variance_weights(data):
    """
    Computes weights based on the inverse of prediction variances.

    Parameters:
        data (DataFrame): DataFrame containing actual values and predictions from different models.

    Returns:
        dict: Dictionary containing model names as keys and their respective weights.
    """
    # Select columns containing predictions (excluding the first two columns)
    prediction_columns = data.columns[2:]
    
    # Compute prediction variances for each model
    prediction_variances = {}
    for model in prediction_columns:
        prediction_variances[model] = np.var(data[model])
    
    # Compute the sum of the inverse prediction variances
    total_inverse_variance = sum(1 / variance for variance in prediction_variances.values())
    
    # Compute weights based on the inverse of prediction variances
    weights = {model: (1 / variance) / total_inverse_variance for model, variance in prediction_variances.items()}
    
    return weights


def compute_error_correlation_weights(data, verbose=True):
    """
    Computes the error correlation weights for different models.

    This function computes the error correlation weights for a set of models based on the error matrix C,
    where C_ij represents the correlation between the errors of model i and model j.

    Parameters:
        data (DataFrame): DataFrame containing actual values and predictions from different models.
        verbose (bool, optional): Whether to print information about the computed weights. Defaults to True.

    Returns:
        dict: Dictionary containing model names as keys and their corresponding weights.
    """
    # Filter out non-numeric columns
    numeric_data = data.select_dtypes(include=[np.number])

    # Drop the 'Actual' column and compute errors for each model
    errors = numeric_data.drop(columns=['Actual']).apply(lambda x: numeric_data['Actual'] - x)

    # Compute the number of models
    num_models = len(errors.columns)

    # Initialize the error matrix C with zeros
    error_matrix = np.zeros((num_models, num_models))

    # Compute the element-wise product of errors between each pair of models
    for i, model_i in enumerate(errors.columns):
        for j, model_j in enumerate(errors.columns):
            error_matrix[i, j] = np.sum(errors[model_i] * errors[model_j])

    # Divide by n to obtain the error covariance matrix
    error_matrix /= len(data)

    # Compute the inverse matrix
    inverse_error_matrix = np.linalg.inv(error_matrix)

    # Sum the elements in each row of the inverse error matrix to get model weights
    row_sums = np.sum(inverse_error_matrix, axis=1)

    # Divide each sum by the total sum of all elements in the inverse error matrix to normalize the weights
    total_sum = np.sum(inverse_error_matrix)
    model_names = data.columns[2:]  # Extract model names

    # Compute model weights
    model_weights = {model: row_sum / total_sum for model, row_sum in zip(model_names, row_sums)}    

    # Print information about the computed weights if verbose is True
    if verbose:
        total_weight = sum(model_weights.values())
        min_weight = min(model_weights.values())
        max_weight = max(model_weights.values())

        print(f'Checking weights...')
        print(f'Sum of weights: {total_weight}')
        print(f'Range of weights: [{min_weight}, {max_weight}]\n')
        print(f'Weights:')
        print(model_weights)

    return model_weights


def ensemble_predictions_given_weights(data, weights):
    """
    Computes predictions using the weighted average ensemble method with given weights.

    Parameters:
        data (DataFrame): DataFrame containing actual values and predictions from different models.
        weights (dict): Dictionary containing model names as keys and their respective weights.

    Returns:
        numpy.ndarray: Weighted average predictions.
    """
    # Select columns containing predictions (excluding the first two columns)
    prediction_columns = data.columns[2:]

    # Compute weighted average predictions
    weighted_predictions = np.sum(data[prediction_columns].values * np.array(list(weights.values())), axis=1)

    return weighted_predictions