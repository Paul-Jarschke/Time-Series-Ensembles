import pandas as pd


def get_metamodel_prediction(
    train_data, next_indiv_predictions, metamodel, *args, **kwargs
):
    """
    Predicts ensemble forecasts using a metamodel.

    Args:
        train_data (pd.DataFrame):              Training data consisting of features and target variable.
        next_indiv_predictions (pd.DataFrame):  Predictions of individual forecasters for the next period.
        metamodel (Any):                        Meta-model for ensemble prediction, must have 'fit' and 'predict' methods.

        *args:      Variable length argument list.
        **kwargs:   Arbitrary keyword arguments.

    Returns:
        pd.Series: Ensemble predictions for the next given period.

    Raises:
        ValueError: If provided metamodel lacks 'fit' and 'predict' methods.
    """

    # Validation: check if provided metamodel is indeed a model with fit and predict methods
    if (
        hasattr(metamodel, "fit")
        and callable(getattr(metamodel, "fit"))
        and hasattr(metamodel, "predict")
        and callable(getattr(metamodel, "predict"))
    ):

        # Split features (= predictions of individual forecasters) and target variable (actual values)
        # Make a copy to avoid manipulating the original dataset
        train_data = train_data.copy()
        target = train_data.pop("Target")
        predictions = train_data

        # Model is already constructed by the model_crafter, thus it can already be trained
        # Train Meta-Model
        metamodel.fit(predictions, target)

        # If target column is still present in the dataframe of the covariates (= historical predictions of individual
        # forecasters) remove it
        # This for now only concerns when next_indiv_predictions is from the historical predictions but not form the
        # future predictions
        if "Target" in next_indiv_predictions.columns:
            next_indiv_predictions = next_indiv_predictions.copy().drop(columns=["Target"])

        # Make ensemble predictions for next given period (horizon is automatically taken from the number of rows in
        # next_indiv_predictions)
        ensemble_prediction = metamodel.predict(next_indiv_predictions)

        # Ensure that output is a pandas Series
        if not isinstance(ensemble_prediction, pd.Series):
            ensemble_prediction = pd.Series(
                data=ensemble_prediction, index=next_indiv_predictions.index
            )
    else:
        raise ValueError("Meta model must have 'fit' and 'predict' ensemblers.")

    return ensemble_prediction
