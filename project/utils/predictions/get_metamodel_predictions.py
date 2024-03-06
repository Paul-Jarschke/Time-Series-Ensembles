import pandas as pd


def get_metamodel_prediction(train_data, next_indiv_predictions, metamodel, *args, **kwargs):

    # Validation: check if provided metamodel is indeed a model with fit and predict methods
    if (hasattr(metamodel, 'fit') and callable(getattr(metamodel, 'fit')) and
            hasattr(metamodel, 'predict') and callable(getattr(metamodel, 'predict'))):

        # Split features (= predictions of individual forecasters) and target variable (actual values) for training
        train_data = train_data.copy() # Make a copy to avoid manipulating the original dataset
        target = train_data.pop('Target')
        predictions = train_data

        # Model is already constructed by the model_crafter, thus it can already be trained

        # Train Meta-Model
        metamodel.fit(predictions, target)

        # Also remove target from the covariates (= predictions of individual forecasters) employed in the
        # ensemble forecast for the next period
        next_indiv_predictions = next_indiv_predictions.copy().drop(columns=['Target'])

        # Make ensemble predictions for next given period
        ensemble_prediction = metamodel.predict(next_indiv_predictions)

        # Ensure that output is a pandas Series
        if not isinstance(ensemble_prediction, pd.Series):
            ensemble_prediction = pd.Series(data=ensemble_prediction, index=next_indiv_predictions.index)
    else:
        raise ValueError('Meta model must have \'fit\' and \'predict\' ensemblers.')

    return ensemble_prediction
