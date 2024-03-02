# Ensemble methods
print('Loading ensemble methods and models...')
from utils.ensembling import equal_weights, inv_rmse_weights, inv_variance_weights, inv_error_cov_weights
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# ------------------------------------------------
# Setting weighting schemes
weighting_schemes = {
    'Simple': equal_weights,
    'Inv_RMSE': inv_rmse_weights,
    'Inv_Variance': inv_variance_weights,
    'Inv_ErrorCov': inv_error_cov_weights
}

# Setting meta models
metamodels = {
    'SVR': {
        # Model object
        'model': SVR,
        'options': {
            'kernel': 'linear'
        }
    },
    'RandomForest': {
        'model': RandomForestRegressor,
        'options': {
            'n_estimators': 100,
            'random_state': None
        }
    }
}
# ------------------------------------------------

# Do not change
ensemble_methods = {
    'weighted': weighting_schemes,
    'meta': metamodels
}
