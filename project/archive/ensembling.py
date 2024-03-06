# Ensemble ensemblers
print('Loading ensemble ensemblers and forecasters...')
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from utils.ensembling import equal_weights, inv_rmse_weights, inv_variance_weights, inv_error_cov_weights

# ------------------------------------------------
# Setting weighting schemes
WEIGHTING_SCHEMES = {
    'Simple': equal_weights,
    'Inverse RMSE': inv_rmse_weights,
    'Inverse Variance': inv_variance_weights,
    'Inverse Error Covariance': inv_error_cov_weights
}

# Setting meta forecasters
METAMODELS = {
    'SVR': {
        # Model object
        'model': SVR,
        'args': {
            'kernel': 'linear'
        }
    },
    'RandomForest': {
        'model': RandomForestRegressor,
        'args': {
            'n_estimators': 100,
            'random_state': None
        }
    }
}
# ------------------------------------------------

# Do not change
ENS_METHODS = {
    'weighted': WEIGHTING_SCHEMES,
    'meta': METAMODELS
}
