#########################################
# Setting up list of forecasting models #
#########################################
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import STLForecaster
from darts.models import XGBModel

print("Loading individual forecasting models...")

# Naive (last)
model_naive = NaiveForecaster(
    strategy="last"
)

# Naive (drift)
model_naive_drift = NaiveForecaster(
    strategy="drift"
)

# Auto-SARIMA
model_SARIMA = AutoARIMA(
    seasonal=True,
    stationary=False,
    d=1,
    trace=False,
    update_pdq=False,
    with_intercept="auto",
    max_p=6,
    max_q=6,
    suppress_warnings=True  # ,
    # simple_differencing=True
)

# Simple Exponential Smoothing
model_ES = ExponentialSmoothing(
)

# Cubic Splines
# (placeholder)

# Theta
model_theta = ThetaForecaster(
)

# STL
model_STL = STLForecaster(
)

# XGBoost
var_lags = 10

# Without covariates
model_XGB = XGBModel(
    lags=var_lags
)

# With covariates
model_XGB_X = XGBModel(
    lags=var_lags,
    lags_past_covariates=var_lags
)

# --------------------------------------------------

FC_MODELS = {
    'no_covariates': {
        'Naive': model_naive,
        'Naive (drift)': model_naive_drift,
        'AutoSARIMA': model_SARIMA,
        'Exponential Smoothing': model_ES,
        # 'Theta': model_theta,
        'STL': model_STL,
        'XGBoost': model_XGB,
    },
    'covariates': {
        'XGBoost': model_XGB_X,
        'AutoSARIMAX': model_SARIMA
    }
}
