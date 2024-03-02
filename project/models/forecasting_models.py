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

################
# Naive (last) #
################
model_naive = NaiveForecaster(
    strategy="last"
)

#################
# Naive (drift) #
#################
model_naive_drift = NaiveForecaster(
    strategy="drift"
)

###############
# Auto-SARIMA #
###############
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

################
# Auto-SARIMAX #
################
model_SARIMAX = AutoARIMA(
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

################################
# Simple Exponential Smoothing #
################################
model_ES = ExponentialSmoothing(
)

#################
# Cubic Splines #
#################

#########
# Theta #
#########
model_theta = ThetaForecaster(
    sp=12  # todo: inferred_sp = 12 if target.index.freq == "M" else NotImplementedError("Implement me for daily etc")
)

#######
# STL #
#######
model_STL = STLForecaster(
    sp=12  # todo inferred_sp = 12 if target.index.freq == "M" else NotImplementedError("Implement me for daily etc")
)

###########
# XGBoost #
###########
var_lags = 12  # todo: adjust depending on frequency; google what value to choose!

model_XGB = XGBModel(
    lags=var_lags
)

###########################
# XGBoost with Covariates #
###########################
model_XGB_X = XGBModel(
    lags=var_lags,
    lags_past_covariates=var_lags
)
# --------------------------------------------------

forecasting_models = {
    "Naive": model_naive,
    "Naive (drift)": model_naive_drift,
    "AutoSARIMA": model_SARIMA,
    "AutoSARIMAX": model_SARIMAX,
    "Exponential Smoothing": model_ES,
    # "Theta": model_theta,
    "STL": model_STL,
    "XGBoost": model_XGB,
    "XGBoost (+ X)": model_XGB_X
}
