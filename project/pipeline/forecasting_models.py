###########################################
## Setting up list of forecasting models ##
###########################################
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import STLForecaster
from darts.models import XGBModel


##################
## Naive (last) ##
##################
model_naive = NaiveForecaster(
    strategy="last"
)

###################
## Naive (drift) ##
###################
model_naive_drift = NaiveForecaster(
    strategy="drift"
)

#################
## Auto-SARIMA ##
#################
model_SARIMA = AutoARIMA(
    seasonal=True,
    stationary=False,
    d=1,
    sp=12,
    trace=False,
    with_intercept="auto",
    max_p=6,
    max_q=6,
    suppress_warnings=True#,
    #simple_differencing=True
)

##################
## Auto-SARIMAX ##
##################
model_SARIMAX = AutoARIMA(
    seasonal=True,
    stationary=False,
    d=1,
    sp=12,
    trace=False,
    with_intercept="auto",
    max_p=6,
    max_q=6,
    suppress_warnings=True#,
    #simple_differencing=True
)

##################################
## Simple Exponential Smoothing ##
##################################
model_ES = ExponentialSmoothing(
)

###################
## Cubic Splines ##
###################

###########
## Theta ##
###########
model_theta = ThetaForecaster(
)

#########
## STL ##
#########
model_STL = STLForecaster(
)

#############
## XGBoost ##
#############
var_lags = 12

model_XGB = XGBModel(
    lags=var_lags,
    lags_future_covariates=None
)

#############################
## XGBoost with Covariates ##
#############################
var_lags = 12

model_XGB_covs = XGBModel(
    lags=var_lags,
    lags_future_covariates=(var_lags,0)
)



######################################################################################################

models = {"Naive": model_naive, 
          "Naive (drift)": model_naive_drift, 
          "AutoSARIMA": model_SARIMA, 
          "AutoSARIMAX": model_SARIMAX, 
          "Exponential Smoothing": model_ES,
          "Theta": model_theta,
          "STL": model_STL,
          "XGBoost": model_XGB,
          "XGBoost (+ covs)": model_XGB_covs
          }

