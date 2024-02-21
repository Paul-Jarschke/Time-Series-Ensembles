###########################################
## Setting up list of forecasting models ##
###########################################
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import STLForecaster

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

##################
## Auto-SARIMAX ##
##################
model_SARIMAX = AutoARIMA(
    seasonal=True,
    sp=12,
    trace=True,
    with_intercept="auto",
    max_p=12,
    max_q=12,
    suppress_warnings=True
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

######################################################################################################

models = {"Naive": model_naive, 
          "Naive (drift)": model_naive_drift, 
          "SARIMAX": model_SARIMAX, 
          "Exponential Smoothing": model_ES,
          "Theta": model_theta,
          "STL": model_STL}

