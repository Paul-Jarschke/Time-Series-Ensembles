# Import modules
import numpy as np
import pandas as pd

from src.utils.paths import *
from src.simulation.class_SarimaxSimulation import SarimaxSimulation

# Set number of periods
start = "2004-01"
end = "2023-12"

# Alternative (has to be specified as arg in function call !)
# years = 10
# nsimulations = years * 12

# Set Seed
seed = 42

###########################
## Exogeneous Variable 1 ##
###########################

exo1_model = SarimaxSimulation(
    start=start, end=end,
    order=(1, 1, 3),  # ARIMA (p,d,q)
    seasonal_order=(2, 0, 0, 12),  # Seasonal Component (P,D,Q,m)
    trend=True, trend_value=0.94,
    ar_coefs=[0.2], ma_coefs=[0.4, 0.25, 0.1],
    sar_coefs=[0.05, 0.01], sma_coefs=[],
    shock_sigma2=100,  # shock/error term variance
    bounded=True, lower_bound=200, upper_bound=500
)

exo1_ts = exo1_model.simulate(seed)

###########################
## Exogeneous Variable 2 ##
###########################

exo2_model = SarimaxSimulation(
    start=start, end=end,
    order=(2, 1, 4),  # ARIMA (p,d,q)
    seasonal_order=(1, 0, 0, 12),  # Seasonal Component (P,D,Q,m)
    trend=False, trend_value=-0.1,
    ar_coefs=[0.7, 0.01], ma_coefs=[0.5, 0.2, 0.1, -0.1],
    sar_coefs=[0.2], sma_coefs=[],
    shock_sigma2=1,  # shock/error term variance
    bounded=True, lower_bound=40, upper_bound=80
)

exo2_ts = exo2_model.simulate(seed)

######################
## Exogeneous Var 3 ##
######################

exo3_model = SarimaxSimulation(
    start=start, end=end,
    order=(3, 1, 1),  # ARIMA (p,d,q)
    seasonal_order=(2, 0, 1, 12),  # Seasonal Component (P,D,Q,m)
    trend=True, trend_value=0.1,
    ar_coefs=[0.15, -0.01, -0.01], ma_coefs=[0.35],
    sar_coefs=[0.3, 0.2], sma_coefs=[-0.2],
    shock_sigma2=4,  # shock/error term variance
    bounded=True, lower_bound=900, upper_bound=950
)

exo3_ts = exo3_model.simulate(seed)

###########################
##  Endogenous Variable  ##
###########################

# Define exogenous variables and respective coefficients
exo_ts = [exo1_ts, exo2_ts, exo3_ts]
exo_coefs = [0.3, 0.15, -0.1]

endo_model = SarimaxSimulation(
    start=start, end=end,
    exog=exo_ts,
    exog_coefs=exo_coefs,
    order=(1, 1, 5),
    seasonal_order=(4, 0, 0, 6),
    trend=True, trend_value=0.1,
    ar_coefs=[0.4], ma_coefs=[0.15, 0.05, -0.005, -0.01, -0.01],
    sar_coefs=[0.1, -0.1, 0.05, -0.01], sma_coefs=[],
    shock_sigma2=5,
    bounded=True, lower_bound=40, upper_bound=90)

endo_ts = endo_model.simulate(seed)

# Store all series in list
series = [endo_ts]
series.extend(exo_ts)

#####################################
##      Add Noise to Variables     ##
## to Immitate Measurement Error   ##
#####################################

# Define the range for mean and standard deviation
mean_range = (-0.8, 0.8)
std_dev_range = (0.2, 0.9)

# Create a list to store the noisy time series
noisy_series = []

##########################################
##  Set up DataFrame with noisy series  ##
##########################################

noisy_df = pd.DataFrame(series)
for ts in series:
    # Randomly draw mean and standard deviation
    mean = np.random.uniform(*mean_range)
    std_dev = np.random.uniform(*std_dev_range)

    # Generate random noise
    noise = np.random.normal(mean, std_dev, len(ts))

    # Add noise to the original time series
    noisy_ts = ts + noise

    noisy_df = pd.concat([noisy_df, noisy_ts], axis=1)

# Rename Columns
noisy_df.columns = ['y', "x1", "x2", "x3"]

# Rename Index to 'Date'
noisy_df.index.name = "Date"
### noisy_df.insert(0, 'Date', endo_ts.index.to_series())

# Format periodic 'date' index to ISO YYYY-MM-DD format
noisy_df.index = pd.PeriodIndex(noisy_df.index, freq="M").to_timestamp().strftime('%Y-%m-%d')

##########################
##  Export Data as CSV  ##
##########################

### die Daten, die wir final verwenden sollten wir zur Transparenz trotz pipeline exportieren
filename = f'noisy_simdata_neu.csv'
noisy_df.to_csv(os.path.join(SIMDATA_DIR, filename))
