# Import modules
import numpy as np
import pandas as pd
from SARIMAX_class import SARIMAX_Simulation


# Set number of periods
start = "01/2014"
end = "12/2024"

# Alternative (has to be specified in function call !)
#years = 10
#nsimulations = years * 12 

# Directory to store simulated data
sim_dir = r'C:/Users/Work/OneDrive/GAU/3. Semester/Statistisches Praktikum/Git/Ensemble-Techniques-TS-Forecasting/Data/Simulations/'

# Set Seed
seed = 42

######################
## Exogeneous var 1 ##
######################

exo1_model = SARIMAX_Simulation(
    start=start, end=end,
    order=(1, 1, 3),                # p, d, q
    seasonal_order=(2, 0, 0, 12),   # P, D, Q, m
    trend=True, trend_value=0.94,
    ar_coefs=[0.2], ma_coefs=[0.4, 0.25, 0.1], sar_coefs=[0.05, 0.01], sma_coefs=[],
    shock_sigma2=100, #shock term variance
    bounded=True, lower_bound=200, upper_bound=500
)

exo1_ts = exo1_model.simulate(seed)


######################
## Exogeneous var 2 ##
######################

exo2_model = SARIMAX_Simulation(
    start=start, end=end,
    order=(2, 1, 4),               # p, d, q
    seasonal_order=(1, 0, 0, 12),  # P, D, Q, m
    trend=False, trend_value=-0.1,
    ar_coefs=[0.7, 0.01], ma_coefs=[0.5, 0.2, 0.1, -0.1], sar_coefs=[0.2], sma_coefs=[],
    shock_sigma2=1,  # shock term variance
    bounded=True, lower_bound=40, upper_bound=80
)

exo2_ts = exo2_model.simulate(seed)


######################
## Exogeneous var 3 ##
######################

exo3_model = SARIMAX_Simulation(
    start=start, end=end,
    order=(3, 1, 1),               # p, d, q
    seasonal_order=(2, 0, 1, 12),  # P, D, Q, m
    trend=True, trend_value=0.1,
    ar_coefs=[0.15, -0.01, -0.01], ma_coefs=[0.35], sar_coefs=[0.3, 0.2], sma_coefs=[-0.2],
    shock_sigma2=4,  # shock term variance
    bounded=True, lower_bound=900, upper_bound=950
)

exo3_ts = exo3_model.simulate(seed)

