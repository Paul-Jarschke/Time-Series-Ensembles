from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Define a custom class for SARIMAX simulation, extending functionality from statsmodels' SARIMAX class.
class SARIMAX_Simulation(sm.tsa.SARIMAX):
    """ 
    This module defines a custom class for SARIMAX simulation, extending functionality from statsmodels' SARIMAX class. 
    It also defines a custom Series class for Simulated Series and methods for simulation and visualization.
    
    Custom SARIMAX simulation class.

    Inherits from statsmodels' SARIMAX class and extends functionality for simulation and visualization.

    Args:
        order (tuple):                      ARIMA order.
        nsimulations (int, optional):       Number of simulations. Defaults to None.
        start (str, optional):              Start date. Defaults to None.
        end (str, optional):                End date. Defaults to None.
        trend (bool, optional):             Flag indicating trend presence. Defaults to False.
        seasonal_order (tuple, optional):   Seasonal order. Defaults to None.
        trend_value (float, optional):      Trend value. Defaults to None.

        ar_coefs (list, optional):          AR coefficients. Defaults to None.
        ma_coefs (list, optional):          MA coefficients. Defaults to None.
        sar_coefs (list, optional):         Seasonal AR coefficients. Defaults to None.
        sma_coefs (list, optional):         Seasonal MA coefficients. Defaults to None.

        shock_sigma2 (float, optional):     Shock variance. Defaults to None.
        bounded (bool, optional):           Flag indicating whether series should be bounded. Defaults to False.
        lower_bound (float, optional):      Lower bound for bounded series. Defaults to None.
        upper_bound (float, optional):      Upper bound for bounded series. Defaults to None.
        exog (DataFrame, optional):         Exogenous variables. Defaults to None.
        exog_coefs (list, optional):        Exogenous coefficients. Defaults to None.

        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Raises:
        ValueError: If `nsimulations` or `start/end` is not defined properly.

    """
    
    def __init__(self, order, nsimulations=None, start=None, end=None, trend=False, 
                 seasonal_order=None, trend_value=None, ar_coefs=None, ma_coefs=None,
                 sar_coefs=None, sma_coefs=None, shock_sigma2=None, bounded=False,
                 lower_bound=None, upper_bound=None, exog=None, exog_coefs=None,
                 *args, **kwargs):
        
        # Check if necessary parameters are provided
        if all(arg is None for arg in (nsimulations, start, end)):
            raise ValueError("Either nsimulations or start/end should be defined.")
        
        # Convert start and end dates to datetime objects
        self.start = datetime.strptime(start, "%m/%Y")
        self.end = datetime.strptime(end, "%m/%Y")
        
        # Generate an array of months between start and end dates
        self.months = np.array(pd.period_range(start=self.start, end=self.end, freq='M'))
        
        # Concatenate exogenous variables if provided as a list
        if isinstance(exog, list):
            exog = pd.concat(exog, axis=1)
        self.exog = exog

        # Determine the number of simulations
        self.nsimulations = nsimulations if nsimulations is not None else len(self.months)
        if self.nsimulations != len(self.months):
            raise ValueError("'nsimulations' does not equal the number of periods between 'start' and 'end'.")
        
        # Store trend settings
        self.bounded = bounded
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.trend_binary = trend
        
        # Initialize endogenous dataset
        empty_dataset = np.zeros(self.nsimulations)
        endog = empty_dataset
    
        # Determine trend choice
        if self.trend_binary:
            trend_choice = 'c'  
        else:
            trend_choice = None
            trend_value = None
            
        # Adjust coefficients based on provided orders
        if seasonal_order[0] == 0:
            sar_coefs = None
        if seasonal_order[2] == 0:
            sma_coefs = None
            
        if order[0] == 0:
            ar_coefs = None
        if order[2] == 0:
            ma_coefs = None
            
        # Create a list of parameters for SARIMAX model
        param_list = [trend_value, exog_coefs, ar_coefs, ma_coefs, sar_coefs, sma_coefs, shock_sigma2]
        self.params = []
        for param in param_list: 
            if param is not None:
                if isinstance(param, int) or isinstance(param, float):
                    param = [param]
                self.params += param
                  
        # Initialize SARIMAX model with specified parameters
        super().__init__(endog=endog, order=order, seasonal_order=seasonal_order,
                         trend=trend_choice, exog=exog, *args, **kwargs)
        
        # Validate parameter count
        if len(self.params) != len(self.param_names):  
            raise ValueError(f"Parameters are not correctly defined. Expected {len(self.param_names)} parameters, but provided {len(self.params)}. Check that you have defined all of {self.param_names}.")
    
    # Define a custom Series class for Simulated Series
    class SimulatedSeries(pd.Series):
        """ 
        Custom class for Simulated Series.

        Inherits from pandas' Series class and adds attributes for simulation details.

        Args:
            series (pd.Series): Simulated series.
            seed (int): Random seed.
            order (tuple): ARIMA order.
            seasonal_order (tuple): Seasonal order.
            trend (bool): Flag indicating trend presence.
            exog (DataFrame): Exogenous variables.

        """
        def __init__(self, series, seed, order, seasonal_order, trend, exog):
            super().__init__(data=series)

            # Initialize series attributes
            self.seed = seed
            self.series = series
            self.months = series.index
            self.month_strings = [str(month) for month in self.months]
            self.seasonal_order = seasonal_order
            self.order = order
            self.seasonal_label = f"${self.seasonal_order[:3]}_{{{self.seasonal_order[3]}}}$"
            
            self.plot_title = "ARIMA"
            self.exog = exog
            self.trend = trend
            
        # Plot method for visualizing the simulated series
        def plot(self, title="Simulation", *args, **kwargs):
            """ 
            Plot method for visualizing the simulated series.

            Args:
                title (str, optional): Plot title. Defaults to "Simulation".
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.

            """
            # Determine the number of exogenous variables
            self.nexog = 0 if self.exog is None else self.exog.shape[1]
    
            # Adjust plot title based on model components
            self.plot_title += f"{'X' if self.nexog > 0 else ''}{self.order}"
            if not all(x == 0 for x in self.seasonal_order[:3]):
                self.plot_title = f"S{self.plot_title}{self.seasonal_label}"
            
            if self.trend:
                self.plot_title = self.plot_title + " with trend"
                if self.nexog > 0:
                    self.plot_title = self.plot_title + f" and {self.nexog} exogenous variables"
            elif self.nexog > 0:
                    self.plot_title = self.plot_title + f" with {self.nexog} exogenous variables"
            
            # Generate the plot
            plt.figure(figsize=(6, 5))
            plt.suptitle(title, size=16, y=0.98)
            plt.title(f"{self.plot_title}", size=14)
            freq = 12
            xticks_positions = np.arange(0, len(self.series), freq*2) 
            plt.xticks(ticks=xticks_positions, rotation=45)
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.plot(self.month_strings, self.series, label='Simulated Data', *args, **kwargs)
            plt.show()
            self.plot_title = "ARIMA"  # Reset title

    # Define simulation method to extend functionality of parent class
    def simulate(self, seed=None):
        """ 
            Simulate method to generate time series data.

            Args:
                seed (int, optional): Random seed. Defaults to None.

            Returns:
                SimulatedSeries: Simulated time series data.

            """
        # Calculate seed value for reproducibility
        seed = 987 + seed + int(np.sum(self.params)) + int(np.sum(self.seasonal_order)) + int(np.sum(self.order)) + int(self.trend_binary)
        np.random.seed(seed)
        
        # Generate random shocks
        shocks = np.random.normal(size=self.nsimulations, loc=0, scale=np.sqrt(self.params[-1]))
        
        # Simulate series using parent class method
        series = super().simulate(params=self.params, 
                                  nsimulations=self.nsimulations, 
                                  random_state=seed,
                                  measurement_shocks=shocks)
        
        # Apply bounding if specified
        if self.bounded:
            a, b = self.lower_bound, self.upper_bound
            series = ((series - min(series)) / (max(series) - min(series))) * (b-a) + a 
        
        # Create pandas Series object with appropriate index
        series = pd.Series(series, index=self.months)
        series.name = "simulated_data"
        series.index.name = "Month"
        series = self.SimulatedSeries(series,
                                      seed=seed,
                                      exog=self.exog,
                                      order=self.order, 
                                      seasonal_order=self.seasonal_order,
                                      trend=self.trend_binary)
        
        return series