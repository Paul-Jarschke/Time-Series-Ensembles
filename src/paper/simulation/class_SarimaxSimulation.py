import os.path
from datetime import datetime
from src.utils.paths import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


# Define a custom class for SARIMAX simulation, extending functionality from statsmodels' SARIMAX class.
class SarimaxSimulation(sm.tsa.SARIMAX):
    """
    This module defines a custom class for SARIMAX simulation, extending functionality from statsmodels' SARIMAX class.
    It also defines the corresponding method for simulation.

    Args:
        order (tuple):                      ARIMA order.
        n_simulations (int, optional):      Number of simulations. Defaults to None.
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

    # Initialize models attributes
    def __init__(self, order, n_simulations=None, start=None, end=None, trend=False,
                 seasonal_order=None, trend_value=None, ar_coefs=None, ma_coefs=None, sar_coefs=None, sma_coefs=None,
                 shock_sigma2=None, bounded=False, lower_bound=None, upper_bound=None, exog=None, exog_coefs=None,
                 trend_choice="c",
                 *args, **kwargs):

        # Input validation
        if not (n_simulations or (start and end)):
            raise ValueError("Either nsimulations or start/end should be defined.")

        # Convert start and end dates to datetime objects
        self.start = datetime.strptime(start, "%Y-%m")
        self.end = datetime.strptime(end, "%Y-%m")

        # Generate an array of months between start and end dates
        self.months = np.array(
            pd.period_range(start=self.start, end=self.end, freq="M")
        )

        # Concatenate exogenous variables to pandas DataFrame if provided as a list of pandas DataFrames
        if isinstance(exog, list):
            exog = pd.concat(exog, axis=1)
        # Otherwise take as it is
        elif isinstance(exog, (pd.Series, pd.DataFrame)):
            self.exog = exog
        elif exog is not None:
            raise ValueError("Provided exogenous variable(s) must be a (list of) pandas Series.")

        # Determine the number of simulations if it is provided, check for valid input
        # Otherwise just take the length from start to end (months)
        self.nsimulations = (
            n_simulations if n_simulations is not None else len(self.months)
        )
        if self.nsimulations != len(self.months):
            raise ValueError(
                "'nsimulations' does not equal the number of periods between 'start' and 'end'."
            )

        # Initialize endogenous variable with empty dataset
        endog = np.zeros(self.nsimulations)

        # Determine trend choice
        if trend:
            trend_choice = trend_choice
            if trend_value is None:
                raise ValueError("You must provide trend value when trend is True.")
        else:
            trend_choice = None
            trend_value = None

        # Adjust coefficients based on provided orders
        # Seasonal AR
        if seasonal_order[0] == 0:
            sar_coefs = None
        # Seasonal MA
        if seasonal_order[2] == 0:
            sma_coefs = None

        if order[0] == 0:
            ar_coefs = None
        if order[2] == 0:
            ma_coefs = None

        # Initialize class attributes
        self.seed = None
        self.ma_coefs = ma_coefs
        self.sar_coefs = sar_coefs
        self.sma_coefs = sma_coefs
        self.exog_coefs = exog_coefs
        self.ar_coefs = ar_coefs
        self.bounded = bounded
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.trend_binary = trend
        self.trend_choice = trend_choice
        self.trend_value = trend_value

        # Create a list of parameters for SARIMAX model
        param_list = [
            trend_value,
            exog_coefs,
            ar_coefs,
            ma_coefs,
            sar_coefs,
            sma_coefs,
            shock_sigma2,
        ]

        # Only add parameter to class parameter list attribute when not None
        self.params = []
        for param in param_list:
            if param is not None:
                # Input validation
                if isinstance(param, list):
                    pass
                if isinstance(param, int) or isinstance(param, float):
                    param = [param]

                for p in param:
                    if not isinstance(p, (int, float)):
                        raise ValueError(f"Provided parameter {param} must be int or float.")
                self.params += param

        # Initialize statsmodels SARIMAX model with specified parameters
        super().__init__(
            endog=endog,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend_choice,
            exog=exog,
            *args,
            **kwargs,
        )

        # Validate parameter count
        if len(self.params) != len(self.param_names):
            raise ValueError(
                f"Parameters are not correctly defined. Expected {len(self.param_names)} parameters, but provided "
                f"{len(self.params)}. Check that you have defined all of {self.param_names}."
            )

    # Define simulation method to extend functionality of parent class
    def simulate(self, seed=None, *args, **kwargs):
        """
        Simulate method to generate time series data.

        Args:
            seed (int, optional): Random seed. Defaults to None.

        Returns:
            SimulatedSeries: Simulated time series data.

        """
        # Calculate seed value for reproducibility
        # seed = (
        #         987
        #         + seed
        #         + int(np.sum(self.params))
        #         + int(np.sum(self.seasonal_order))
        #         + int(np.sum(self.order))
        #         + int(self.trend_binary)
        # )
        np.random.seed(seed)
        self.seed = seed

        # Generate random shocks (error term) coming from a normal distribution
        # Mean zero and defined error variance (last position in params)
        shocks = np.random.normal(
            size=self.nsimulations, loc=0, scale=np.sqrt(self.params[-1])
        )

        # Simulate series using parent class method (from statsmodels)
        series = super().simulate(
            params=self.params,
            nsimulations=self.nsimulations,
            random_state=seed,
            measurement_shocks=shocks,
            exog=self.exog  # ??
        )

        # Apply bounding if specified (this just scales and moves the series)
        # For more realistic series that fall between certain intervals (e.g. are positive etc.)
        if self.bounded:
            a, b = self.lower_bound, self.upper_bound
            series = ((series - min(series)) / (max(series) - min(series))) * (
                    b - a
            ) + a

        # Add all attributes from parent class to series

        # Assign our custom SimulatedSeries class to series (defined below this class)
        series = SimulatedSeries(
            series,
            seed=seed,
            exog=self.exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            months=self.months,
            trend=self.trend_binary,
        )

        return series


# Define a custom Series class for resulting Simulated Series that simulated data is assigned to (inherits from
# pd.Series; e.g., for plotting)
class SimulatedSeries(pd.Series):
    """
    Custom class for Simulated Series. Inherits from pandas' Series class and adds attributes for simulation details.
    Adjusts pandas .plot() method for our purposes.

    Args:
        series (pd.Series): Simulated series.
        seed (int): Random seed.
        order (tuple): ARIMA order.
        seasonal_order (tuple): Seasonal order.
        trend (bool): Flag indicating trend presence.
        exog (DataFrame): Exogenous variables.

    """

    def __init__(self, series, seed, order, seasonal_order, trend, exog, months, *args, **kwargs):

        # Assign attributes from parent class
        super().__init__(data=series, index=months, name="simulated_data")

        # Initialize series attributes
        self.index.name = "Month"
        self.seed = seed
        self.series = series
        self.months = super().index
        self.seasonal_order = seasonal_order
        self.order = order

        self.exog = exog
        self.trend = trend
        # Determine the number of exogenous variables
        self.n_exog = 0 if self.exog is None else self.exog.shape[1]

    # Plot method for visualizing the simulated series/overwrite pandas plotting method
    def plot(self, title=None, export_name=None, export_path=None, custom_ax=None, *args, **kwargs):
        """
        Plot method for visualizing the simulated series.

        Args:
            export_path:
            title (str, optional): Plot title. Defaults to "Simulation".
            title (str, optional): When given, adding a custom superior title.
            custom_ax (pyplot axis object, optional): An axis to plot to. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """

        plotted_series = self.series
        plotted_series = pd.Series(plotted_series, index=[str(month) for month in self.months])

        plot_title = "ARIMA"  # Initialize title

        seasonal_label = (
            f"${self.seasonal_order[:3]}_{{{self.seasonal_order[3]}}}$"
        )

        # Adjust plot title based on model components
        plot_title += f"{'X' if self.n_exog > 0 else ''}{self.order}"
        if not all(x == 0 for x in self.seasonal_order[:3]):
            plot_title = f"S{plot_title}{seasonal_label}"

        if not title:
            if self.trend:
                plot_title = plot_title + " with Trend"
                if self.n_exog > 0:
                    plot_title = (
                            plot_title + f" and Covariate Influence"
                    )
            elif self.n_exog > 0:
                plot_title = (
                        plot_title + f" with Covariate Influence"
                )

        # Replace axis if specified
        if custom_ax:
            ax = custom_ax
            if title in ['Exogenous 1', 'Exogenous 2', 'Exogenous 3']:
                plot_title = None
            else:
                plot_title = title
        else:
            fig, ax = plt.subplots()
            fig.set_size_inches(6, 5)
            if title:
                if "\n" in title:
                    ax.get_figure().suptitle(title, size=16, y=1.025)
                else:
                    ax.get_figure().suptitle(title, size=16, y=0.98)

        ax.set_title(plot_title, size=14)

        plotted_series.plot(ax=ax, *args, **kwargs)

        freq = 12
        xticks_positions = np.arange(0, len(self.series), freq * 2)
        xticks_labels = self.months[xticks_positions]
        ax.set_xticks(xticks_positions, xticks_labels, rotation=45)
        # ax.set_xlabel("Time", fontsize=12)

        if not ax:
            ax.set_ylabel("Value", fontsize=14)
        elif title in ["Exogenous 1", "Weak Seasonal ARIMA"]:
            ax.set_ylabel("Value", fontsize=14)

        if not ax:
            ax.set_xlabel("Time", fontsize=14)
        elif title in ["Strong Seasonal ARIMA\nwith Covariate Influence",  "Exogenous 2"]:
            ax.set_xlabel("Time", fontsize=14)

        # self.month_strings = [str(month) for month in self.months]

        if export_name:
            if export_path:
                export_full_path = os.path.join(export_path, export_name)
            else:
                export_full_path = os.path.join(os.getcwd(), export_name)
            plt.savefig(export_full_path, dpi=300)

        # When no axis provided show and close plot.
        if not ax:
            return plt.show()
