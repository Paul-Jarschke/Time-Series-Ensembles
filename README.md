<p align="center" width="100%" >
    <img width="66%" src="https://github.com/Paul-Jarschke/Time-Series-Ensembles/blob/main/paper/plots/TimeTango.png">
</p>
  <h1 align="center" style="font-size:23">
A Framework for Ensemble Time Series Forecasts
  </h1>

## :bulb: Overview

TimeTango is the result of our participation in the Practical Statistical Training project offered for students in 
the Master of Applied Statistics at the University of Göttingen. Developed by [Paul 
Jarschke](https://www.linkedin.com/in/paul-jarschke-b4439b1a3/) and [Leon Löppert](https://www.linkedin.com/in/leonloeppert/), this framework focuses on ensemble methods for time series forecasting. 
The framework primarily implements the pipeline outlined in our research paper, encompassing data preprocessing, 
individual forecasts, ensemble forecasts, performance evaluation, and future forecasting. We leverage single 
forecasters from the [sktime](https://www.sktime.net/) and [darts](https://unit8co.github.io/darts/) packages. 
Additionally, TimeTango offers the flexibility to incorporate meta-regressors from the [scikit-learn](https://scikit-learn.org/stable/) module for 
Python.

## :hourglass_flowing_sand: Installation Guide

To use TimeTango, follow these steps:

1. Clone the repository:
   ```bash
    git clone https://github.com/Paul-Jarschke/Time-Series-Ensembles.git
   ```
2. Navigate to the cloned directory:
    ```bash
    cd Time-Series-Ensembles
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. You're all set! Start using TimeTango by exploring the provided notebooks and scripts. If you encounter any issues or have feedback, please don't hesitate to reach out to us via LinkedIn.

## :file_folder: Folder Structure

The TimeTango repository is organized into the following directories:

- `data`: Contains various datasets used for experimentation and analysis, including simulated data, test data, 
  covariates and EUR-USD exchange rate data. In particular, it includes the data used in our analysis, i.e., 
  simulated datasets 
  demonstrating increasing patterns (from random walk to SARIMAX) and the simulated covariates.

- `paper`: Holds notebooks used for the analysis presented in our paper, ensuring reproducibility. These 
  are related to data simulation, execution of the pipeline for simulated and EUR-USD exchange rate data, 
  configuration files for individual and ensemble models, and implemented metrics. Additionally, it contains an analysis notebook responsible for generating figures and tables presented in the paper.

- `src`: Houses all Python source code files, including the main pipeline and its subprocesses such as 
  preprocessing, individual forecasts, ensemble forecasts, performance evaluation, and future forecasts. It also 
  contains helper functions, visualization tools, and code for the custom weighting schemes for weighted ensembles.

- `user`: This directory is designed for users (like you :wink:) and includes a Jupyter notebook named `run_pipeline.ipynb`, which 
  allows users to execute the main `run_pipeline()` function. Additionally, it contains the necessary input 
  configuration files for the pipeline (`ensemblers.yml`, `forecasters.yml`, and `metrics.yml`), as well as a folder for 
  pipeline outputs (historical and future forecasts, accuracy ranking table, and log files). All necessary inputs are preloaded for the EUR-USD exchange rate data example. 

Feel free to explore these directories to better understand the structure and functionalities of TimeTango.


## :rocket: Let's Generate Ensemble Predictions!

1. **Load dataset**:

   Start by loading your dataset. You can either use one of the sample datasets provided in the **data** folder or upload a new one, preferably in CSV format. Import it into your environment as a pandas DataFrame or Series containing timestamps, target values, and optionally covariates. You can leverage our implemented `csv_reader()` function from the `src.utils.helpers.csv_reader` module, which offers convenient features.


2. **Configure your single forecasters (first stage)**:

   Open the `forecasters.yml` configuration file and familiarize yourself with its structure. At the top level, there 
   is a differentiation between models using covariates and those that do not. Within each group, the structure includes the model name (arbitrary) followed by `model`, `package`, and `options`:
   - `model`: The forecaster's class name from the sktime or darts package.
   - `package`: The package name (darts or sktime).
   - `options`: Input arguments specific to the forecaster's class, organized in sublevels.

   Here is an example:
   ```yaml
    without_covariates:  # These models do not include covariates for fitting and prediction.
      Naive: 
        model: NaiveForecaster
        package: sktime
        options:  # These are the input arguments involved in the forecaster's class.
          strategy: last
      AutoTheta:
        model: StatsForecastAutoTheta
        package: darts
        options:
          season_length: 5
      ...
    with_covariates:  # These models do include covariates for fitting and prediction.
    # If you do not want to include covariate models just leave an empty {} behind the "with_covariates".
      AutoSARIMAX: # Note that this is the forecaster's name! If you want to select_forecasters, use this!
        model: AutoARIMA
        package: sktime
        options:
          seasonal: True
          stationary: False
          d: 1
          trace: False
          update_pdq: False
          with_intercept: auto
          max_p: 5
          max_q: 5
          suppress_warnings: True
      ...
   ```
   

3. **Configure your ensemble approaches (second stage)**:

   Open the `ensemblers.yml` configuration file to understand its structure. At the top level, differentiate between 
   weighing schemes and meta-models used for stacking. For weighing schemes, define each scheme's name mapped to 
   its corresponding Python file in `/src/models/ensemblers/weighted/`.

   If desired, introduce new weighing schemes into this folder and the pipeline recognizes it automatically. These 
   weighing schemes should accept a pandas DataFrame of forecasts and return a named dictionary of weights.

   For meta-models, follow the same structure as for forecasting models: `model`, `package`, and `options` at the 
   third level. Currently, the framework only supports `scikit-learn` regressors for meta-models.

   Here is an example:
   ```yaml
    weighted:
       Simple: equal_weights  # Note that the key "Simple" is the ensembler's name! If you want to select_ensemblers, 
                               use this string!
       Inverse RMSE: inv_rmse_weights
       Inverse Variance: inv_variance_weights
       Inverse Error Covariance: inv_error_cov_weights
       ...
    meta:
       SVR:  # Note that this is the ensembler's name! If you want to select_ensemblers, use this string!
         model: SVR
         package: sklearn
         options:
           kernel: linear
       ...
   ```
   
   
4. **Configure desired performance metrics for evaluation:**

   In the `metrics.yml` file, select desired performance evaluation metrics by uncommenting them. Here is how the file 
   looks:
   ```yaml
    # MAE: mean_absolute_error
    # MSE: mean_squared_error
    # MdAE: median_absolute_error
    # MdSE: median_squared_error
    MAPE: mean_absolute_percentage_error
    RMSE: root_mean_squared_error
    ...
    ```
   

5. **Import Inputs and Functions:**

   Now import the necessary dependencies before running the pipeline:
   ```python
    from src.utils.paths import *
    from src.pipeline.run_pipeline import run_pipeline
    from src.models import MODELS
    from src.metrics import METRICS
   ```
   

6. **Run Pipeline:**
   
   Below is the main function for running the pipeline. All possible input arguments are listed in the DocString of 
   the next section. 

   Example usage:
   ```python
    output_dict = (
      run_pipeline(
        df=df, models=MODELS, metrics=METRICS,
        fh=1,
        # start="2020-01-01", end="2020-03-31",
        agg_method='last', agg_freq='B',
        target='bid_close', covariates=None,
        select_forecaster='all',
        select_ensemblers='all',
        export_path=PIPE_OUTPUT_DIR,
        verbose=True
      )
    )
    ```
   

7. **Display results:**

   View the results in the export directory (`/user/outputs/`) or access them directly in Python:
    ```python
     from src.paper import display_ranking_table
     display_ranking_table(output_dict['metrics ranking'])
    ```     


## :gear: Run Pipeline Function
This section describes the input arguments for the main function `run_pipeline()`.
```python
def run_pipeline(df, models, metrics,
                 fh=None,
                 start=None, end=None,
                 date_col='infer', date_format=None,
                 target='infer', covariates='infer', exclude=None,
                 agg_method=None, agg_freq=None,
                 select_forecasters='all', forecast_init_train=0.3,
                 autosarimax_refit_interval=0.15,
                 select_ensemblers='all', ensemble_init_train=0.25,
                 sort_by='MAPE',
                 export_path=None, errors='raise', verbose=False,
                 paper=False,
                 *args, **kwargs):
    """
    Run pipeline of data preprocessing, individual, and ensemble forecasting, and subsequent model ranking.

    Parameters:
    -----------
        df : pandas.DataFrame, pandas.Series or dict:
            Input DataFrame containing date, targets (and optionally covariates). Can also be a dictionary of
            dataframes with DataFrame names in the keys and DataFrames/Series in the corresponding value.
        models : dict
            Dictionary containing the forecasters and ensemblers models (approach, names, class names, package name,
            and options). This can be imported from the 'models' module of the project.
            Edit the '.yml' file to add/remove models.ch
        metrics : dict
            List of performance measures for model ranking in historical predictions.
            Can be imported from the 'metrics' module of the project. Edit '.yml' files to add/remove metrics.
        fh : int, optional
            When provided, pipeline not only performs historical evaluation of forecasters and ensemblers but also
            provides out-of-sample future predictions along the whole provided forecast horizon.
        start : str, optional
            Filter data to start from date string. Expects ISO DateTimeIndex format "YYYY-MMMM-DDD" (default: None).
        end : str, optional
            Filter data to end on date string. Expects ISO DateTimeIndex format "YYYY-MMMM-DDD" (default: None).
        date_col : str or int, optional
            Name or index of the date column in the input data (default: 'infer', searches for ISO formatted column).
        date_format : str, optional
            Custom format of the date column if date_col is specified (default: None, expects ISO format YYYY-MM-DD).
        target : str, int, optional
            Name or positional index of the target column in the input data
            (default: 'infer', takes first column after the date was set).
        covariates : str, int, or list, optional
            Names of covariates columns in the input data (default: 'infer', takes all columns after date and target
            are inferred.).
        exclude : str, int, or list, optional
            List of columns (string or positional index) to exclude from the input data (default: None).
        agg_method : str, optional
            Aggregation method for preprocessing.
            One of the pandas methods 'first', 'last', 'min', 'max', and 'mean' (default: None).
        agg_freq : str, optional
            DateTimeIndex aggregation frequency for preprocessing (default: None).
        select_forecasters : str or list, optional
            Specify which forecaster classes to use (default: 'all').
        forecast_init_train : float, optional
            Initial forecasters' training set size as a fraction of preprocessed data (default: 0.3, results in a
            30%/80% train-test split of the data).
        autosarimax_refit_interval : float, optional
            Refit interval for AutoSARIMA model (default: 0.33, corresponds to fitting at 0%, 33%, and 66% of ensemblers
            training set).
        select_ensemblers : str or list, optional
            Specify which ensemblers to use (default: 'all').
        ensemble_init_train : float, optional
            Initial ensemblers' training set size as a fraction of individual forecasters' predictions (default: 0.25,
            results in a 25%/75% train-test split of the data).
        sort_by : str, optional
            Performance measure to sort by for model ranking (default: 'MAPE').
        export_path : str or os.PathLike, optional
            Exports results to provided path
        errors : str, optional
            How to handle errors (default: 'raise').
        verbose : bool, optional
            If True, prints progress, intermediate results and steps console and log file (default: False).
        *args:
            Additional positional arguments.
        **kwargs:
            Additional keyword arguments.

    Returns:
    --------
    dict: Dictionary containing the following keys as pandas Series or DataFrames:
        - 'target and covariates': Tuple of preprocessed target and covariates.
        - 'historical_individual_predictions': Individual forecasters' predictions.
        - 'full predictions': Full ensemble predictions.
        - 'metrics ranking': Rankings based on specified metrics.
    If input is a dictionary, then the results will be a dictionary of dictionaries for each DataFrame in the input
    dictionary.
    """
    ... 
    return output
```

## :loveletter: Contact
If you encounter any issues or have feedback, please don't hesitate to reach out to [Paul 
Jarschke](https://www.linkedin.com/in/paul-jarschke-b4439b1a3/) and [Leon Löppert](https://www.linkedin.com/in/leonloeppert/) via LinkedIn.