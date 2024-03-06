def prepare_eurusd_data(data, n=1000):
    """
    Prepare EURUSD data for time series analysis.

    Args:
        data (pd.DataFrame): DataFrame containing time series data.
        n (int, optional): Number of rows to select from the beginning of the DataFrame. Defaults to 1000.

    Returns:
        Tuple[darts.timeseries.TimeSeries, darts.timeseries.TimeSeries, darts.timeseries.TimeSeries]:
            A tuple containing the prepared time series data:
            - series: Time series object representing the entire dataset.
            - target_series: Time series object representing the target variable (bid_open).
            - cov_series: Time series object representing the covariate variables.

    Note:
        This function prepares the input DataFrame by selecting the first 'n' rows,
        removing the 'datetime' column, and separating the target variable (bid_open)
        and covariates for further analysis.
    """
    # Select the first n rows
    df = data.head(n)

    # Save datetime column
    dates = df['datetime']

    # Reset index to a range of integers
    df.reset_index(drop=True, inplace=True)
    df.drop('datetime', axis=1, inplace=True)

    # Extract target and covariate columns
    target_col = df['bid_open']
    covariate_cols = df[['bid_high', 'bid_low', 'bid_close', 'ask_open', 'ask_high', 'ask_low', 'ask_close']]

    # Create Darts TimeSeries objects
    series = TimeSeries.from_dataframe(df)
    target_series = TimeSeries.from_series(target_col)
    cov_series = TimeSeries.from_series(covariate_cols)

    return series, target_series, cov_series, dates