import pandas as pd


def identify_date_column(df, date_format):
    # identifies where the date is stored (index or column), transforms it to timestamp,
    # infers frequency, set it as pandas index
    # input is pandas df
    for column_name, column_content in df.items():
        # Search in columns
        if column_content.dtype == 'object':
            try:
                pd.to_datetime(column_content, format=date_format)
                identified_location = column_name
                return identified_location
            except ValueError:
                pass
        # Look in index
        try:
            pd.to_datetime(df.index, format=date_format)
            identified_location = 'index'
            return identified_location
        # Raise error if date could neither be found in index nor in columns
        except ValueError:
            raise ValueError(
                'Date information can not be inferred. ',
                'Please specify name or position using \'date_col\' argument.'
            )


def target_covariate_split(df, target='infer', covariates='infer', exclude=None):
    """
    Separates a pandas DataFrame into target and covariates.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target (int or str, optional): The column index or name of the target variable. Infers first column as target
        by default.
        covariates (None, int, str, or list of int and/or str, optional): The column indices or names of the covariates.
        Infers all existing columns after the first column as covariates by default.
        exclude (int, str, or list of int and/or str, optional): Indices or names of the columns to be excluded.

    Returns:
        tuple: A tuple containing the target variable DataFrame and the covariates DataFrame.
    """
    # Exclude columns if specified
    if exclude is not None:
        # if not isinstance(exclude, list):
        #     exclude = [exclude]
        # elif isinstance(exclude, list):
        #     exclude = [var if isinstance(var, str) else df.index[var] for var in exclude]
        # else:
        #     raise ValueError('Excluded columns must be provided either as int, str, or list of int/str.')
        # it is already ensured in pipeline that exclude is of correct for dropping (str or list of str)
        df = df.drop(exclude, axis=1)

    # Infer covariates if option is chosen
    if target == 'infer':
        target = df.columns[0]
    # elif isinstance(target, int):
    #     target = df.iloc[:, target].copy()
    # it is already ensured in pipeline that target is of correct type (str)
    target = df[target].copy()

    if covariates == 'infer':
        n_covariates = len(df.columns) - 1  # reduce by 1 because of target
        # Infer covariates if option is chosen
        if n_covariates > 0:
            covariates = list(range(1, n_covariates + 1))
            covariates = df.columns[covariates]
        else:
            covariates = None
        # Select covariates based on input
        # if isinstance(covariates, int):
        #    covariates = df.iloc[:, covariates].copy()
    # it is already ensured in pipeline that covariates is of correct type (str or list of str)
    if covariates is not None:
        covariates = df[covariates].copy()
        # elif isinstance(covariates, list):
        # covariates = [cov if isinstance(cov, int) else df.columns.get_loc(cov) for cov in covariates]
        # else:
        #     raise ValueError("Provided covariates must be provided either as int, str, or list of int/str.")

    return target, covariates


def aggregate_data(data, agg_freq: str, date_col='index', columns='all', method='mean', drop_nan=True):
    """
    Aggregate data to specified frequency by specified method.

    Parameters:
    - data (pandas DataFrame or Series): DataFrame containing data to aggregate.
    - date_col (str, optional): Name of the date column. Needs to be specified only if date is not yet in the index.
    - columns (str or list of str, optional): Name of the aggregated columns to be returned by the function.\
    Takes all columns by default.
    - agg_freq (str, optional): Desired aggregation frequency. (E.g., 'B', 'D', 'M', 'Y', ...)
    - method (str, optional): Pandas aggregation method (E.g., 'last', 'mean', 'min', 'max'). Defaults to 'mean'.
    - drop_nan (bool, optional): Whether to drop NaN values after aggregation. Defaults to True.

    Returns:
    - aggregated_df (DataFrame): Aggregated DataFrame.
    """

    # Input validation
    assert isinstance(data, (pd.DataFrame, pd.Series)), 'Data must be a pandas DataFrame or Series.'
    assert isinstance(agg_freq, str), 'Aggregation frequency must be provided as str.'
    assert method in ['last', 'mean', 'min', 'max'], 'Aggregation method must be one of ["last", "mean", "min", "max"].'

    # Ensure that index contains dates
    if date_col != 'index':
        data.set_index(date_col, inplace=True)

    # Check if pandas object has valid time index (DatetimeIndex, PeriodIndex, TimedeltaIndex), if not transfer index
    if not isinstance(data.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        data.index = pd.DatetimeIndex(data.index, freq='infer')

    # Take subset
    if columns != 'all':
        data = data[columns]

    # Aggregate based on method
    aggregated_df = data.resample(agg_freq).apply(method)

    # Optionally drop rows with NaN values
    if drop_nan:
        aggregated_df.dropna(inplace=True)

    return aggregated_df
