import inspect
import os

import pandas as pd

from src.utils.helpers.console_outputs import vprint


def csv_reader(PATH, file_name, date_col=0, columns='all', *args, **kwargs): 
    """
    Read a CSV file from the specified directory path and return it as a pandas DataFrame.

    Parameters:
    - PATH (str):                           The directory path where the CSV file is located.
    - file_name (str):                      The name of the CSV file.
    - date_col (int, optional):             The index of the column to be used as the DataFrame index. Default is the first column.
    - columns (list of int/str, optional):  Subset of columns to select, denoted either by column labels or indices stored in a list-like object.

    - *args:    Additional positional arguments to be passed to pandas.read_csv().
    - **kwargs: Additional keyword arguments to be passed to pandas.read_csv().

    Returns:
    - df (pandas DataFrame): The DataFrame containing the data from the CSV file.
    """

    # Remove '.csv' from file_name if present
    if file_name.endswith('.csv'):
        file_name = file_name.replace('csv', '')

    # Combine the directory path and file name
    FILE = os.path.join(PATH, file_name + '.csv')

    # Read data, set time index end select columns
    columns = None if columns == 'all' else columns
    df = pd.read_csv(FILE, index_col=date_col, usecols=columns, *args, **kwargs)

    # Store the file name (without '.csv') as a flag in the DataFrame attributes
    df.attrs = {'file_name': file_name}

    return df


def csv_exporter(export_path,  *args, file_names=None):
    """
        Export pandas DataFrames to CSV files.

        This function exports DataFrames to CSV files. The export_path specifies the directory where
        the CSV files will be saved. Each DataFrame is saved as a separate CSV file with the name
        corresponding to the variable name of the DataFrame.

        Parameters:
            export_path (str or os.PathLike):           The directory path where the CSV files will be saved.
            file_names (str or list of str, optional):  Export file name with '.csv' ending. If not defined,
                                                        infers it from object name.

            *args: Variable-length argument list of DataFrames to export.

        Notes:
            This function relies on the 'verbose' variable being accessible from the calling scope.

        Raises:
            KeyError: If the 'verbose' variable is not found in the calling scope.

        Example:
            csv_exporter("/path/to/export", df1, df2)
            his will export df1 and df2 as CSV files called "df1.csv" and "df2.csv" to the "/path/to/export"
            directory.
        """
    
    # Accessing variables from the calling scope
    parent_objects = inspect.currentframe().f_back.f_locals

    # Check if the 'verbose' variable is defined in the calling scope
    try:
        verbose = parent_objects['verbose']
    except KeyError:
        raise KeyError("'verbose' variable not found in the calling scope. Make sure it's defined.")

    # Export each DataFrame to a CSV file
    if isinstance(export_path, (os.PathLike, str)):
        # Loop over dataframes
        for i, df in enumerate(args):
            if isinstance(df, (pd.DataFrame, pd.Series)): # Don't export NoneTypes

                # Create file_name string from input
                file_name = file_names

                # Special handling for lists and NoneTypes
                # Extract file name from list
                if isinstance(file_names, list):
                    file_name = file_names[i]
                # Infer file name from object name
                elif file_names is None:
                    for par_obj_name, par_obj in parent_objects.items():
                        if par_obj is df:
                            file_name = par_obj_name
                            vprint(f"\nExporting {file_name} as csv to {export_path}...")

                # Export df with extracted string file_name
                if isinstance(file_name, str):
                    df.to_csv(os.path.join(export_path, f"{file_name}.csv"), index=True)
                # Export with specified file_name
                else:
                    raise ValueError('Can not export DataFrame. Please provide filename!')

        # Print export statements
        if isinstance(file_names, str):
            vprint(f"\nExporting {file_names} as csv to {export_path}...")
        elif isinstance(file_names, list):
            vprint(f"\nExporting {' and '.join(file_names).replace('_', ' ')} as csv to {export_path}...")
        vprint("...finished!")
