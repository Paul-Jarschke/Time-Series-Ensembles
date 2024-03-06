import inspect
import os

import pandas as pd

from utils.helpers.console_outputs import vprint


def csv_reader(PATH, file_name, date_col=0, columns='all', *args, **kwargs):
    """
        Read a CSV file from the specified directory path and return it as a pandas DataFrame.

        Parameters:
        - PATH (str): The directory path where the CSV file is located.
        - file_name (str): The name of the CSV file.
        - date_col (int, optional): The column index to be used as the index for the DataFrame. Default is first column.
        - columns (list of int/str, optional): Subset of columns to select, denoted either \
        by column labels or column indices stored in a list-like object.
        - *args: Additional positional arguments to be passed to pandas.read_csv().
        - **kwargs: Additional keyword arguments to be passed to pandas.read_csv().

        Returns:
        - df (pandas DataFrame): The DataFrame containing the data from the CSV file.
        """

    # Remove '.csv' from file_name
    if file_name.endswith('.csv'):
        file_name = file_name.replace('csv', '')

    # Combine the directory path and file name
    FILE = os.path.join(PATH, file_name + '.csv')

    # Read data, set time index end select columns
    columns = None if columns == 'all' else columns
    df = pd.read_csv(FILE, index_col=date_col, usecols=columns, *args, **kwargs)

    # Pass file name (without '.csv') as a flag to DataFrame
    df.attrs = {'file_name': file_name}

    return df


def csv_exporter(export_path,  *args, file_name=None):
    """
        Export pandas DataFrames to CSV files.

        This function exports DataFrames to CSV files. The export_path specifies the directory where
        the CSV files will be saved. Each DataFrame is saved as a separate CSV file with the name
        corresponding to the variable name of the DataFrame.

        Parameters:
            export_path (str or os.PathLike): The directory path where the CSV files will be saved.
            file_name (str with '.csv' ending, optional): Export file name. If not defined, infers it from object name.
            *args: Variable-length argument list of DataFrames to export.

        Notes:
            This function relies on the 'verbose' variable being accessible from the calling scope.

        Raises:
            KeyError: If the 'verbose' variable is not found in the calling scope.

        Example:
            csv_exporter("/path/to/export", df1, df2)
            # This will export df1 and df2 as CSV files to the "/path/to/export" directory.
        """
    parent_objects = inspect.currentframe().f_back.f_locals
    try:
        verbose = parent_objects['verbose']
    except KeyError:
        raise KeyError("'verbose' variable not found in the calling scope. Make sure it's defined.")

    if isinstance(export_path, (os.PathLike, str)):
        for df in args:
            if isinstance(file_name, str):
                df.to_csv(os.path.join(export_path, f"{file_name}"), index=True)
                vprint(f"Exporting DataFrame as csv...")
            for par_obj_name, par_obj in parent_objects.items():
                if par_obj is df:
                    df.to_csv(os.path.join(export_path, f"{par_obj_name}.csv"), index=True)
                    vprint(f"Exporting {par_obj_name.replace('_', ' ')} as csv...")
        vprint("...finished!\n")