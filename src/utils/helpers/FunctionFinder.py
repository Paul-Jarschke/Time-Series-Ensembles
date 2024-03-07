import importlib.util
import os
from src.utils.paths import *

class FunctionFinder:
    """A class to find and store functions from Python files in a given directory."""

    def __init__(self):
        """Initialize the FunctionFinder."""
        self.functions = {}  # Dictionary to store functions

    def find_functions(self, directory=None):
        """
        Find functions in Python files within a directory.

        Args:
            directory (str, optional):  The directory path to search for Python files.
                                        If not provided, the current working directory is used.

        Returns:
            None: The found functions are stored in the `functions` attribute of the object.

        """

        # Get the current working directory
        if directory is None:
            directory = os.getcwd()

        # Temporary dictionary to store functions found
        functions = {}

        # Iterate over files in the specified directory and subdirectories
        functions = {}
        for root, _, files in os.walk(directory):
            for filename in files:

                # Check if the file is a Python file
                if filename.endswith(".py") and "__init__" not in filename:

                    # Remove the .py extension to get the module name
                    filename_no_py = filename[:-3]

                    # Generate module name from file path
                    module_name = (
                        root.replace(ROOT_DIR, "")[1:]
                        .replace("\\", ".")
                        .replace("\/", ".")
                        + "."
                        + filename_no_py
                    )

                    # Import the module
                    module = importlib.import_module(module_name)

                    # Iterate over objects in the module
                    for name in dir(module):
                        obj = getattr(module, name)
                        # Check if the object is callable (function or class)
                        if callable(obj):
                            # Add the callable object to the dictionary
                            functions[name] = obj

        # Update the functions attribute with the found functions
        self.functions = functions
