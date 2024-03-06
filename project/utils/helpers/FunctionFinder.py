import importlib.util

from utils.paths import *


class FunctionFinder:
    def __init__(self):
        self.functions = {}  # Dictionary to store functions

    def find_functions(self, directory=None):
        if directory is None:
            directory = os.getcwd()  # Get the current working directory
        # Iterate over files in the current directory and save functions found in dictionary
        functions = {}
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.endswith('.py') and "__init__" not in filename:  # Check if the file is a Python file
                    filename_no_py = filename[:-3]  # Remove the .py extension to get the module name

                    module_name = (root.replace(ROOT_DIR, "")[1:].replace("\\", ".").replace("\/", ".") +
                                   "." + filename_no_py)
                    module = importlib.import_module(module_name)  # Import the module
                    # Iterate over objects in the module
                    for name in dir(module):
                        obj = getattr(module, name)
                        # Check if the object is callable (function or class)
                        if callable(obj):
                            functions[name] = obj  # Add the callable object to the dictionary
        self.functions = functions
