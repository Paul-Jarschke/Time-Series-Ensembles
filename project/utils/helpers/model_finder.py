import darts.models
import sklearn.utils
import sktime.registry

from utils.helpers.FunctionFinder import FunctionFinder
from utils.paths import *


def model_finder(model_function, package_name=''):
    """
    Finds and returns the function or constructor object corresponding to the provided model function name 
    within specified packages or local project libraries.

    Args:
        model_function (str):           The name of the model function to be found.
        package_name (str, optional):   The name of the package where the model function is expected to be found. 
                                        Defaults to an empty string, indicating search in the local project libraries.

    Returns:
        object: The function or constructor object corresponding to the provided model function name.

    Raises:
        ValueError: If the model function is not found either in the specified package or locally, 
                    or if the specified package is not supported.

    Note:
        - For 'sktime' models, the function utilizes sktime's registry to find the model.
        - For 'darts' models, the function searches within the darts library.
        - For 'sklearn' models, the function utilizes sklearn's utility to find the model.
        - For unnamed packages, it assumes search within the local project's custom models directory.
        - If the model function is not found in any specified packages or locally, 
          it raises a ValueError with appropriate error message.
    """
    # Define supported packages
    supported_packages = ['sktime', 'darts', 'sklearn']

    # Define exception when model was not found in package:
    def not_found_error():
        """Raises a ValueError when the model function is not found."""
        error_string = (f"Model \'{model_function}\' not found in package" +
                        (f" \'{package_name}\'" if package_name != '' else ''))
        raise ValueError(error_string)

    # Returns the function/constructor object from a string and package information
    # For sktime models
    if package_name == 'sktime':
        # Retrieve all sktime estimators
        models_dict = dict(sktime.registry.all_estimators())
        if model_function in models_dict:
            model_function = models_dict[model_function]
            return model_function
        else:
            not_found_error()

    # For darts models
    elif package_name == 'darts':
        # Retrieve all darts model names
        models_dict = dir(darts.models)
        if model_function in models_dict:
            model_function = getattr(darts.models, model_function)
        else:
            not_found_error()


    # For sklearn models
    elif package_name == 'sklearn':
        # Retrieve all sklearn estimators
        models_dict = dict(sklearn.utils.all_estimators())
        if model_function in models_dict:
            model_function = models_dict[model_function]
        else:
            not_found_error()

    # For unnamed packages (assuming project custom weighting schemes here for now)
    elif package_name == '':
        # Use the FunctionFinder to find the model function in this project's library
        Finder = FunctionFinder()
        Finder.find_functions(directory=MODELS_DIR)
        model_function = Finder.functions[model_function]

    # Raise ValueError if model is from none of these packages:
    else:
        raise ValueError(f'Model {model_function} not found locally or in supported packages: '
                         f'{", ".join(str(e) for e in supported_packages)}')

    return model_function

