import darts.models
import sklearn.utils
import sktime.registry

from utils.helpers.FunctionFinder import FunctionFinder
from utils.paths import *


def model_finder(model_function, package_name=''):

    # Define supported packages
    supported_packages = ['sktime', 'darts', 'sklearn']

    # Define exception when model was not found in package:
    def not_found_error():
        error_string = (f"Model \'{model_function}\' not found in package" +
                        (f" \'{package_name}\'" if package_name != '' else ''))
        raise ValueError(error_string)

    # Returns the function/constructor object from a string and package information
    # For sktime models
    if package_name == 'sktime':
        models_dict = dict(sktime.registry.all_estimators())
        if model_function in models_dict:
            model_function = models_dict[model_function]
            return model_function
        else:
            not_found_error()

    # For darts models
    elif package_name == 'darts':
        models_dict = dir(darts.models)
        if model_function in models_dict:
            model_function = getattr(darts.models, model_function)
        else:
            not_found_error()


    # For sklearn models
    elif package_name == 'sklearn':
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

    # If model is from none of these packages:
    else:
        raise ValueError(f'Model {model_function} not found locally or in supported packages: '
                         f'{", ".join(str(e) for e in supported_packages)}')

    return model_function

