import yaml

from src.utils.helpers.model_finder import model_finder
from src.utils.paths import *

stdout = sys.stdout

print("Loading models...")

# Open forecasters' specification from yaml files
with open(os.path.join(INPUT_DIR, "forecasters.yml"), 'r') as f:
    forecasters = yaml.safe_load(f)

with open(os.path.join(INPUT_DIR, "ensemblers.yml"), 'r') as f:
    ensemblers = yaml.safe_load(f)

# If desired, exclude forecasters or ensemblers (temporarily) by their name
EXCLUDED_MODELS = []

# Set up dictionary of models
MODELS = {
    'FORECASTERS': forecasters,
    'ENSEMBLERS': ensemblers
}

# Process MODELS dictionary
# The goal is to make a large model dictionary storing information
# about all the model types, approaches, functions, names, packages, and arguments
# In particular, replace model function strings with actual constructor objects from corresponding package
# Save results as tuple per model and then as list of tuples per model type


# Loop over model types (forecasters or ensemblers)
for model_type, models_dict in MODELS.items():
    sys.stdout = stdout
    # Loop over model approaches
    # for model_type == 'FORECASTERS': 'with_covariates' and 'without_covariates'
    # for model_type == 'ENSEMBLERS': 'weighted' and 'meta'
    for approach_name, approach_dict in models_dict.items():
        # Delete and skip approach when it is not properly defined/has no content
        if approach_dict in [None, {}]:
            continue
        # Loop over models in this approach
        for model_name, MODEL in approach_dict.items():

            # Delete model, when its name is excluded by user or when it is not properly defined/has no content
            if model_name in EXCLUDED_MODELS or MODEL in [None, {}]:
                continue

            # the model_functions of forecasting models and ensemble models are stored in a dictionary,
            # together with package and function argument information => extract this information
            if not isinstance(MODEL, dict):
                # if not a dictionary (e.g., weighting scheme), it is assumed that the function
                # is just in the values of the current approach dictionary items
                # Furthermore, no information on package_name and options is required
                MODEL = {
                    'model': MODEL,
                    'package': '',
                    'options': {},
                }
            # Currently model function is still stored as string => find the corresponding function/constructor
            MODEL['model'] = model_finder(package_name=MODEL['package'], model_function=MODEL['model'])
            models_dict[approach_name][model_name] = MODEL

        MODELS[model_type] = models_dict





