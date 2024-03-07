# Import FunctionFinder and METRICS_DIR and external dependencies
import os

import sktime.performance_metrics.forecasting._functions  # Import custom metrics
import yaml

from src.utils.helpers.FunctionFinder import FunctionFinder
from src.utils.paths import USER_DIR, METRICS_DIR

# Print statement indicating loading of metric functions
print('Loading metrics...')

# Read in yml file
with open(os.path.join(USER_DIR, "metrics.yml"), 'r') as f:
    METRICS = yaml.safe_load(f)


# Find custom implemented metrics
Finder = FunctionFinder()
Finder.find_functions(METRICS_DIR)  # Load functions from .py files in the current directory


# If desired, exclude metrics (temporarily) manually
EXCLUDED_METRICS = []
for element in EXCLUDED_METRICS:
    METRICS.pop(element)

# Replace METRICS function name with actual constructor object
sktime_functions = dir(sktime.performance_metrics.forecasting._functions)
for metric_name, metric_function in METRICS.items():
    if metric_function in sktime_functions:
        METRICS[metric_name] = getattr(sktime.performance_metrics.forecasting._functions, metric_function)
    else:
        METRICS[metric_name] = Finder.functions[metric_function]

# Note:
# You can simply add new metrics as .py file, and they will be available to the pipeline
# Generic format:
# Input arguments must be y_actual and y_predicted (in this order)
# Output must be a single value
