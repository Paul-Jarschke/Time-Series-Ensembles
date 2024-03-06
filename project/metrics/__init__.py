# Import FunctionFinder and METRICS_DIR
from utils.helpers import FunctionFinder
from utils.paths import METRICS_DIR

# Import metrics from individual files
from metrics.rmse import *
from metrics.smape import *
from metrics.mape import *

# Print statement indicating loading of metric functions
print('Loading metrics...')

# Find implemented metrics
loader = FunctionFinder()
loader.find_functions(METRICS_DIR)  # Load functions from .py files in the current directory

# If desired, exclude metrics (temporarily)
EXCLUDED_METRICS = []
for element in EXCLUDED_METRICS:
    loader.functions.pop(element)

# Set up metrics dictionary
METRICS = loader.functions

# Note:
# You can simply add new metrics as .py file, and they will be available to the pipeline

# Generic format:
# Metrics should take input arguments predictions, targets and return a single value in the end.

# Outlook:
# Implement metrics from sktime library

