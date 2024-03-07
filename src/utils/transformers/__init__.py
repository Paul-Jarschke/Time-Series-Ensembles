# Import functions from individual files
from .transform_darts import *
from .transform_sktime import *

# Print statement indicating loading of metric functions
print("Loading data transformers...")

TRANSFORMERS = {
    "sktime": transform_to_sktime,
    "sktime.covariates": transform_sktime_lagged_covariates,
    "darts": transform_to_darts,
}
