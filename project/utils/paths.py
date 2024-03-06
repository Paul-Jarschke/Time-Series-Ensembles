import os

print("Loading paths...")

# Setting up relevant paths
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
EXPORT_DIR = os.path.join(ROOT_DIR, "results")
SIMDATA_DIR = os.path.join(DATA_DIR, "simulated")
TESTDATA_DIR = os.path.join(DATA_DIR, "testing")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
METRICS_DIR = os.path.join(ROOT_DIR, "metrics")