import os

print("Loading paths...")
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
EXPORT_DIR = os.path.join(DATA_DIR, "results")
SIMDATA_DIR = os.path.join(DATA_DIR, "simulated")
TESTDATA_DIR = os.path.join(DATA_DIR, "testing")