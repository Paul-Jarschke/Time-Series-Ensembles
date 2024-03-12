import os
import sys

print("Loading paths...")

# Package root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Paper directory
PAPER_DIR = os.path.join(ROOT_DIR, "paper")
PAPERDATA_DIR = os.path.join(PAPER_DIR, "analyzed_data")
PLOT_DIR = os.path.join(PAPER_DIR, "plots")
PLOTSIM_DIR = os.path.join(PLOT_DIR, "simulations")
TABLE_DIR = os.path.join(PAPER_DIR, "tables")
PAPER_PIPE_DIR = os.path.join(PAPER_DIR, "pipeline")
PAPER_PIPE_INPUT_DIR = os.path.join(PAPER_PIPE_DIR, "inputs")
PAPER_PIPE_OUTPUT_DIR = os.path.join(PAPER_PIPE_DIR, "outputs")

# Data directory and subdirectories
DATA_DIR = os.path.join(ROOT_DIR, "data")
SIMDATA_DIR = os.path.join(DATA_DIR, "simulated")
TESTDATA_DIR = os.path.join(DATA_DIR, "test_data")
EUR_USD_DIR = os.path.join(DATA_DIR, "eurusd")

# User directory and subdirectories
USER_DIR = os.path.join(ROOT_DIR, "user")
PIPE_INPUT_DIR = os.path.join(USER_DIR, "inputs")
PIPE_OUTPUT_DIR = os.path.join(USER_DIR, "outputs")

# Source code directory and subdirectories
SRC_DIR = os.path.join(ROOT_DIR, "src")
MODELS_DIR = os.path.join(SRC_DIR, "models")
METRICS_DIR = os.path.join(SRC_DIR, "metrics")

sys.path.append(ROOT_DIR)
