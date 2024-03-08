import os
import sys

print("Loading paths...")

# Package root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Paper directory
PAPER_DIR = os.path.join(ROOT_DIR, "paper")
ANALYSIS_DIR = os.path.join(PAPER_DIR, "analysis")
PAPERDATA_DIR = os.path.join(PAPER_DIR, "current_data")
PLOT_DIR = os.path.join(ANALYSIS_DIR, "plots")

# Data directory and subdirectories
DATA_DIR = os.path.join(ROOT_DIR, "data")
SIMDATA_DIR = os.path.join(DATA_DIR, "simulation")
TESTDATA_DIR = os.path.join(DATA_DIR, "test_data")

# User directory and subdirectories
USER_DIR = os.path.join(ROOT_DIR, "user")
EXPORT_DIR = os.path.join(USER_DIR, "results")

# Source code directory and subdirectories
SRC_DIR = os.path.join(ROOT_DIR, "src")
MODELS_DIR = os.path.join(SRC_DIR, "models")
METRICS_DIR = os.path.join(SRC_DIR, "metrics")

sys.path.append(ROOT_DIR)
