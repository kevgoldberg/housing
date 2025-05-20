"""Configuration constants for the housing package."""

# Data paths
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

# Output directories
VIS_DIR = "visualizations"

# Random seed
SEED = 42

# Hyperparameter grid for Ridge regression
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
