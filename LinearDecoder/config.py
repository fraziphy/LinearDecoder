# config.py

import os
from pathlib import Path

# Define the directory where data is stored
DATA_DIR = Path(os.environ.get('LINEAR_DECODER_DATA', '../data/'))

# Kernel parameters
TAU = 10  # Time constant in milliseconds (ms)

# L2 regularization (Ridge Regression)
LAMBDA_REG = 1e-3  # Regularization strength

# Stratified K-Fold Cross-Validation parameters
K_FOLD = 10  # Number of folds
CONFIDENCE_INTERVAL = 95  # Confidence interval percentage

# You can add a function to get all config variables
def get_config():
    return {
        'DATA_DIR': DATA_DIR,
        'TAU': TAU,
        'LAMBDA_REG': LAMBDA_REG,
        'K_FOLD': K_FOLD,
        'CONFIDENCE_INTERVAL': CONFIDENCE_INTERVAL
    }

if __name__ == "__main__":
    print("Current configuration:")
    for key, value in get_config().items():
        print(f"{key}: {value}")
