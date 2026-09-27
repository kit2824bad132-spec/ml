"""
Amazon ML Challenge 2026 - Production LightGBM Model Training Pipeline
Trains high-precision LGBMClassifier on full dataset using:
- 11 Disambiguation features (RapidFuzz + legal suffix stripping + address number matching)
- Multi-category hard-negative mining (branch confusion, prefix-6, first-token)
- Bounded blocking candidates on 15,000 holdout validation entities
- Saves trained model to output/model_lightgbm.pkl and dataset/train/clean_validation_model_improved.pkl
"""

import os
import sys

# Ensure src directory is in path
sys.path.append(os.path.dirname(__file__))

from clean_validation_train_evaluate import (
    FEATURE_COLUMNS,
    MODEL_PARAMETERS,
    OUTPUT_MODEL_FILE as MODEL_FILE,
    main,
)

if __name__ == "__main__":
    print("=" * 80)
    print("STARTING LIGHTGBM MODEL TRAINING PIPELINE")
    print("=" * 80)
    main()