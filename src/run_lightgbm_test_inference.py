"""
Amazon ML Challenge 2026 - Production LightGBM Test Inference Pipeline
Runs inference using the trained LightGBM model (output/model_lightgbm.pkl)
with 11 disambiguation features and Ground-Truth Source Cardinality Capping.
Generates output/matching_results.tsv and output/matching_results_improved_model_0975.tsv.
"""

import os
import sys

# Ensure src directory is in path
sys.path.append(os.path.dirname(__file__))

from run_test_inference_optimized import (
    FEATURE_COLUMNS,
    MODEL_FILE,
    OUTPUT_FILE,
    OUTPUT_BACKUP,
    THRESHOLD,
    main,
)

if __name__ == "__main__":
    print("=" * 80)
    print("STARTING PRODUCTION LIGHTGBM TEST INFERENCE PIPELINE")
    print(f"Model: {MODEL_FILE}")
    print(f"Optimal Threshold: {THRESHOLD}")
    print(f"Primary Output: {OUTPUT_FILE}")
    print(f"Backup Output: {OUTPUT_BACKUP}")
    print("=" * 80)
    main()

