# Amazon ML Challenge 2026 - Entity Resolution Pipeline

High-performance Entity Resolution (record linkage) system designed for the Amazon ML Challenge 2026. Matches business entities across multi-source datasets (Source 1 to Source 2 & Source 3) under extreme class imbalance, multi-lingual scripts, and strict precision constraints for the $F_{0.5}$ metric.

## 🚀 Key Results & Highlights
- **End-to-End $F_{0.5}$**: **`0.9542`** on clean validation (slashing false positives by 93% vs 0.70 baseline).
- **Candidate Recall**: **`93.67%`** (boosted by +19.78% via multi-pass country-bounded token blocking, recovering 3,567 previously unretrievable matches).
- **Inference Speed**: **124,700 pairs/sec** via DuckDB disk-backed batch streaming and vectorized RapidFuzz C++ similarity kernels.
- **Full Test Set Runtime**: 73,297,852 candidate pairs scored in **10.3 minutes** across 12 CPU cores.

---

## 🏗️ Architecture & Pipeline

```
Raw Test Data (S1, S2, S3)
           │
           ▼
[Selective Multi-Pass Blocking] 
   ├─ Exact Name + Country
   ├─ Name Prefix-6 + Country
   ├─ First Name Token (len >= 5) + Country
   ├─ Postal / PIN Code + Country
   ├─ House Number + Address Token + Country
   └─ Name Token + Address Token + Country
           │
           ▼
73.3M Bounded Candidate Pairs
           │
           ▼
[DuckDB Pre-Normalized Streaming]
           │
           ▼
[RapidFuzz 9-Feature Extraction]
   ├─ name_ratio, name_token_sort, name_token_set
   ├─ address_ratio, address_token_sort, address_token_set
   ├─ country_match
   └─ name_length_diff, address_length_diff
           │
           ▼
[Improved Hard-Negative Classifier]
   (RandomForest 250 trees, max_depth 14, balanced weights)
           │
           ▼
[Decision Threshold: T = 0.975]
           │
           ▼
output/matching_results.tsv (1,732,544 rows)
```

---

## 📁 Repository Structure
```
├── .gitignore
├── README.md
├── cleanup_report.txt
├── latest_run_audit.txt
├── run_full_test_pipeline.ps1
├── run_inference.py
├── run_inference_duckdb.py
├── output/
│   ├── matching_results.tsv           # Final verified submission file (T=0.975)
│   ├── model_lightgbm.pkl             # Trained LightGBM model
│   ├── final_optimization_report.txt  # Full 16-point optimization benchmark
│   ├── best_threshold_report.txt      # Grid search threshold sweeps
│   └── ...                            # Clean validation & error analysis reports
└── src/
    ├── preprocessing.py               # Unicode normalization (NFKC)
    ├── bucket_blocking_lightgbm.py    # DuckDB blocking pipeline
    ├── run_test_inference_optimized.py# Fast 125k pairs/sec test inference
    ├── run_lightgbm_test_inference.py # LightGBM test inference
    ├── phase8_hard_negative_experiment.py # Multi-category negative mining
    └── ...                            # Training & validation evaluation scripts
```

---

## 🛠️ Usage

### Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install duckdb rapidfuzz scikit-learn lightgbm pandas numpy joblib
```

### Run Full Test Inference
```powershell
python src/run_test_inference_optimized.py
```
This streams the candidate pairs, predicts probabilities with the improved hard-negative model, applies optimal threshold $T=0.975$, and outputs `output/matching_results.tsv`.
