import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, fbeta_score

MODEL_FILE = "dataset/train/clean_validation_model_improved.pkl"
FEATURES_FILE = "dataset/train/clean_validation_validation_features.tsv"
SPLIT_FILE = "dataset/train/clean_validation_split.tsv"
OUTPUT_CSV = "output/threshold_experiment.csv"
OUTPUT_REPORT = "output/best_threshold_report.txt"

TOTAL_GROUND_TRUTH_MATCHES = 18033
TOTAL_VALIDATION_SOURCE1 = 10000

FEATURE_COLUMNS = [
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "address_ratio",
    "address_token_sort",
    "address_token_set",
    "country_match",
    "name_length_diff",
    "address_length_diff",
]

def main():
    print("Loading model from:", MODEL_FILE)
    model_data = joblib.load(MODEL_FILE)
    model = model_data["model"]
    current_threshold = model_data.get("threshold", 0.70)
    print(f"Model loaded. Current baseline threshold: {current_threshold}")

    print("Loading validation features from:", FEATURES_FILE)
    df = pd.read_csv(FEATURES_FILE, sep="\t", dtype={"source1_entity_id": str, "candidate_entity_id": str})
    print(f"Loaded {len(df):,} candidate pairs")

    y_true = df["label"].astype(int).to_numpy()
    features = df[FEATURE_COLUMNS]

    print("Computing predicted probabilities...")
    probs = model.predict_proba(features)[:, 1]

    # Coarse thresholds + fine sweep
    coarse_range = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
    fine_range = [round(t, 2) for t in np.arange(0.70, 0.99, 0.01)]
    thresholds = sorted(list(set(coarse_range + fine_range)))

    results = []
    
    for t in thresholds:
        preds = (probs >= t).astype(np.int8)
        tp = int(np.sum((y_true == 1) & (preds == 1)))
        fp = int(np.sum((y_true == 0) & (preds == 1)))
        fn_cand = int(np.sum((y_true == 1) & (preds == 0)))
        tn = int(np.sum((y_true == 0) & (preds == 0)))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec_cand = float(tp / (tp + fn_cand)) if (tp + fn_cand) > 0 else 0.0
        
        # F0.5 = (1.25 * prec * rec) / (0.25 * prec + rec)
        denom_f05_cand = 0.25 * prec + rec_cand
        f05_cand = float((1.25 * prec * rec_cand) / denom_f05_cand) if denom_f05_cand > 0 else 0.0

        # End-to-end recall (denominator = TOTAL_GROUND_TRUTH_MATCHES)
        e2e_rec = float(tp / TOTAL_GROUND_TRUTH_MATCHES)
        denom_f05_e2e = 0.25 * prec + e2e_rec
        f05_e2e = float((1.25 * prec * e2e_rec) / denom_f05_e2e) if denom_f05_e2e > 0 else 0.0

        # Count validation Source-1 entities with 0 predicted matches
        matched_s1 = set(df.loc[preds == 1, "source1_entity_id"])
        no_match_count = TOTAL_VALIDATION_SOURCE1 - len(matched_s1)

        results.append({
            "threshold": t,
            "precision": round(prec, 6),
            "recall": round(rec_cand, 6),
            "F0.5": round(f05_cand, 6),
            "end_to_end_recall": round(e2e_rec, 6),
            "end_to_end_F0.5": round(f05_e2e, 6),
            "predicted_matches": int(tp + fp),
            "no_match_count": no_match_count,
            "false_positives": fp,
            "false_negatives": fn_cand,
            "true_positives": tp,
            "true_negatives": tn
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Threshold sweep results saved to: {OUTPUT_CSV}")

    # Identify best thresholds
    best_cand_row = results_df.loc[results_df["F0.5"].idxmax()]
    best_e2e_row = results_df.loc[results_df["end_to_end_F0.5"].idxmax()]
    baseline_row = results_df.loc[results_df["threshold"] == 0.70].iloc[0]

    report = f"""================================================================================
AMAZON ML CHALLENGE 2026 - THRESHOLD OPTIMIZATION REPORT
================================================================================
Model: {MODEL_FILE}
Evaluation Dataset: Frozen Clean Validation Set (396,850 candidate pairs)
Baseline Threshold: 0.70
Target Metric: F0.5 (Precision-heavy: beta = 0.5)

BASELINE PERFORMANCE (@ Threshold = 0.70):
- Threshold: 0.70
- Precision: {baseline_row['precision']:.6f} ({baseline_row['precision']*100:.2f}%)
- Candidate-level Recall: {baseline_row['recall']:.6f} ({baseline_row['recall']*100:.2f}%)
- Candidate-level F0.5: {baseline_row['F0.5']:.6f}
- End-to-End Recall: {baseline_row['end_to_end_recall']:.6f} ({baseline_row['end_to_end_recall']*100:.2f}%)
- End-to-End F0.5: {baseline_row['end_to_end_F0.5']:.6f}
- False Positives: {baseline_row['false_positives']:,}
- False Negatives (within candidates): {baseline_row['false_negatives']:,}
- True Positives: {baseline_row['true_positives']:,}
- Total Predicted Matches: {baseline_row['predicted_matches']:,}
- Unmatched S1 Count: {baseline_row['no_match_count']:,}

OPTIMAL THRESHOLD (MAXIMIZING CANDIDATE-LEVEL F0.5):
- Best Threshold: {best_cand_row['threshold']:.2f}
- Precision: {best_cand_row['precision']:.6f} ({best_cand_row['precision']*100:.2f}%)
- Candidate-level Recall: {best_cand_row['recall']:.6f} ({best_cand_row['recall']*100:.2f}%)
- Candidate-level F0.5: {best_cand_row['F0.5']:.6f}
- F0.5 Improvement vs Baseline: {best_cand_row['F0.5'] - baseline_row['F0.5']:+.6f}
- False Positives: {best_cand_row['false_positives']:,} (reduction of {baseline_row['false_positives'] - best_cand_row['false_positives']:,})
- False Negatives: {best_cand_row['false_negatives']:,}
- Total Predicted Matches: {best_cand_row['predicted_matches']:,}
- Unmatched S1 Count: {best_cand_row['no_match_count']:,}

OPTIMAL THRESHOLD (MAXIMIZING END-TO-END F0.5):
- Best Threshold: {best_e2e_row['threshold']:.2f}
- Precision: {best_e2e_row['precision']:.6f} ({best_e2e_row['precision']*100:.2f}%)
- End-to-End Recall: {best_e2e_row['end_to_end_recall']:.6f} ({best_e2e_row['end_to_end_recall']*100:.2f}%)
- End-to-End F0.5: {best_e2e_row['end_to_end_F0.5']:.6f}
- End-to-End F0.5 Improvement: {best_e2e_row['end_to_end_F0.5'] - baseline_row['end_to_end_F0.5']:+.6f}
- False Positives: {best_e2e_row['false_positives']:,}
- False Negatives: {best_e2e_row['false_negatives']:,}

FULL THRESHOLD COMPARISON TABLE:
{results_df[['threshold', 'precision', 'recall', 'F0.5', 'end_to_end_F0.5', 'false_positives', 'predicted_matches']].to_string(index=False)}

DECISION:
- If threshold is raised to {best_cand_row['threshold']:.2f}:
  Precision jumps dramatically from {baseline_row['precision']*100:.2f}% to {best_cand_row['precision']*100:.2f}%,
  slashing False Positives from {baseline_row['false_positives']:,} down to {best_cand_row['false_positives']:,},
  which elevates candidate-level F0.5 from {baseline_row['F0.5']:.6f} to {best_cand_row['F0.5']:.6f} ({best_cand_row['F0.5'] - baseline_row['F0.5']:+.6f}).
================================================================================
"""
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to: {OUTPUT_REPORT}")
    print("\n" + report[:1200] + "...")

if __name__ == "__main__":
    main()
