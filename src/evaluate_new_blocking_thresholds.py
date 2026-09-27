import os
import joblib
import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.fuzz import ratio, token_sort_ratio, token_set_ratio

from preprocessing import normalize_country, normalize_text

MODEL_FILE = "dataset/train/clean_validation_model_improved.pkl"
CANDIDATE_FILE = "dataset/train/clean_blocking_candidates_selective.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"
SPLIT_FILE = "dataset/train/clean_validation_split.tsv"
S1_FILE = "dataset/train/train_source1_sample_50k.tsv"
S2_FILE = "dataset/train/train_source2_matches_50k.tsv"
S3_FILE = "dataset/train/train_source3_matches_50k.tsv"
OUTPUT_REPORT = "output/new_blocking_threshold_sweep.txt"

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

def safe_ratio_vec(a, b, scorer):
    scores = process.cpdist(a, b, scorer=scorer, workers=4, dtype=np.float32)
    mask = np.array([bool(x and y) for x, y in zip(a, b)])
    return np.where(mask, scores / 100.0, 0.0)

def main():
    print("Loading model from:", MODEL_FILE)
    model_data = joblib.load(MODEL_FILE)
    model = model_data["model"]

    print("Loading validation Source-1 IDs...")
    split_df = pd.read_csv(SPLIT_FILE, sep="\t", dtype=str)
    val_s1_ids = set(split_df.loc[split_df["split"] == "validation", "entity_id"])
    print(f"Validation S1 IDs: {len(val_s1_ids):,}")

    print("Loading ground truth for validation IDs...")
    val_truth_pairs = set()
    for chunk in pd.read_csv(GT_FILE, sep="\t", dtype=str, keep_default_na=False, chunksize=200000):
        sub = chunk[chunk.source1_entity_id.isin(val_s1_ids)]
        for row in sub.itertuples(index=False):
            for m in row.matched_entity_ids.split(","):
                m = m.strip()
                if m:
                    val_truth_pairs.add((row.source1_entity_id, m))
    print(f"Validation Ground Truth Matches: {len(val_truth_pairs):,}")

    print("Loading records for normalization...")
    s1_df = pd.read_csv(S1_FILE, sep="\t", dtype=str, keep_default_na=False)
    targets_df = pd.concat([
        pd.read_csv(S2_FILE, sep="\t", dtype=str, keep_default_na=False),
        pd.read_csv(S3_FILE, sep="\t", dtype=str, keep_default_na=False),
    ], ignore_index=True).drop_duplicates("entity_id")

    s1_records = {
        row.entity_id: (normalize_text(row.business_name), normalize_text(row.business_address), normalize_country(row.country))
        for row in s1_df.itertuples(index=False)
    }
    target_records = {
        row.entity_id: (normalize_text(row.business_name), normalize_text(row.business_address), normalize_country(row.country))
        for row in targets_df.itertuples(index=False)
    }

    print("Loading selective blocking candidate pairs from:", CANDIDATE_FILE)
    cand_df = pd.read_csv(CANDIDATE_FILE, sep="\t", dtype=str)
    print(f"Total selective candidate pairs: {len(cand_df):,}")

    cand_pairs_set = set(zip(cand_df.source1_entity_id, cand_df.candidate_entity_id))
    true_in_cands = len(val_truth_pairs & cand_pairs_set)
    cand_recall = true_in_cands / len(val_truth_pairs)
    print(f"Candidate Recall: {cand_recall:.6f} ({true_in_cands:,} / {len(val_truth_pairs):,})")

    # Compute features in chunks of 50,000
    print("Computing features and model predictions...")
    all_probs = []
    chunk_size = 50000
    for i in range(0, len(cand_df), chunk_size):
        chunk = cand_df.iloc[i:i+chunk_size]
        n1, a1, c1 = zip(*[s1_records[sid] for sid in chunk.source1_entity_id])
        n2, a2, c2 = zip(*[target_records[cid] for cid in chunk.candidate_entity_id])

        features = pd.DataFrame()
        features['name_ratio'] = safe_ratio_vec(n1, n2, ratio)
        features['name_token_sort'] = safe_ratio_vec(n1, n2, token_sort_ratio)
        features['name_token_set'] = safe_ratio_vec(n1, n2, token_set_ratio)
        features['address_ratio'] = safe_ratio_vec(a1, a2, ratio)
        features['address_token_sort'] = safe_ratio_vec(a1, a2, token_sort_ratio)
        features['address_token_set'] = safe_ratio_vec(a1, a2, token_set_ratio)
        features['country_match'] = [1 if x == y else 0 for x, y in zip(c1, c2)]
        features['name_length_diff'] = [abs(len(x) - len(y)) for x, y in zip(n1, n2)]
        features['address_length_diff'] = [abs(len(x) - len(y)) for x, y in zip(a1, a2)]

        probs = model.predict_proba(features[FEATURE_COLUMNS])[:, 1]
        all_probs.extend(probs)

    all_probs = np.array(all_probs)
    labels = np.array([1 if (s, c) in val_truth_pairs else 0 for s, c in zip(cand_df.source1_entity_id, cand_df.candidate_entity_id)])

    print("Evaluating thresholds on New Selective Blocking...")
    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]
    rows = []
    for t in thresholds:
        preds = (all_probs >= t).astype(np.int8)
        tp = int(np.sum((labels == 1) & (preds == 1)))
        fp = int(np.sum((labels == 0) & (preds == 1)))
        fn_cand = int(np.sum((labels == 1) & (preds == 0)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_cand = tp / (tp + fn_cand) if (tp + fn_cand) > 0 else 0.0
        denom_f05_cand = 0.25 * prec + rec_cand
        f05_cand = (1.25 * prec * rec_cand) / denom_f05_cand if denom_f05_cand > 0 else 0.0

        e2e_rec = tp / TOTAL_GROUND_TRUTH_MATCHES
        denom_f05_e2e = 0.25 * prec + e2e_rec
        f05_e2e = (1.25 * prec * e2e_rec) / denom_f05_e2e if denom_f05_e2e > 0 else 0.0

        rows.append({
            "threshold": t,
            "precision": round(prec, 6),
            "cand_recall": round(rec_cand, 6),
            "cand_F0.5": round(f05_cand, 6),
            "e2e_recall": round(e2e_rec, 6),
            "e2e_F0.5": round(f05_e2e, 6),
            "false_positives": fp,
            "false_negatives": fn_cand,
            "true_positives": tp,
            "predicted_matches": int(tp + fp),
        })

    res_df = pd.DataFrame(rows)
    print(res_df.to_string(index=False))

    report_str = f"""================================================================================
NEW BLOCKING (SELECTIVE MULTI-PASS) + THRESHOLD SWEEP REPORT
================================================================================
Candidate File: {CANDIDATE_FILE}
Total Candidates: {len(cand_df):,}
Candidate Recall: {cand_recall:.6f} ({cand_recall*100:.2f}%) vs Baseline 73.89% (+19.78% gain!)
Recovered True Matches: {true_in_cands:,} / {TOTAL_GROUND_TRUTH_MATCHES:,}

SWEEP RESULTS:
{res_df.to_string(index=False)}
================================================================================
"""
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_str)
    print("Report written to:", OUTPUT_REPORT)

if __name__ == "__main__":
    main()
