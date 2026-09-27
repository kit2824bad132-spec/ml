"""
Phase 10 & 11: Combined Best Configuration & In-Depth Error Analysis
Combines Best Blocking, Best Model, and Best Threshold on Clean Validation.
Performs comprehensive error taxonomy:
  A. True match not in candidates
  B. True match in candidates but model scored low
  C. Wrong high-confidence match
  D. Multiple plausible matches
  E. No-match case
  F. Missing name
  G. Missing address
  H. Name/Address conflict
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from rapidfuzz import process
from rapidfuzz.fuzz import ratio, token_sort_ratio, token_set_ratio
from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix

from preprocessing import normalize_country, normalize_text

SPLIT_FILE = "dataset/train/clean_validation_split.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"
S1_FILE = "dataset/train/train_source1_sample_50k.tsv"
S2_FILE = "dataset/train/train_source2_matches_50k.tsv"
S3_FILE = "dataset/train/train_source3_matches_50k.tsv"
SELECTIVE_CANDIDATE_FILE = "dataset/train/clean_blocking_candidates_selective.tsv"
BASELINE_CANDIDATE_FILE = "dataset/train/clean_validation_candidates.tsv"

OUTPUT_ERROR_ANALYSIS = "output/clean_validation_error_analysis.txt"
OUTPUT_COMBINED_REPORT = "output/combined_best_configuration_report.txt"

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
    scores = process.cpdist(a, b, scorer=scorer, workers=6, dtype=np.float32)
    mask = np.array([bool(x and y) for x, y in zip(a, b)])
    return np.where(mask, scores / 100.0, 0.0)

def run_evaluation(model_path, candidate_path, best_threshold):
    print(f"Loading model from {model_path}...")
    model_obj = joblib.load(model_path)
    model = model_obj["model"] if isinstance(model_obj, dict) and "model" in model_obj else model_obj

    print("Loading clean validation split...")
    split_df = pd.read_csv(SPLIT_FILE, sep="\t", dtype=str)
    val_s1_ids = set(split_df.loc[split_df["split"] == "validation", "entity_id"])
    total_val_s1 = len(val_s1_ids)

    print("Loading validation ground truth...")
    val_truth_dict = defaultdict(set)
    total_gt_matches = 0
    for chunk in pd.read_csv(GT_FILE, sep="\t", dtype=str, keep_default_na=False, chunksize=200000):
        sub = chunk[chunk.source1_entity_id.isin(val_s1_ids)]
        for row in sub.itertuples(index=False):
            for m in row.matched_entity_ids.split(","):
                m = m.strip()
                if m:
                    val_truth_dict[row.source1_entity_id].add(m)
                    total_gt_matches += 1

    print(f"Total Validation S1 Entities: {total_val_s1:,}")
    print(f"Total Validation Ground Truth Matches: {total_gt_matches:,}")

    print("Loading raw records...")
    s1_df = pd.read_csv(S1_FILE, sep="\t", dtype=str, keep_default_na=False)
    s1_df = s1_df[s1_df.entity_id.isin(val_s1_ids)].copy()
    s1_records = {
        row.entity_id: {
            "name": normalize_text(row.business_name),
            "address": normalize_text(row.business_address),
            "country": normalize_country(row.country),
            "raw_name": row.business_name.strip(),
            "raw_address": row.business_address.strip(),
        }
        for row in s1_df.itertuples(index=False)
    }

    target_df = pd.concat([
        pd.read_csv(S2_FILE, sep="\t", dtype=str, keep_default_na=False),
        pd.read_csv(S3_FILE, sep="\t", dtype=str, keep_default_na=False),
    ], ignore_index=True).drop_duplicates("entity_id")
    target_records = {
        row.entity_id: {
            "name": normalize_text(row.business_name),
            "address": normalize_text(row.business_address),
            "country": normalize_country(row.country),
            "raw_name": row.business_name.strip(),
            "raw_address": row.business_address.strip(),
        }
        for row in target_df.itertuples(index=False)
    }

    print(f"Loading candidate pairs from {candidate_path}...")
    cand_df = pd.read_csv(candidate_path, sep="\t", dtype=str)
    cand_df = cand_df[cand_df.source1_entity_id.isin(val_s1_ids)].copy()
    total_candidates = len(cand_df)
    s1_cand_counts = cand_df.groupby("source1_entity_id").size()
    avg_cands_s1 = total_candidates / total_val_s1
    max_cands_s1 = s1_cand_counts.max() if not s1_cand_counts.empty else 0

    # Blocking recall
    cand_pairs_set = set(zip(cand_df.source1_entity_id, cand_df.candidate_entity_id))
    true_pairs_set = {(sid, cid) for sid, cids in val_truth_dict.items() for cid in cids}
    
    true_matches_in_cands = true_pairs_set & cand_pairs_set
    cand_recall = len(true_matches_in_cands) / total_gt_matches

    print(f"Candidates: {total_candidates:,}")
    print(f"Candidate Recall: {cand_recall:.6f} ({len(true_matches_in_cands):,} / {total_gt_matches:,})")

    # Vectorized feature computation & prediction in chunks
    print("Computing features & predictions...")
    all_probs = []
    chunk_size = 50000
    for i in range(0, total_candidates, chunk_size):
        chunk = cand_df.iloc[i:i+chunk_size]
        n1 = [s1_records[sid]["name"] for sid in chunk.source1_entity_id]
        a1 = [s1_records[sid]["address"] for sid in chunk.source1_entity_id]
        c1 = [s1_records[sid]["country"] for sid in chunk.source1_entity_id]

        n2 = [target_records[cid]["name"] for cid in chunk.candidate_entity_id]
        a2 = [target_records[cid]["address"] for cid in chunk.candidate_entity_id]
        c2 = [target_records[cid]["country"] for cid in chunk.candidate_entity_id]

        feats = pd.DataFrame()
        feats["name_ratio"] = safe_ratio_vec(n1, n2, ratio)
        feats["name_token_sort"] = safe_ratio_vec(n1, n2, token_sort_ratio)
        feats["name_token_set"] = safe_ratio_vec(n1, n2, token_set_ratio)
        feats["address_ratio"] = safe_ratio_vec(a1, a2, ratio)
        feats["address_token_sort"] = safe_ratio_vec(a1, a2, token_sort_ratio)
        feats["address_token_set"] = safe_ratio_vec(a1, a2, token_set_ratio)
        feats["country_match"] = (np.array(c1) == np.array(c2)).astype(int)
        feats["name_length_diff"] = np.abs(np.array([len(x) for x in n1]) - np.array([len(x) for x in n2]))
        feats["address_length_diff"] = np.abs(np.array([len(x) for x in a1]) - np.array([len(x) for x in a2]))

        chunk_probs = model.predict_proba(feats[FEATURE_COLUMNS].values)[:, 1]
        all_probs.extend(chunk_probs)

    cand_df["prob"] = all_probs
    cand_df["pred"] = (cand_df["prob"] >= best_threshold).astype(int)
    cand_df["is_true"] = [1 if pair in true_pairs_set else 0 for pair in zip(cand_df.source1_entity_id, cand_df.candidate_entity_id)]

    # Metrics
    tp = int(((cand_df["pred"] == 1) & (cand_df["is_true"] == 1)).sum())
    fp = int(((cand_df["pred"] == 1) & (cand_df["is_true"] == 0)).sum())
    fn_cand = int(((cand_df["pred"] == 0) & (cand_df["is_true"] == 1)).sum())
    fn_blocking = total_gt_matches - len(true_matches_in_cands)
    total_fn = fn_cand + fn_blocking

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    cand_rec = tp / len(true_matches_in_cands) if len(true_matches_in_cands) > 0 else 0.0
    e2e_rec = tp / total_gt_matches
    cand_f05 = (1.25 * prec * cand_rec) / (0.25 * prec + cand_rec) if (0.25 * prec + cand_rec) > 0 else 0.0
    e2e_f05 = (1.25 * prec * e2e_rec) / (0.25 * prec + e2e_rec) if (0.25 * prec + e2e_rec) > 0 else 0.0

    print("\n--- RESULTS AT THRESHOLD", best_threshold, "---")
    print(f"Precision: {prec:.6f} ({tp:,} / {tp+fp:,})")
    print(f"Candidate-Level Recall: {cand_rec:.6f} ({tp:,} / {len(true_matches_in_cands):,})")
    print(f"End-to-End Recall: {e2e_rec:.6f} ({tp:,} / {total_gt_matches:,})")
    print(f"Candidate-Level F0.5: {cand_f05:.6f}")
    print(f"End-to-End F0.5: {e2e_f05:.6f}")
    print(f"False Positives: {fp:,}")
    print(f"False Negatives (Model): {fn_cand:,}")
    print(f"False Negatives (Blocking): {fn_blocking:,}")
    print(f"Total False Negatives: {total_fn:,}")

    # ============================================================
    # PHASE 11: ERROR ANALYSIS TAXONOMY
    # ============================================================
    print("\nRunning Phase 11 Error Categorization...")
    error_counts = defaultdict(int)
    error_samples = defaultdict(list)

    # Category A: True match not in candidates
    unretrieved_pairs = true_pairs_set - cand_pairs_set
    error_counts["A_TRUE_MATCH_NOT_IN_CANDIDATES"] = len(unretrieved_pairs)
    for sid, cid in list(unretrieved_pairs)[:5]:
        s_rec = s1_records.get(sid, {})
        c_rec = target_records.get(cid, {})
        error_samples["A_TRUE_MATCH_NOT_IN_CANDIDATES"].append({
            "s1_id": sid, "c_id": cid,
            "s_name": s_rec.get("raw_name"), "c_name": c_rec.get("raw_name"),
            "s_addr": s_rec.get("raw_address"), "c_addr": c_rec.get("raw_address"),
            "s_country": s_rec.get("country"), "c_country": c_rec.get("country"),
        })

    # Category B: True match in candidates but model scored low
    model_fn_df = cand_df[(cand_df["is_true"] == 1) & (cand_df["pred"] == 0)]
    error_counts["B_TRUE_MATCH_LOW_MODEL_SCORE"] = len(model_fn_df)
    for row in model_fn_df.head(5).itertuples():
        s_rec = s1_records.get(row.source1_entity_id, {})
        c_rec = target_records.get(row.candidate_entity_id, {})
        error_samples["B_TRUE_MATCH_LOW_MODEL_SCORE"].append({
            "s1_id": row.source1_entity_id, "c_id": row.candidate_entity_id,
            "prob": row.prob,
            "s_name": s_rec.get("raw_name"), "c_name": c_rec.get("raw_name"),
            "s_addr": s_rec.get("raw_address"), "c_addr": c_rec.get("raw_address"),
        })

    # Category C: Wrong high confidence match (FP with prob >= threshold)
    fp_df = cand_df[(cand_df["is_true"] == 0) & (cand_df["pred"] == 1)]
    error_counts["C_WRONG_HIGH_CONFIDENCE_MATCH"] = len(fp_df)
    for row in fp_df.head(5).itertuples():
        s_rec = s1_records.get(row.source1_entity_id, {})
        c_rec = target_records.get(row.candidate_entity_id, {})
        error_samples["C_WRONG_HIGH_CONFIDENCE_MATCH"].append({
            "s1_id": row.source1_entity_id, "c_id": row.candidate_entity_id,
            "prob": row.prob,
            "s_name": s_rec.get("raw_name"), "c_name": c_rec.get("raw_name"),
            "s_addr": s_rec.get("raw_address"), "c_addr": c_rec.get("raw_address"),
        })

    # Category D: Multiple plausible matches (S1 with >1 positive prediction)
    pred_pos_df = cand_df[cand_df["pred"] == 1]
    s1_pred_counts = pred_pos_df.groupby("source1_entity_id").size()
    multi_match_s1 = set(s1_pred_counts[s1_pred_counts > 1].index)
    error_counts["D_MULTIPLE_PLAUSIBLE_MATCHES"] = len(multi_match_s1)

    # Category E: No-match case (S1 with 0 ground truth matches, or S1 where no prediction was made)
    s1_with_gt = set(val_truth_dict.keys())
    s1_no_gt = val_s1_ids - s1_with_gt
    s1_with_pred = set(pred_pos_df["source1_entity_id"])
    s1_no_pred = val_s1_ids - s1_with_pred
    error_counts["E_NO_MATCH_SOURCE1_CASES"] = len(s1_no_gt)
    error_counts["E_UNPREDICTED_SOURCE1_CASES"] = len(s1_no_pred)

    # Specific attribute error categories in False Positives & False Negatives:
    missing_name_count = 0
    missing_addr_count = 0
    conflict_count = 0

    all_error_df = pd.concat([
        cand_df[(cand_df["pred"] == 1) & (cand_df["is_true"] == 0)],
        cand_df[(cand_df["pred"] == 0) & (cand_df["is_true"] == 1)],
    ])

    for row in all_error_df.itertuples():
        s_rec = s1_records.get(row.source1_entity_id, {})
        c_rec = target_records.get(row.candidate_entity_id, {})
        s_name = s_rec.get("name", "")
        c_name = c_rec.get("name", "")
        s_addr = s_rec.get("address", "")
        c_addr = c_rec.get("address", "")

        if len(s_name) < 2 or len(c_name) < 2:
            missing_name_count += 1
        if len(s_addr) < 4 or len(c_addr) < 4:
            missing_addr_count += 1
        
        # Name/Address Conflict: high name similarity (>=0.85) but low address similarity (<=0.40) or vice versa
        n_sim = ratio(s_name, c_name) / 100.0 if s_name and c_name else 0.0
        a_sim = ratio(s_addr, c_addr) / 100.0 if s_addr and c_addr else 0.0
        if (n_sim >= 0.85 and a_sim <= 0.40) or (a_sim >= 0.85 and n_sim <= 0.40):
            conflict_count += 1

    error_counts["F_MISSING_OR_SHORT_NAME"] = missing_name_count
    error_counts["G_MISSING_OR_SHORT_ADDRESS"] = missing_addr_count
    error_counts["H_NAME_ADDRESS_CONFLICT"] = conflict_count

    total_error_instances = error_counts["A_TRUE_MATCH_NOT_IN_CANDIDATES"] + error_counts["B_TRUE_MATCH_LOW_MODEL_SCORE"] + error_counts["C_WRONG_HIGH_CONFIDENCE_MATCH"]

    print("\n--- ERROR TAXONOMY BREAKDOWN ---")
    for k, v in error_counts.items():
        pct = (v / total_error_instances * 100) if total_error_instances > 0 and k.startswith(("A", "B", "C")) else 0
        print(f"  {k:<35}: {v:>6,}" + (f" ({pct:>5.1f}%)" if pct > 0 else ""))

    # Write detailed error report
    with open(OUTPUT_ERROR_ANALYSIS, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("AMAZON ML CHALLENGE 2026 - CLEAN VALIDATION ERROR ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Candidate File: {candidate_path}\n")
        f.write(f"Threshold: {best_threshold}\n")
        f.write(f"Total True Matches: {total_gt_matches:,}\n")
        f.write(f"Total Errors (A + B + C): {total_error_instances:,}\n")
        f.write("-" * 80 + "\n\n")

        f.write("ERROR CATEGORY BREAKDOWN:\n")
        for k, v in error_counts.items():
            pct = (v / total_error_instances * 100) if total_error_instances > 0 and k.startswith(("A", "B", "C")) else 0
            f.write(f"  {k:<35}: {v:>6,}" + (f" ({pct:>5.1f}% of primary errors)\n" if pct > 0 else "\n"))

        f.write("\n" + "-" * 80 + "\n")
        f.write("REPRESENTATIVE ERROR EXAMPLES:\n")
        for cat, samples in error_samples.items():
            f.write(f"\n--- {cat} ---\n")
            for i, s in enumerate(samples, 1):
                f.write(f"Sample {i}:\n")
                for sk, sv in s.items():
                    f.write(f"  {sk}: {sv}\n")

        f.write("\n" + "=" * 80 + "\n")

    # Write Phase 10 Combined Report
    with open(OUTPUT_COMBINED_REPORT, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("AMAZON ML CHALLENGE 2026 - COMBINED BEST CONFIGURATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write("CONFIGURATION:\n")
        f.write(f"Model Artifact: {model_path}\n")
        f.write(f"Blocking Strategy: Selective Multi-Pass Blocking (4 country passes + 2 token-co-occurrence passes)\n")
        f.write(f"Candidate File: {candidate_path}\n")
        f.write(f"Decision Threshold: {best_threshold}\n\n")
        f.write("CLEAN VALIDATION METRICS:\n")
        f.write(f"Baseline F0.5: 0.900691 (Candidate-Level) / 0.845879 (End-to-End)\n")
        f.write(f"Final Candidate F0.5: {cand_f05:.6f}\n")
        f.write(f"Final End-to-End F0.5: {e2e_f05:.6f}\n")
        f.write(f"Candidate F0.5 Improvement: {cand_f05 - 0.900691:+.6f}\n")
        f.write(f"End-to-End F0.5 Improvement: {e2e_f05 - 0.845879:+.6f}\n\n")
        f.write(f"Candidate Recall: {cand_recall:.6f} ({cand_recall*100:.2f}% vs Baseline 73.89%)\n")
        f.write(f"Precision: {prec:.6f} ({prec*100:.2f}%)\n")
        f.write(f"Recall (Model): {cand_rec:.6f} ({cand_rec*100:.2f}%)\n")
        f.write(f"End-to-End Recall: {e2e_rec:.6f} ({e2e_rec*100:.2f}%)\n")
        f.write(f"False Positives: {fp:,}\n")
        f.write(f"False Negatives (Model): {fn_cand:,}\n")
        f.write(f"False Negatives (Blocking): {fn_blocking:,}\n")
        f.write(f"Total False Negatives: {total_fn:,}\n\n")
        f.write("CANDIDATE STATISTICS:\n")
        f.write(f"Total Candidates: {total_candidates:,}\n")
        f.write(f"Average Candidates per Source-1: {avg_cands_s1:.2f}\n")
        f.write(f"Maximum Candidates per Source-1: {max_cands_s1}\n")
        f.write("=" * 80 + "\n")

    print(f"Reports saved to {OUTPUT_ERROR_ANALYSIS} and {OUTPUT_COMBINED_REPORT}")

if __name__ == "__main__":
    import sys
    model_p = sys.argv[1] if len(sys.argv) > 1 else "dataset/train/clean_validation_model_improved.pkl"
    cand_p = sys.argv[2] if len(sys.argv) > 2 else "dataset/train/clean_blocking_candidates_selective.tsv"
    thresh = float(sys.argv[3]) if len(sys.argv) > 3 else 0.97
    run_evaluation(model_p, cand_p, thresh)
