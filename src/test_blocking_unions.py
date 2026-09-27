"""
Analyze and optimize blocking unions from clean_blocking_pass_candidates.tsv.gz
Evaluates candidate recall, candidate size, and model performance.
Zero leakage: Ground truth is loaded ONLY after candidate set is frozen.
"""

import gzip
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from rapidfuzz import process
from rapidfuzz.fuzz import ratio, token_sort_ratio, token_set_ratio
import joblib
from sklearn.metrics import precision_score, recall_score, fbeta_score

from preprocessing import normalize_country, normalize_text

SPLIT_FILE = "dataset/train/clean_validation_split.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"
S1_FILE = "dataset/train/train_source1_sample_50k.tsv"
S2_FILE = "dataset/train/train_source2_matches_50k.tsv"
S3_FILE = "dataset/train/train_source3_matches_50k.tsv"
CURRENT_FILE = "dataset/train/clean_validation_candidates.tsv"
PASS_FILE = "dataset/train/clean_blocking_pass_candidates.tsv.gz"
MODEL_FILE = "dataset/train/clean_validation_model_improved.pkl"

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

def main():
    print("Loading baseline candidate pairs...")
    current = pd.read_csv(CURRENT_FILE, sep="\t", dtype=str)
    current_pairs = set(zip(current.source1_entity_id, current.candidate_entity_id))
    print(f"Baseline pairs: {len(current_pairs):,}")

    print("Loading pass candidates by strategy...")
    pass_pairs = defaultdict(set)
    with gzip.open(PASS_FILE, "rt", encoding="utf-8") as f:
        for chunk in pd.read_csv(f, sep="\t", dtype=str, chunksize=100000):
            for row in chunk.itertuples(index=False):
                pass_pairs[row.blocking_pass].add((row.source1_entity_id, row.candidate_entity_id))

    for strat, pairs in pass_pairs.items():
        print(f"  {strat:<45}: {len(pairs):>10,} pairs")

    # Load validation ground truth for recall evaluation
    split_df = pd.read_csv(SPLIT_FILE, sep="\t", dtype=str)
    val_s1_ids = set(split_df.loc[split_df["split"] == "validation", "entity_id"])
    
    val_truth_pairs = set()
    for chunk in pd.read_csv(GT_FILE, sep="\t", dtype=str, keep_default_na=False, chunksize=200000):
        sub = chunk[chunk.source1_entity_id.isin(val_s1_ids)]
        for row in sub.itertuples(index=False):
            for m in row.matched_entity_ids.split(","):
                m = m.strip()
                if m:
                    val_truth_pairs.add((row.source1_entity_id, m))
    total_gt = len(val_truth_pairs)
    print(f"\nTotal Validation Ground Truth Matches: {total_gt:,}")

    # Evaluate multiple blocking union strategies
    union_strategies = {
        "1. Current Baseline (4 passes)": current_pairs,
        "2. Selective Union (Baseline + HouseNum + NameAddrToken)": current_pairs | pass_pairs["house_number_address_token_country"] | pass_pairs["name_token_address_token_country"],
        "3. Selective + ExactNameAnyCountry": current_pairs | pass_pairs["house_number_address_token_country"] | pass_pairs["name_token_address_token_country"] | pass_pairs["exact_name_any_country"],
        "4. Selective + AddressTailCity": current_pairs | pass_pairs["house_number_address_token_country"] | pass_pairs["name_token_address_token_country"] | pass_pairs["address_tail_city_token_country"],
        "5. Selective + ExactNameAny + AddressTailCity": current_pairs | pass_pairs["house_number_address_token_country"] | pass_pairs["name_token_address_token_country"] | pass_pairs["exact_name_any_country"] | pass_pairs["address_tail_city_token_country"],
    }

    print("\n--- BLOCKING UNION COMPARISON ---")
    results = []
    for name, pairs in union_strategies.items():
        recovered = len(val_truth_pairs & pairs)
        rec = recovered / total_gt
        n_pairs = len(pairs)
        avg_s1 = n_pairs / len(val_s1_ids)
        # Calculate max candidates per S1
        s1_counts = defaultdict(int)
        for sid, _ in pairs:
            s1_counts[sid] += 1
        max_s1 = max(s1_counts.values()) if s1_counts else 0
        print(f"\nStrategy: {name}")
        print(f"  Total Pairs: {n_pairs:,} (Growth: {n_pairs/len(current_pairs):.3f}x)")
        print(f"  Candidate Recall: {rec:.6f} ({recovered:,} / {total_gt:,} matches)")
        print(f"  Avg / Max per S1: {avg_s1:.1f} / {max_s1}")
        results.append({
            "name": name,
            "pairs": pairs,
            "n_pairs": n_pairs,
            "recovered": recovered,
            "recall": rec,
            "avg_s1": avg_s1,
            "max_s1": max_s1,
        })

    # Save candidate files for top candidates to evaluate with model
    print("\nWriting Strategy 4 and Strategy 5 candidate files for model evaluation...")
    strat4_df = pd.DataFrame(sorted(union_strategies["4. Selective + AddressTailCity"]), columns=["source1_entity_id", "candidate_entity_id"])
    strat4_file = "dataset/train/clean_blocking_candidates_strat4.tsv"
    strat4_df.to_csv(strat4_file, sep="\t", index=False)
    print(f"Saved: {strat4_file} ({len(strat4_df):,} pairs)")

    strat5_df = pd.DataFrame(sorted(union_strategies["5. Selective + ExactNameAny + AddressTailCity"]), columns=["source1_entity_id", "candidate_entity_id"])
    strat5_file = "dataset/train/clean_blocking_candidates_strat5.tsv"
    strat5_df.to_csv(strat5_file, sep="\t", index=False)
    print(f"Saved: {strat5_file} ({len(strat5_df):,} pairs)")

if __name__ == "__main__":
    main()
