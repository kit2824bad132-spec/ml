"""
Phase 8 & 9: Hard Negative Generation, Ratio Experiments, and Model Comparison
Exclusively uses training Source-1 entities (40,000) and training ground truth.
Zero validation ground truth or validation S1 entities are accessed.
"""

import os
import re
import math
import random
import time
from collections import defaultdict
import joblib
import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.fuzz import ratio, token_sort_ratio, token_set_ratio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix

from preprocessing import normalize_country, normalize_text

S1_FILE = "dataset/train/train_source1_sample_50k.tsv"
S2_FILE = "dataset/train/train_source2_matches_50k.tsv"
S3_FILE = "dataset/train/train_source3_matches_50k.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"
SPLIT_FILE = "dataset/train/clean_validation_split.tsv"
VAL_FEATURES_FILE = "dataset/train/clean_validation_validation_features.tsv"
OUTPUT_REPORT = "output/hard_negative_ratio_experiments.txt"

RANDOM_SEED = 42
MAX_TOKEN_FREQUENCY = 5000
SHORTLIST_SIZE = 300

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

MODEL_PARAMETERS = {
    "n_estimators": 250,
    "max_depth": 14,
    "min_samples_leaf": 3,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

def safe_ratio_vec(a, b, scorer):
    scores = process.cpdist(a, b, scorer=scorer, workers=6, dtype=np.float32)
    mask = np.array([bool(x and y) for x, y in zip(a, b)])
    return np.where(mask, scores / 100.0, 0.0)

def extract_numbers(text):
    return set(re.findall(r"\b\d+\b", text))

def build_token_index(records, field):
    index = defaultdict(list)
    frequencies = defaultdict(int)
    for cid, rec in records.items():
        tokens = {t for t in rec[field].split() if len(t) >= 2}
        for token in tokens:
            index[token].append(cid)
            frequencies[token] += 1
    return index, frequencies

def token_shortlist(text, index, frequencies):
    scores = defaultdict(float)
    tokens = {t for t in text.split() if len(t) >= 2}
    for token in tokens:
        freq = frequencies.get(token, 0)
        if freq == 0 or freq > MAX_TOKEN_FREQUENCY:
            continue
        weight = 1.0 / math.log2(freq + 2)
        for cid in index[token]:
            scores[cid] += weight
    return sorted(scores, key=lambda c: (-scores[c], c))[:SHORTLIST_SIZE]

def compute_features_df(pairs_df, s1_records, target_records):
    n1 = [s1_records[sid]["name"] for sid in pairs_df.source1_entity_id]
    a1 = [s1_records[sid]["address"] for sid in pairs_df.source1_entity_id]
    c1 = [s1_records[sid]["country"] for sid in pairs_df.source1_entity_id]

    n2 = [target_records[cid]["name"] for cid in pairs_df.candidate_entity_id]
    a2 = [target_records[cid]["address"] for cid in pairs_df.candidate_entity_id]
    c2 = [target_records[cid]["country"] for cid in pairs_df.candidate_entity_id]

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
    return feats

def main():
    start_time = time.time()
    print("Loading split data...")
    split_df = pd.read_csv(SPLIT_FILE, sep="\t", dtype=str)
    train_s1_ids = set(split_df.loc[split_df["split"] == "train", "entity_id"])
    val_s1_ids = set(split_df.loc[split_df["split"] == "validation", "entity_id"])
    print(f"Training S1 count: {len(train_s1_ids):,}, Validation S1 count: {len(val_s1_ids):,}")

    print("Loading training records...")
    s1_df = pd.read_csv(S1_FILE, sep="\t", dtype=str, keep_default_na=False)
    s1_df = s1_df[s1_df.entity_id.isin(train_s1_ids)].copy()
    s1_records = {
        row.entity_id: {
            "name": normalize_text(row.business_name),
            "address": normalize_text(row.business_address),
            "country": normalize_country(row.country),
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
        }
        for row in target_df.itertuples(index=False)
    }
    all_target_cids = list(target_records.keys())

    print("Loading training ground truth (strictly for train_s1_ids)...")
    train_truth = defaultdict(set)
    for chunk in pd.read_csv(GT_FILE, sep="\t", dtype=str, keep_default_na=False, chunksize=200000):
        sub = chunk[chunk.source1_entity_id.isin(train_s1_ids)]
        for row in sub.itertuples(index=False):
            for m in row.matched_entity_ids.split(","):
                m = m.strip()
                if m and m in target_records:
                    train_truth[row.source1_entity_id].add(m)
    
    positives = []
    for sid, targets in train_truth.items():
        for cid in sorted(targets):
            positives.append((sid, cid, 1, "positive"))
    positives_df = pd.DataFrame(positives, columns=["source1_entity_id", "candidate_entity_id", "label", "category"])
    print(f"Total training positive pairs: {len(positives_df):,}")

    # Build token indices on training targets
    print("Building token indices for candidate retrieval...")
    name_idx, name_freq = build_token_index(target_records, "name")
    addr_idx, addr_freq = build_token_index(target_records, "address")

    # Generate Hard Negatives across categories
    print("Generating categorized hard negatives for training S1...")
    rng = random.Random(RANDOM_SEED)
    categorized_negatives = defaultdict(list)
    random_negatives = []

    for idx, (sid, s_rec) in enumerate(s1_records.items(), 1):
        if idx % 10000 == 0:
            print(f"Processed {idx:,}/{len(s1_records):,} training S1 entities...")
        true_set = train_truth[sid]

        # 1. Random negatives (2 per S1)
        r_chosen = set()
        attempts = 0
        while len(r_chosen) < 2 and attempts < 100:
            attempts += 1
            cid = rng.choice(all_target_cids)
            if cid not in true_set:
                r_chosen.add(cid)
        for cid in r_chosen:
            random_negatives.append((sid, cid, 0, "random_negative"))

        # 2. Hard negatives via token shortlist
        n_short = token_shortlist(s_rec["name"], name_idx, name_freq)
        a_short = token_shortlist(s_rec["address"], addr_idx, addr_freq)
        cand_pool = set(n_short) | set(a_short)

        s_nums = extract_numbers(s_rec["address"])

        best_high_name = None
        best_high_addr = None
        best_dual = None
        best_number_mismatch = None
        best_country_mismatch = None

        for cid in cand_pool:
            if cid in true_set:
                continue
            c_rec = target_records[cid]
            n_sim = ratio(s_rec["name"], c_rec["name"]) / 100.0 if s_rec["name"] and c_rec["name"] else 0.0
            a_sim = ratio(s_rec["address"], c_rec["address"]) / 100.0 if s_rec["address"] and c_rec["address"] else 0.0
            diff_country = bool(s_rec["country"] and c_rec["country"] and s_rec["country"] != c_rec["country"])

            # High name confusion (similar name, different location)
            if n_sim >= 0.70 and a_sim <= 0.65:
                if best_high_name is None or n_sim > best_high_name[0]:
                    best_high_name = (n_sim, cid)

            # High address confusion (same building/area, different name)
            if a_sim >= 0.70 and n_sim <= 0.65:
                if best_high_addr is None or a_sim > best_high_addr[0]:
                    best_high_addr = (a_sim, cid)

            # Dual moderate near-miss (name >= 0.65 and address >= 0.50)
            if n_sim >= 0.65 and a_sim >= 0.50:
                dual_score = (n_sim + a_sim) / 2.0
                if best_dual is None or dual_score > best_dual[0]:
                    best_dual = (dual_score, cid)

            # Number mismatch in address (e.g. 104 Main vs 108 Main)
            if s_nums and a_sim >= 0.60:
                c_nums = extract_numbers(c_rec["address"])
                if c_nums and s_nums != c_nums:
                    if best_number_mismatch is None or a_sim > best_number_mismatch[0]:
                        best_number_mismatch = (a_sim, cid)

            # Country conflict with high name
            if diff_country and n_sim >= 0.65:
                if best_country_mismatch is None or n_sim > best_country_mismatch[0]:
                    best_country_mismatch = (n_sim, cid)

        selected_cids = set()
        if best_high_name and best_high_name[1] not in selected_cids:
            categorized_negatives["high_name"].append((sid, best_high_name[1], 0, "high_name"))
            selected_cids.add(best_high_name[1])
        if best_high_addr and best_high_addr[1] not in selected_cids:
            categorized_negatives["high_address"].append((sid, best_high_addr[1], 0, "high_address"))
            selected_cids.add(best_high_addr[1])
        if best_dual and best_dual[1] not in selected_cids:
            categorized_negatives["dual_near_miss"].append((sid, best_dual[1], 0, "dual_near_miss"))
            selected_cids.add(best_dual[1])
        if best_number_mismatch and best_number_mismatch[1] not in selected_cids:
            categorized_negatives["number_mismatch"].append((sid, best_number_mismatch[1], 0, "number_mismatch"))
            selected_cids.add(best_number_mismatch[1])
        if best_country_mismatch and best_country_mismatch[1] not in selected_cids:
            categorized_negatives["country_mismatch"].append((sid, best_country_mismatch[1], 0, "country_mismatch"))
            selected_cids.add(best_country_mismatch[1])

    print("Generated hard negative counts by category:")
    for cat, items in categorized_negatives.items():
        print(f"  {cat}: {len(items):,}")
    print(f"  random_negatives: {len(random_negatives):,}")

    all_hard_list = []
    for cat, items in categorized_negatives.items():
        all_hard_list.extend(items)
    
    # Load Validation Features for direct evaluation
    print("\nLoading clean validation features for evaluation...")
    val_feats_df = pd.read_csv(VAL_FEATURES_FILE, sep="\t")
    y_val = val_feats_df["label"].values.astype(int)
    X_val = val_feats_df[FEATURE_COLUMNS].values
    print(f"Validation rows: {len(val_feats_df):,}, positive: {(y_val == 1).sum():,}, negative: {(y_val == 0).sum():,}")

    # Build Training Sets for Different Configurations / Ratios
    # Total Positives = len(positives_df) = 72,777
    N_pos = len(positives_df)
    
    # Let's define the configurations:
    # Model A: Baseline legacy (already trained and saved as clean_validation_model_baseline.pkl)
    # Model B: Current improved (already trained and saved as clean_validation_model_improved.pkl)
    # Model C variants:
    # 1. Ratio 1:1 (~73k negatives: 36.5k random + 36.5k hard)
    # 2. Ratio 1:2 (~145k negatives: 72.5k random + 72.5k hard)
    # 3. Ratio 1:3 (~218k negatives: 80k random + ~138k hard)
    # 4. Ratio 1:5 (~364k negatives: 80k random + ~284k hard)
    # 5. Model C Balanced Hard (All categories combined + random negatives, approx 200k negatives)

    configs = {}
    
    # Prep negatives pools
    rng.shuffle(random_negatives)
    rng.shuffle(all_hard_list)

    def assemble_dataset(n_rand, n_hard):
        rand_sub = random_negatives[:n_rand]
        hard_sub = all_hard_list[:n_hard]
        neg_df = pd.DataFrame(rand_sub + hard_sub, columns=["source1_entity_id", "candidate_entity_id", "label", "category"])
        train_df = pd.concat([positives_df, neg_df], ignore_index=True)
        train_df = train_df.drop_duplicates(["source1_entity_id", "candidate_entity_id"]).reset_index(drop=True)
        return train_df

    configs["Ratio_1_to_1"] = assemble_dataset(n_rand=int(N_pos * 0.5), n_hard=int(N_pos * 0.5))
    configs["Ratio_1_to_2"] = assemble_dataset(n_rand=int(N_pos * 1.0), n_hard=int(N_pos * 1.0))
    configs["Ratio_1_to_3"] = assemble_dataset(n_rand=int(N_pos * 1.0), n_hard=int(N_pos * 2.0))
    configs["Ratio_1_to_5"] = assemble_dataset(n_rand=int(N_pos * 1.5), n_hard=int(N_pos * 3.5))
    configs["Model_C_All_Hard"] = assemble_dataset(n_rand=len(random_negatives), n_hard=len(all_hard_list))

    print("\n--- Training Set Sizes ---")
    for name, df in configs.items():
        n_pos = (df.label == 1).sum()
        n_neg = (df.label == 0).sum()
        print(f"  {name}: Total={len(df):,}, Pos={n_pos:,}, Neg={n_neg:,} (Ratio {n_neg/n_pos:.2f}:1)")

    # Evaluate Model A (Baseline) and Model B (Current Improved) from disk first
    print("\nEvaluating existing Model A (Baseline legacy) and Model B (Current improved)...")
    eval_results = []

    model_a_data = joblib.load("dataset/train/clean_validation_model_baseline.pkl")
    model_a = model_a_data["model"]
    probs_a = model_a.predict_proba(X_val)[:, 1]

    model_b_data = joblib.load("dataset/train/clean_validation_model_improved.pkl")
    model_b = model_b_data["model"]
    probs_b = model_b.predict_proba(X_val)[:, 1]

    def evaluate_probabilities(name, probs, threshold_list=[0.70, 0.80, 0.90, 0.95, 0.96, 0.97, 0.98]):
        best_f05 = -1
        best_t = None
        best_metrics = None
        for t in threshold_list:
            preds = (probs >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
            prec = precision_score(y_val, preds, zero_division=0)
            rec = recall_score(y_val, preds, zero_division=0)
            f05 = fbeta_score(y_val, preds, beta=0.5, zero_division=0)
            if f05 > best_f05:
                best_f05 = f05
                best_t = t
                best_metrics = {
                    "threshold": t,
                    "precision": prec,
                    "recall": rec,
                    "f05": f05,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                }
        # Also compute at fixed 0.70
        preds_70 = (probs >= 0.70).astype(int)
        tn70, fp70, fn70, tp70 = confusion_matrix(y_val, preds_70, labels=[0, 1]).ravel()
        p70 = precision_score(y_val, preds_70, zero_division=0)
        r70 = recall_score(y_val, preds_70, zero_division=0)
        f70 = fbeta_score(y_val, preds_70, beta=0.5, zero_division=0)
        return {
            "name": name,
            "f05_at_070": f70,
            "prec_at_070": p70,
            "rec_at_070": r70,
            "fp_at_070": fp70,
            "fn_at_070": fn70,
            "best_threshold": best_t,
            "best_f05": best_f05,
            "best_precision": best_metrics["precision"],
            "best_recall": best_metrics["recall"],
            "best_fp": best_metrics["fp"],
            "best_fn": best_metrics["fn"],
        }

    eval_results.append(evaluate_probabilities("Model_A_Baseline_Legacy", probs_a))
    eval_results.append(evaluate_probabilities("Model_B_Current_Improved", probs_b))

    trained_models = {}
    # Train each new configuration and evaluate
    for name, train_df in configs.items():
        print(f"\nTraining configuration {name} ({len(train_df):,} pairs)...")
        t0 = time.time()
        X_train_df = compute_features_df(train_df, s1_records, target_records)
        y_train = train_df.label.values.astype(int)
        X_train = X_train_df[FEATURE_COLUMNS].values
        print(f"Features computed in {time.time() - t0:.1f}s. Fitting RandomForest...")

        rf = RandomForestClassifier(**MODEL_PARAMETERS)
        t_fit = time.time()
        rf.fit(X_train, y_train)
        print(f"Fit completed in {time.time() - t_fit:.1f}s.")

        probs = rf.predict_proba(X_val)[:, 1]
        res = evaluate_probabilities(name, probs)
        eval_results.append(res)
        trained_models[name] = rf

        print(f"  Results for {name}:")
        print(f"    @ 0.70: F0.5={res['f05_at_070']:.6f}, Prec={res['prec_at_070']:.4f}, Rec={res['rec_at_070']:.4f}, FP={res['fp_at_070']:,}, FN={res['fn_at_070']:,}")
        print(f"    Best @ {res['best_threshold']}: F0.5={res['best_f05']:.6f}, Prec={res['best_precision']:.4f}, Rec={res['best_recall']:.4f}, FP={res['best_fp']:,}, FN={res['best_fn']:,}")

    # Save the best model
    best_res = max(eval_results, key=lambda x: x["best_f05"])
    print(f"\nBest overall configuration: {best_res['name']} with Best F0.5 = {best_res['best_f05']:.6f} at threshold {best_res['best_threshold']}")

    # Check if a new model beat Model B
    if best_res["name"] in trained_models:
        best_model_obj = trained_models[best_res["name"]]
        joblib.dump({
            "model": best_model_obj,
            "threshold": best_res["best_threshold"],
            "features": FEATURE_COLUMNS,
            "parameters": MODEL_PARAMETERS,
            "config_name": best_res["name"],
        }, "dataset/train/clean_validation_model_candidate_c.pkl")
        print(f"Saved candidate Model C artifact to: dataset/train/clean_validation_model_candidate_c.pkl")

    # Generate Full Report
    print(f"\nWriting evaluation report to {OUTPUT_REPORT}...")
    report_lines = [
        "=" * 80,
        "AMAZON ML CHALLENGE 2026 - HARD NEGATIVE EXPERIMENTS & MODEL COMPARISON",
        "=" * 80,
        f"Execution Time: {time.time() - start_time:.1f}s",
        f"Validation Set: Frozen Clean Validation Candidates (396,850 pairs)",
        f"Validation Ground Truth Matches: 13,324 true positives present",
        "-" * 80,
        f"{'Model Name':<26} {'F0.5 @0.70':<11} {'Prec@0.70':<10} {'FP@0.70':<8} {'Best T':<7} {'Best F0.5':<11} {'Best Prec':<10} {'Best Rec':<10} {'Best FP':<8}",
        "-" * 80,
    ]
    for r in eval_results:
        report_lines.append(
            f"{r['name']:<26} {r['f05_at_070']:<11.6f} {r['prec_at_070']:<10.4f} {r['fp_at_070']:<8} "
            f"{r['best_threshold']:<7.2f} {r['best_f05']:<11.6f} {r['best_precision']:<10.4f} {r['best_recall']:<10.4f} {r['best_fp']:<8}"
        )
    report_lines.extend([
        "-" * 80,
        "\nKEY FINDINGS & DECISION RULE:",
        f"1. Model A (Baseline legacy): F0.5@0.70 = {eval_results[0]['f05_at_070']:.6f}, Best F0.5 = {eval_results[0]['best_f05']:.6f}",
        f"2. Model B (Current improved): F0.5@0.70 = {eval_results[1]['f05_at_070']:.6f}, Best F0.5 = {eval_results[1]['best_f05']:.6f}",
    ])
    for r in eval_results[2:]:
        report_lines.append(f"3. {r['name']}: F0.5@0.70 = {r['f05_at_070']:.6f}, Best F0.5 = {r['best_f05']:.6f} (FP={r['best_fp']:,})")
    
    report_lines.extend([
        "=" * 80,
    ])

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"Report written to {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()
