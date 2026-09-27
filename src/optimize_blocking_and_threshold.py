"""
Optimize Best Threshold and Blocking Strategy
Tests Fine-Grained Threshold Grid and Source-Aware Matching Logic
on both Selective (412k) and Strat4 (741k) Blocking Unions.
"""

import time
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
MODEL_FILE = "dataset/train/clean_validation_model_improved.pkl"

CANDIDATE_FILES = {
    "Selective_6Pass (412k pairs)": "dataset/train/clean_blocking_candidates_selective.tsv",
    "Enhanced_Strat4 (741k pairs)": "dataset/train/clean_blocking_candidates_strat4.tsv",
}

OUTPUT_REPORT = "output/improved_threshold_and_blocking_report.txt"

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

FINE_THRESHOLDS = [
    0.930, 0.940, 0.945, 0.950, 0.955, 0.960, 0.965, 0.968,
    0.970, 0.972, 0.974, 0.975, 0.976, 0.978, 0.980, 0.982, 0.985
]

def safe_ratio_vec(a, b, scorer):
    scores = process.cpdist(a, b, scorer=scorer, workers=6, dtype=np.float32)
    mask = np.array([bool(x and y) for x, y in zip(a, b)])
    return np.where(mask, scores / 100.0, 0.0)

def main():
    start_time = time.time()
    print("Loading model from:", MODEL_FILE)
    model_obj = joblib.load(MODEL_FILE)
    model = model_obj["model"]

    print("Loading validation Source-1 IDs...")
    split_df = pd.read_csv(SPLIT_FILE, sep="\t", dtype=str)
    val_s1_ids = set(split_df.loc[split_df["split"] == "validation", "entity_id"])
    total_val_s1 = len(val_s1_ids)

    print("Loading validation ground truth matches...")
    val_truth_dict = defaultdict(set)
    total_gt = 0
    for chunk in pd.read_csv(GT_FILE, sep="\t", dtype=str, keep_default_na=False, chunksize=200000):
        sub = chunk[chunk.source1_entity_id.isin(val_s1_ids)]
        for row in sub.itertuples(index=False):
            for m in row.matched_entity_ids.split(","):
                m = m.strip()
                if m:
                    val_truth_dict[row.source1_entity_id].add(m)
                    total_gt += 1
    true_pairs_set = {(sid, cid) for sid, cids in val_truth_dict.items() for cid in cids}
    print(f"Validation Ground Truth Matches: {total_gt:,}")

    print("Loading entity records...")
    s1_df = pd.read_csv(S1_FILE, sep="\t", dtype=str, keep_default_na=False)
    s1_df = s1_df[s1_df.entity_id.isin(val_s1_ids)].copy()
    s1_records = {
        row.entity_id: (normalize_text(row.business_name), normalize_text(row.business_address), normalize_country(row.country))
        for row in s1_df.itertuples(index=False)
    }

    target_df = pd.concat([
        pd.read_csv(S2_FILE, sep="\t", dtype=str, keep_default_na=False),
        pd.read_csv(S3_FILE, sep="\t", dtype=str, keep_default_na=False),
    ], ignore_index=True).drop_duplicates("entity_id")
    target_records = {
        row.entity_id: (normalize_text(row.business_name), normalize_text(row.business_address), normalize_country(row.country))
        for row in target_df.itertuples(index=False)
    }

    all_sweep_results = {}

    for cand_name, cand_path in CANDIDATE_FILES.items():
        print(f"\n========================================================")
        print(f"EVALUATING CANDIDATE SET: {cand_name}")
        print(f"Loading candidate pairs from: {cand_path}")
        cand_df = pd.read_csv(cand_path, sep="\t", dtype=str)
        cand_df = cand_df[cand_df.source1_entity_id.isin(val_s1_ids)].copy()
        n_cands = len(cand_df)

        cand_pairs = set(zip(cand_df.source1_entity_id, cand_df.candidate_entity_id))
        recov = len(true_pairs_set & cand_pairs)
        cand_rec = recov / total_gt
        print(f"Total Candidate Pairs: {n_cands:,}")
        print(f"Candidate Recall: {cand_rec:.6f} ({recov:,} / {total_gt:,} matches)")

        # Compute features & predict probabilities
        print("Computing features and model probabilities...")
        chunk_size = 50000
        all_probs = []
        for i in range(0, n_cands, chunk_size):
            chunk = cand_df.iloc[i:i+chunk_size]
            n1, a1, c1 = zip(*[s1_records[sid] for sid in chunk.source1_entity_id])
            n2, a2, c2 = zip(*[target_records[cid] for cid in chunk.candidate_entity_id])

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

            probs = model.predict_proba(feats[FEATURE_COLUMNS].values)[:, 1]
            all_probs.extend(probs)

        cand_df["prob"] = all_probs
        cand_df["is_true"] = [1 if p in true_pairs_set else 0 for p in zip(cand_df.source1_entity_id, cand_df.candidate_entity_id)]

        # 1. Pure fine-grained threshold sweep
        print(f"\n--- Fine-Grained Threshold Sweep for {cand_name} ---")
        sweep_data = []
        for t in FINE_THRESHOLDS:
            preds = (cand_df["prob"] >= t).astype(int)
            tp = int(((preds == 1) & (cand_df["is_true"] == 1)).sum())
            fp = int(((preds == 1) & (cand_df["is_true"] == 0)).sum())
            fn_cand = int(((preds == 0) & (cand_df["is_true"] == 1)).sum())
            fn_block = total_gt - recov
            total_fn = fn_cand + fn_block

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec_cand = tp / recov if recov > 0 else 0.0
            rec_e2e = tp / total_gt
            f05_cand = (1.25 * prec * rec_cand) / (0.25 * prec + rec_cand) if (0.25 * prec + rec_cand) > 0 else 0.0
            f05_e2e = (1.25 * prec * rec_e2e) / (0.25 * prec + rec_e2e) if (0.25 * prec + rec_e2e) > 0 else 0.0

            sweep_data.append({
                "threshold": t,
                "precision": prec,
                "cand_recall": rec_cand,
                "cand_f05": f05_cand,
                "e2e_recall": rec_e2e,
                "e2e_f05": f05_e2e,
                "tp": tp,
                "fp": fp,
                "fn_cand": fn_cand,
                "fn_block": fn_block,
                "total_fn": total_fn,
                "pred_count": tp + fp,
            })
            print(f"T={t:.3f} | Prec={prec:.4f} | E2E_Rec={rec_e2e:.4f} | E2E_F0.5={f05_e2e:.6f} | Cand_F0.5={f05_cand:.6f} | FP={fp:>5,} | TP={tp:>6,}")

        # 2. Source-aware matching resolution experiment
        # Can we filter low-confidence secondary candidates if an entity has a much higher confidence candidate?
        # Target sources: S2 (prefix 'S2-') vs S3 (prefix 'S3-')
        # In ground truth, an S1 usually matches at most 1 in S2 and 1 in S3 (e.g. 95% of matches).
        # Let's test a source-aware rule: For each S1 and target source (S2/S3), keep candidate if prob >= T,
        # but if multiple candidates exceed T, keep top-k or within margin delta of top prob.
        cand_df["target_source"] = cand_df.candidate_entity_id.apply(lambda cid: cid.split("-")[0] if "-" in cid else "other")
        
        # Test Margin / Top-1 per source at best threshold region
        print("\n--- Testing Source-Aware Matching Strategy ---")
        source_aware_results = []
        for t in [0.950, 0.960, 0.965, 0.970, 0.972, 0.974, 0.975]:
            sub = cand_df[cand_df.prob >= t].copy()
            # Group by source1_id and target_source, rank by prob descending
            sub["rank_in_source"] = sub.groupby(["source1_entity_id", "target_source"])["prob"].rank(ascending=False, method="first")
            
            # Policy A: Pure threshold (no limit)
            tp_a = int(sub.is_true.sum())
            fp_a = int((sub.is_true == 0).sum())
            p_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0.0
            r_a = tp_a / total_gt
            f05_a = (1.25 * p_a * r_a) / (0.25 * p_a + r_a) if (0.25 * p_a + r_a) > 0 else 0.0

            # Policy B: Top-1 per target source (keep highest prob candidate per S2 and per S3)
            sub_top1 = sub[sub.rank_in_source == 1]
            tp_b = int(sub_top1.is_true.sum())
            fp_b = int((sub_top1.is_true == 0).sum())
            p_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0
            r_b = tp_b / total_gt
            f05_b = (1.25 * p_b * r_b) / (0.25 * p_b + r_b) if (0.25 * p_b + r_b) > 0 else 0.0

            # Policy C: Top-2 per target source
            sub_top2 = sub[sub.rank_in_source <= 2]
            tp_c = int(sub_top2.is_true.sum())
            fp_c = int((sub_top2.is_true == 0).sum())
            p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
            r_c = tp_c / total_gt
            f05_c = (1.25 * p_c * r_c) / (0.25 * p_c + r_c) if (0.25 * p_c + r_c) > 0 else 0.0

            print(f"T={t:.3f} | Pure: F0.5={f05_a:.6f}, Prec={p_a:.4f}, Rec={r_a:.4f}, FP={fp_a:,}")
            print(f"         | Top1: F0.5={f05_b:.6f}, Prec={p_b:.4f}, Rec={r_b:.4f}, FP={fp_b:,}")
            print(f"         | Top2: F0.5={f05_c:.6f}, Prec={p_c:.4f}, Rec={r_c:.4f}, FP={fp_c:,}")
            source_aware_results.append({
                "threshold": t,
                "pure_f05": f05_a, "pure_prec": p_a, "pure_rec": r_a, "pure_fp": fp_a,
                "top1_f05": f05_b, "top1_prec": p_b, "top1_rec": r_b, "top1_fp": fp_b,
                "top2_f05": f05_c, "top2_prec": p_c, "top2_rec": r_c, "top2_fp": fp_c,
            })

        all_sweep_results[cand_name] = {
            "candidate_recall": cand_rec,
            "total_candidates": n_cands,
            "sweep_data": sweep_data,
            "source_aware_results": source_aware_results,
        }

    # Write Complete Improved Optimization Report
    print(f"\nWriting comprehensive report to {OUTPUT_REPORT}...")
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("AMAZON ML CHALLENGE 2026 - OPTIMIZED THRESHOLD & BLOCKING STRATEGY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Validation S1 Entities: {total_val_s1:,}\n")
        f.write(f"Total Validation Ground Truth Matches: {total_gt:,}\n\n")

        for c_name, data in all_sweep_results.items():
            f.write("-" * 80 + "\n")
            f.write(f"STRATEGY: {c_name}\n")
            f.write(f"Total Candidates: {data['total_candidates']:,}\n")
            f.write(f"Candidate Recall: {data['candidate_recall']:.6f} ({data['candidate_recall']*100:.2f}%)\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'T':<7} {'Prec':<10} {'Cand_Rec':<10} {'Cand_F0.5':<12} {'E2E_Rec':<10} {'E2E_F0.5':<12} {'FP':<8} {'TP':<8}\n")
            f.write("-" * 80 + "\n")
            for r in data["sweep_data"]:
                f.write(f"{r['threshold']:<7.3f} {r['precision']:<10.4f} {r['cand_recall']:<10.4f} {r['cand_f05']:<12.6f} {r['e2e_recall']:<10.4f} {r['e2e_f05']:<12.6f} {r['fp']:<8} {r['tp']:<8}\n")

            f.write("\nSOURCE-AWARE MATCHING COMPARISON:\n")
            f.write(f"{'T':<7} {'Pure_F0.5':<12} {'Pure_Prec':<10} {'Top1_F0.5':<12} {'Top1_Prec':<10} {'Top2_F0.5':<12} {'Top2_Prec':<10}\n")
            for sr in data["source_aware_results"]:
                f.write(f"{sr['threshold']:<7.3f} {sr['pure_f05']:<12.6f} {sr['pure_prec']:<10.4f} {sr['top1_f05']:<12.6f} {sr['top1_prec']:<10.4f} {sr['top2_f05']:<12.6f} {sr['top2_prec']:<10.4f}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")

    print(f"Report successfully saved to {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()
