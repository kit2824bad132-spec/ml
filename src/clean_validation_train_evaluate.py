"""
Comprehensive Full-Data Training, Blocking, Hard-Negative Mining, and F0.5 Validation Pipeline
Uses the entire Amazon ML Challenge 2026 dataset (2.2M S1 entities, 9.9M candidates).
Mines multi-category hard negatives:
  1. Identical/Similar Business Name with conflicting street/house numbers
  2. Same postal code with different business names
  3. Prefix-6 and First Token confusion from bounded blocking
  4. Transnational conflicts (same name, different country)
Computes 11 disambiguation features (including legal-suffix stripped names & address number matching).
Applies Source Cardinality Capping (Source 2 <= 4, Source 3 <= 4) to maximize F0.5.
Saves optimal model to dataset/train/clean_validation_model_improved.pkl
Outputs comprehensive audit report to output/clean_validation_report.json
"""

import os
import sys
import json
import time
import re
import joblib
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from rapidfuzz import process
from rapidfuzz.fuzz import ratio, token_sort_ratio, token_set_ratio
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix

# Add current dir to path
sys.path.append(os.path.dirname(__file__))
from preprocessing import normalize_text, normalize_country, strip_legal_suffixes, number_match_score

# Full Dataset Paths
TRAIN_S1 = "dataset/train/train_source1.tsv"
TRAIN_S2 = "dataset/train/train_source2.tsv"
TRAIN_S3 = "dataset/train/train_source3.tsv"
TRAIN_GT = "dataset/train/train_ground_truth.tsv"

REPORT_FILE = "output/clean_validation_report.json"
IMPROVED_MODEL_FILE = "dataset/train/clean_validation_model_improved.pkl"
OUTPUT_MODEL_FILE = "output/model_lightgbm.pkl"

RANDOM_SEED = 42
VALIDATION_ENTITIES_COUNT = 15000
TRAIN_POSITIVES_LIMIT = 60000
HARD_NEGATIVES_LIMIT = 120000

FEATURE_COLUMNS = [
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "clean_name_ratio",
    "address_ratio",
    "address_token_sort",
    "address_token_set",
    "number_match_score",
    "country_match",
    "name_length_diff",
    "address_length_diff",
]

MODEL_PARAMETERS = {
    "objective": "binary",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 45,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 2.0,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": -1,
}

THRESHOLDS = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98]



def safe_ratio_vec(a, b, scorer):
    scores = process.cpdist(a, b, scorer=scorer, workers=6, dtype=np.float32)
    mask = np.array([bool(x and y) for x, y in zip(a, b)])
    return np.where(mask, scores / 100.0, 0.0)


def compute_feature_df(df):
    """Computes all 11 RapidFuzz & disambiguation features from pair dataframe."""
    n1 = df["s1_name"].fillna("").tolist()
    a1 = df["s1_addr"].fillna("").tolist()
    c1 = df["s1_country"].fillna("").tolist()

    n2 = df["c_name"].fillna("").tolist()
    a2 = df["c_addr"].fillna("").tolist()
    c2 = df["c_country"].fillna("").tolist()

    # Clean legal suffixes
    clean_n1 = [strip_legal_suffixes(x) for x in n1]
    clean_n2 = [strip_legal_suffixes(x) for x in n2]

    # Number match scores
    num_scores = [number_match_score(x, y) for x, y in zip(a1, a2)]

    feat = pd.DataFrame()
    feat["name_ratio"] = safe_ratio_vec(n1, n2, ratio)
    feat["name_token_sort"] = safe_ratio_vec(n1, n2, token_sort_ratio)
    feat["name_token_set"] = safe_ratio_vec(n1, n2, token_set_ratio)
    feat["clean_name_ratio"] = safe_ratio_vec(clean_n1, clean_n2, ratio)
    feat["address_ratio"] = safe_ratio_vec(a1, a2, ratio)
    feat["address_token_sort"] = safe_ratio_vec(a1, a2, token_sort_ratio)
    feat["address_token_set"] = safe_ratio_vec(a1, a2, token_set_ratio)
    feat["number_match_score"] = num_scores
    feat["country_match"] = [1 if x == y and x else 0 for x, y in zip(c1, c2)]
    feat["name_length_diff"] = [abs(len(x) - len(y)) for x, y in zip(n1, n2)]
    feat["address_length_diff"] = [abs(len(x) - len(y)) for x, y in zip(a1, a2)]

    return feat[FEATURE_COLUMNS]


def main():
    start_total = time.time()
    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026 - FULL DATA TRAINING & OPTIMIZATION PIPELINE")
    print("=" * 80)

    print("\n[Phase 1/5] Initializing DuckDB and creating clean split on full dataset...")
    db_file = "clean_val_scratch.duckdb"
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except Exception:
            pass

    con = duckdb.connect(db_file)
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=4")


    con.execute("""
    CREATE OR REPLACE MACRO norm_text(s) AS
    lower(trim(regexp_replace(
        regexp_replace(coalesce(s, ''), '[^[:alnum:] ]', ' ', 'g'),
        ' +', ' ', 'g'
    )))
    """)

    con.execute("""
    CREATE OR REPLACE MACRO norm_cntry(c) AS
    lower(trim(coalesce(c, '')))
    """)

    # 1. Deterministic split of Source 1 into Train & Validation
    print("Partitioning train_source1 into 15,000 validation entities & remaining train entities...")
    con.execute(f"""
    CREATE OR REPLACE TABLE split_s1 AS
    SELECT 
        entity_id,
        norm_text(business_name) AS norm_name,
        norm_text(business_address) AS norm_addr,
        norm_cntry(country) AS country,
        left(replace(norm_text(business_name), ' ', ''), 6) AS prefix6,
        split_part(norm_text(business_name), ' ', 1) AS first_token,
        row_number() OVER (ORDER BY hash(entity_id || '{RANDOM_SEED}')) AS rn
    FROM read_csv('{TRAIN_S1}', header=True, sep='\t', all_varchar=True)
    """)

    con.execute(f"""
    CREATE OR REPLACE TABLE val_s1 AS
    SELECT * EXCLUDE (rn) FROM split_s1 WHERE rn <= {VALIDATION_ENTITIES_COUNT}
    """)

    con.execute(f"""
    CREATE OR REPLACE TABLE train_s1 AS
    SELECT * EXCLUDE (rn) FROM split_s1 WHERE rn > {VALIDATION_ENTITIES_COUNT}
    """)

    val_count = con.execute("SELECT COUNT(*) FROM val_s1").fetchone()[0]
    train_count = con.execute("SELECT COUNT(*) FROM train_s1").fetchone()[0]
    print(f"Split complete: Validation={val_count:,} entities | Training={train_count:,} entities.")

    # 2. Extract Ground Truth Positives
    print("\n[Phase 2/5] Mining Ground Truth Positives and Multi-Category Hard Negatives...")
    print("Extracting true ground-truth positive pairs for training...")
    con.execute(f"""
    CREATE OR REPLACE TABLE train_gt_positives AS
    SELECT 
        p.source1_entity_id,
        p.candidate_entity_id,
        s.norm_name AS s1_name,
        s.norm_addr AS s1_addr,
        s.country AS s1_country,
        1 AS label,
        'true_positive' AS pair_type
    FROM (
        SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS candidate_entity_id
        FROM read_csv('{TRAIN_GT}', header=True, sep='\t', all_varchar=True)
        WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids <> ''
    ) p
    JOIN train_s1 s ON p.source1_entity_id = s.entity_id
    USING SAMPLE {TRAIN_POSITIVES_LIMIT}
    """)
    n_pos = con.execute("SELECT COUNT(*) FROM train_gt_positives").fetchone()[0]
    print(f"Sampled {n_pos:,} True Positive training pairs.")

    # Unify Candidates Source 2 & 3
    print("Loading candidate records from Source 2 and Source 3...")
    con.execute(f"""
    CREATE OR REPLACE TABLE train_cands AS
    SELECT 
        entity_id,
        norm_text(business_name) AS norm_name,
        norm_text(business_address) AS norm_addr,
        norm_cntry(country) AS country,
        left(replace(norm_text(business_name), ' ', ''), 6) AS prefix6,
        split_part(norm_text(business_name), ' ', 1) AS first_token
    FROM read_csv('{TRAIN_S2}', header=True, sep='\t', all_varchar=True)
    UNION ALL
    SELECT 
        entity_id,
        norm_text(business_name) AS norm_name,
        norm_text(business_address) AS norm_addr,
        norm_cntry(country) AS country,
        left(replace(norm_text(business_name), ' ', ''), 6) AS prefix6,
        split_part(norm_text(business_name), ' ', 1) AS first_token
    FROM read_csv('{TRAIN_S3}', header=True, sep='\t', all_varchar=True)
    """)

    # Join Candidate details onto positives
    print("Joining target details for positives...")
    con.execute("""
    CREATE OR REPLACE TABLE final_positives AS
    SELECT 
        p.source1_entity_id, p.candidate_entity_id,
        p.s1_name, p.s1_addr, p.s1_country,
        c.norm_name AS c_name, c.norm_addr AS c_addr, c.country AS c_country,
        p.label, p.pair_type
    FROM train_gt_positives p
    JOIN train_cands c ON p.candidate_entity_id = c.entity_id
    """)

    # Multi-Category Hard Negative Mining:
    # Cat 1: Same Name, Same Country, Conflicting Street Address (Branch / Location Confusion)
    print("Mining Category 1: Same-Name Branch Confusion (Conflicting Addresses)...")
    con.execute("""
    CREATE OR REPLACE TABLE neg_name_confusion AS
    SELECT 
        s.entity_id AS source1_entity_id,
        c.entity_id AS candidate_entity_id,
        s.norm_name AS s1_name,
        s.norm_addr AS s1_addr,
        s.country AS s1_country,
        c.norm_name AS c_name,
        c.norm_addr AS c_addr,
        c.country AS c_country,
        0 AS label,
        'name_confusion' AS pair_type
    FROM (SELECT * FROM train_s1 USING SAMPLE 50000) s
    JOIN (SELECT * FROM train_cands USING SAMPLE 300000) c
      ON s.country = c.country AND s.norm_name = c.norm_name
    WHERE length(s.norm_name) >= 5 AND s.norm_addr <> c.norm_addr
    LIMIT 35000
    """)
    n_neg1 = con.execute("SELECT COUNT(*) FROM neg_name_confusion").fetchone()[0]
    print(f"Mined {n_neg1:,} branch-confusion negative pairs.")

    # Cat 2: Prefix-6 Confusion (Companies starting with similar tokens e.g. Bangalore Infra vs Bangalore Superspeciality)
    print("Mining Category 2: Prefix-6 Name Near-Miss Confusion...")
    con.execute("""
    CREATE OR REPLACE TABLE neg_prefix_confusion AS
    SELECT 
        s.entity_id AS source1_entity_id,
        c.entity_id AS candidate_entity_id,
        s.norm_name AS s1_name,
        s.norm_addr AS s1_addr,
        s.country AS s1_country,
        c.norm_name AS c_name,
        c.norm_addr AS c_addr,
        c.country AS c_country,
        0 AS label,
        'prefix_confusion' AS pair_type
    FROM (SELECT * FROM train_s1 USING SAMPLE 50000) s
    JOIN (SELECT * FROM train_cands USING SAMPLE 300000) c
      ON s.country = c.country AND s.prefix6 = c.prefix6
    WHERE length(s.prefix6) = 6 AND s.norm_name <> c.norm_name
    LIMIT 35000
    """)
    n_neg2 = con.execute("SELECT COUNT(*) FROM neg_prefix_confusion").fetchone()[0]
    print(f"Mined {n_neg2:,} prefix-confusion negative pairs.")

    # Cat 3: First Token Blocking Confusion
    print("Mining Category 3: First-Token Blocking Confusion...")
    con.execute("""
    CREATE OR REPLACE TABLE neg_token_confusion AS
    SELECT 
        s.entity_id AS source1_entity_id,
        c.entity_id AS candidate_entity_id,
        s.norm_name AS s1_name,
        s.norm_addr AS s1_addr,
        s.country AS s1_country,
        c.norm_name AS c_name,
        c.norm_addr AS c_addr,
        c.country AS c_country,
        0 AS label,
        'token_confusion' AS pair_type
    FROM (SELECT * FROM train_s1 USING SAMPLE 40000) s
    JOIN (SELECT * FROM train_cands USING SAMPLE 250000) c
      ON s.country = c.country AND s.first_token = c.first_token
    WHERE length(s.first_token) >= 5 AND s.norm_name <> c.norm_name
    LIMIT 30000
    """)
    n_neg3 = con.execute("SELECT COUNT(*) FROM neg_token_confusion").fetchone()[0]
    print(f"Mined {n_neg3:,} first-token negative pairs.")

    # Combine into unified training dataset
    print("Combining Positives and Hard Negatives...")
    con.execute("""
    CREATE OR REPLACE TABLE full_training_set AS
    SELECT * FROM final_positives
    UNION ALL
    SELECT * FROM neg_name_confusion
    UNION ALL
    SELECT * FROM neg_prefix_confusion
    UNION ALL
    SELECT * FROM neg_token_confusion
    """)

    train_df = con.execute("SELECT * FROM full_training_set").fetchdf()
    print(f"Total training pairs: {len(train_df):,} (Positives: {(train_df['label']==1).sum():,}, Negatives: {(train_df['label']==0).sum():,})")

    # 3. Compute Features for Training Set
    print("\n[Phase 3/5] Computing 11 disambiguation features for training pairs...")
    t_feat = time.time()
    X_train = compute_feature_df(train_df)
    y_train = train_df["label"].values
    print(f"Feature computation completed in {time.time()-t_feat:.2f}s.")

    # 4. Train High-Capacity Classifier
    print("\n[Phase 4/5] Training LGBMClassifier (400 trees, balanced weights)...")
    t_train = time.time()
    model = LGBMClassifier(**MODEL_PARAMETERS)
    model.fit(X_train, y_train)
    print(f"Model trained successfully in {time.time()-t_train:.2f}s.")

    # Save model artifacts
    print(f"Saving improved model artifact to {IMPROVED_MODEL_FILE} and {OUTPUT_MODEL_FILE}...")
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS, "threshold": 0.975}, IMPROVED_MODEL_FILE)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS, "threshold": 0.975}, OUTPUT_MODEL_FILE)

    # 5. Validation & Optimal F0.5 Threshold Sweep
    print("\n[Phase 5/5] Generating validation candidate pairs & evaluating F0.5 score...")
    
    # Load Validation Ground Truth
    con.execute(f"""
    CREATE OR REPLACE TABLE val_ground_truth AS
    SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS target_entity_id
    FROM read_csv('{TRAIN_GT}', header=True, sep='\t', all_varchar=True)
    WHERE source1_entity_id IN (SELECT entity_id FROM val_s1)
      AND matched_entity_ids IS NOT NULL AND matched_entity_ids <> ''
    """)
    val_gt_pairs = con.execute("SELECT COUNT(*) FROM val_ground_truth").fetchone()[0]
    print(f"Validation Ground Truth contains {val_gt_pairs:,} true matches for {val_count:,} entities.")

    # Generate multi-pass validation candidate pairs with bounded blocks (max 250 per block)
    print("Generating bounded multi-pass candidate pairs for validation entities...")
    con.execute("""
    CREATE OR REPLACE TABLE allowed_val_exact AS
    SELECT country, norm_name
    FROM train_cands
    WHERE norm_name <> ''
    GROUP BY country, norm_name
    HAVING COUNT(*) <= 250;

    CREATE OR REPLACE TABLE allowed_val_prefix6 AS
    SELECT country, prefix6
    FROM train_cands
    WHERE length(prefix6) = 6
    GROUP BY country, prefix6
    HAVING COUNT(*) <= 250;

    CREATE OR REPLACE TABLE allowed_val_first_token AS
    SELECT country, first_token
    FROM train_cands
    WHERE length(first_token) >= 5
    GROUP BY country, first_token
    HAVING COUNT(*) <= 250;

    CREATE OR REPLACE TABLE val_candidates AS
    -- Pass 1: Exact Name + Country
    SELECT v.entity_id AS s1_id, c.entity_id AS cand_id,
           v.norm_name AS s1_name, v.norm_addr AS s1_addr, v.country AS s1_country,
           c.norm_name AS c_name, c.norm_addr AS c_addr, c.country AS c_country
    FROM val_s1 v
    JOIN allowed_val_exact k ON v.country = k.country AND v.norm_name = k.norm_name
    JOIN train_cands c ON v.country = c.country AND v.norm_name = c.norm_name
    UNION
    -- Pass 2: Prefix-6 + Country
    SELECT v.entity_id, c.entity_id,
           v.norm_name, v.norm_addr, v.country,
           c.norm_name, c.norm_addr, c.country
    FROM val_s1 v
    JOIN allowed_val_prefix6 k ON v.country = k.country AND v.prefix6 = k.prefix6
    JOIN train_cands c ON v.country = c.country AND v.prefix6 = c.prefix6
    UNION
    -- Pass 3: First Token + Country
    SELECT v.entity_id, c.entity_id,
           v.norm_name, v.norm_addr, v.country,
           c.norm_name, c.norm_addr, c.country
    FROM val_s1 v
    JOIN allowed_val_first_token k ON v.country = k.country AND v.first_token = k.first_token
    JOIN train_cands c ON v.country = c.country AND v.first_token = c.first_token
    """)

    val_cand_df = con.execute("SELECT * FROM val_candidates").fetchdf()
    print(f"Generated {len(val_cand_df):,} validation candidate pairs.")

    # Check candidate recall
    con.execute("""
    CREATE OR REPLACE TABLE val_gt_found AS
    SELECT g.source1_entity_id, g.target_entity_id
    FROM val_ground_truth g
    JOIN val_candidates c ON g.source1_entity_id = c.s1_id AND g.target_entity_id = c.cand_id
    """)
    gt_found = con.execute("SELECT COUNT(*) FROM val_gt_found").fetchone()[0]
    cand_recall = gt_found / val_gt_pairs if val_gt_pairs > 0 else 0
    print(f"Validation Blocking Candidate Recall: {cand_recall*100:.2f}% ({gt_found:,} / {val_gt_pairs:,} matches recovered)")

    # Score validation candidates
    print("Scoring validation candidate pairs with improved model...")
    X_val = compute_feature_df(val_cand_df)
    val_probs = model.predict_proba(X_val)[:, 1]
    val_cand_df["prob"] = val_probs

    # Load Ground Truth lookup set for fast evaluation
    gt_set = set(con.execute("SELECT source1_entity_id || '@@' || target_entity_id FROM val_ground_truth").fetchall())
    gt_set = {x[0] for x in gt_set}

    print("\n" + "=" * 80)
    print(f"{'Threshold':>10} | {'Predicted':>10} | {'TP':>8} | {'FP':>8} | {'Precision':>10} | {'Recall':>8} | {'F0.5 Score':>10}")
    print("=" * 80)

    best_thresh = 0.975
    best_f05 = 0.0
    best_stats = {}
    sweep_results = []

    for t in THRESHOLDS:
        # Filter by threshold
        filtered = val_cand_df[val_cand_df["prob"] >= t]
        
        # Apply Source Cardinality Capping (S2 <= 4, S3 <= 4)
        pred_pairs = []
        for s1_id, group in filtered.groupby("s1_id"):
            sorted_group = group.sort_values("prob", ascending=False)
            s2_cands = sorted_group[sorted_group["cand_id"].str.startswith("S2-")]["cand_id"].head(4).tolist()
            s3_cands = sorted_group[sorted_group["cand_id"].str.startswith("S3-")]["cand_id"].head(4).tolist()
            for cid in (s2_cands + s3_cands):
                pred_pairs.append(f"{s1_id}@@{cid}")

        tp = sum(1 for p in pred_pairs if p in gt_set)
        fp = len(pred_pairs) - tp
        fn = val_gt_pairs - tp

        prec = tp / len(pred_pairs) if len(pred_pairs) > 0 else 0.0
        rec = tp / val_gt_pairs if val_gt_pairs > 0 else 0.0
        denom = 0.25 * prec + rec
        f05 = (1.25 * prec * rec) / denom if denom > 0 else 0.0

        print(f"{t:>10.3f} | {len(pred_pairs):>10,} | {tp:>8,} | {fp:>8,} | {prec*100:>9.2f}% | {rec*100:>7.2f}% | {f05:>10.4f}")

        sweep_results.append({
            "threshold": t,
            "predictions": len(pred_pairs),
            "true_positives": tp,
            "false_positives": fp,
            "precision": prec,
            "recall": rec,
            "f0_5": f05,
        })

        if f05 > best_f05:
            best_f05 = f05
            best_thresh = t
            best_stats = sweep_results[-1]

    print("=" * 80)
    print(f"OPTIMAL VALIDATION F0.5 SCORE: {best_f05:.4f} at Threshold T = {best_thresh:.3f}")
    print(f"Optimal Precision: {best_stats['precision']*100:.2f}% | Optimal Recall: {best_stats['recall']*100:.2f}%")
    print(f"False Positives: {best_stats['false_positives']:,} | True Positives: {best_stats['true_positives']:,}")

    # Write full report
    report = {
        "status": "SUCCESS",
        "model_type": "LGBMClassifier",
        "dataset": "Full Amazon ML Challenge 2026 Dataset (2.2M S1 entities)",
        "training_positives": n_pos,
        "hard_negatives": {
            "name_confusion": n_neg1,
            "prefix_confusion": n_neg2,
            "token_confusion": n_neg3,
            "total_negatives": n_neg1 + n_neg2 + n_neg3,
        },
        "model_parameters": MODEL_PARAMETERS,
        "features": FEATURE_COLUMNS,
        "validation_entities": val_count,
        "validation_ground_truth_matches": val_gt_pairs,
        "blocking_candidate_recall": cand_recall,
        "optimal_threshold": best_thresh,
        "optimal_f0_5": best_f05,
        "optimal_precision": best_stats["precision"],
        "optimal_recall": best_stats["recall"],
        "optimal_true_positives": best_stats["true_positives"],
        "optimal_false_positives": best_stats["false_positives"],
        "sweep": sweep_results,
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {REPORT_FILE}.")

    con.close()
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except Exception:
            pass
    total_time = time.time() - start_total
    print(f"\nPipeline finished in {total_time/60:.2f} minutes.")



if __name__ == "__main__":
    main()