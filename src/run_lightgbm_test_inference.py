"""
High-Performance LightGBM Test Inference Pipeline for Amazon ML Challenge 2026
Applies production LightGBM model (output/model_lightgbm.pkl)
with optimized threshold T = 0.9995 (and tracks baseline T = 0.760).
Pre-normalizes entity tables in DuckDB to maximize streaming throughput.
Generates output/matching_results.tsv and output/matching_results_lightgbm_optimized.tsv.
"""

import os
import sys
import time
import joblib
import duckdb
import numpy as np
import pandas as pd
from collections import defaultdict
from rapidfuzz import process
from rapidfuzz.fuzz import ratio, token_sort_ratio, token_set_ratio

# Input Paths
TEST_DIR = "dataset/test"
S1_FILE = os.path.join(TEST_DIR, "test_source1.tsv")
S2_FILE = os.path.join(TEST_DIR, "test_source2.tsv")
S3_FILE = os.path.join(TEST_DIR, "test_source3.tsv")
CANDIDATE_FILE = os.environ.get("ER_CANDIDATE_FILE", "output/candidate_pairs_lightgbm.tsv")
MODEL_FILE = os.environ.get("ER_MODEL_FILE", "output/model_lightgbm.pkl")
OUTPUT_FILE_OPTIMIZED = "output/matching_results_lightgbm_optimized.tsv"
OUTPUT_FILE_FINAL = "output/matching_results.tsv"
DB_FILE = "lightgbm_inference_temp.duckdb"

# Validated Optimal Threshold for LightGBM
THRESHOLD_OPTIMIZED = float(os.environ.get("ER_THRESHOLD", "0.9995"))
THRESHOLD_BASELINE = 0.760

def safe_ratio_vec(a, b, scorer):
    scores = process.cpdist(a, b, scorer=scorer, workers=6, dtype=np.float32)
    mask = np.array([bool(x and y) for x, y in zip(a, b)])
    return np.where(mask, scores / 100.0, 0.0)

def main():
    start_total = time.time()
    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026 - LIGHTGBM TEST INFERENCE PIPELINE")
    print("=" * 80)
    print(f"Model File: {MODEL_FILE}")
    print(f"Candidate File: {CANDIDATE_FILE}")
    print(f"Primary Optimized Threshold: {THRESHOLD_OPTIMIZED}")
    print(f"Baseline Threshold: {THRESHOLD_BASELINE}")
    print(f"Target Submission File: {OUTPUT_FILE_FINAL}")

    print("\n[1/4] Loading LightGBM model artifact...")
    model_obj = joblib.load(MODEL_FILE)
    model = model_obj["model"]
    feature_cols = model_obj["feature_columns"]
    print(f"LightGBM model loaded successfully. Features: {feature_cols}")

    print("\n[2/4] Initializing DuckDB and pre-normalizing entity tables...")
    if os.path.exists(DB_FILE):
        try:
            os.remove(DB_FILE)
        except Exception:
            pass

    con = duckdb.connect(database=DB_FILE)
    con.execute("SET memory_limit='10GB'")
    con.execute("SET threads=6")
    con.execute("SET preserve_insertion_order=false")

    con.execute("""
    CREATE OR REPLACE MACRO normalize_str(s) AS
    lower(trim(regexp_replace(
        regexp_replace(coalesce(s, ''), '[^[:alnum:] ]', ' ', 'g'),
        ' +', ' ', 'g'
    )))
    """)

    con.execute("""
    CREATE OR REPLACE MACRO normalize_cntry(c) AS
    lower(trim(coalesce(c, '')))
    """)

    print("Loading and pre-normalizing Source 1...")
    con.execute(f"""
    CREATE OR REPLACE TABLE s1 AS
    SELECT 
        entity_id,
        normalize_str(business_name) AS norm_name,
        normalize_str(business_address) AS norm_addr,
        normalize_cntry(country) AS norm_country
    FROM read_csv('{S1_FILE}', header=True, sep='\\t', all_varchar=True)
    """)
    s1_count = con.execute("SELECT count(*) FROM s1").fetchone()[0]
    print(f"Source 1 pre-normalized: {s1_count:,} records.")

    print("Loading and pre-normalizing Source 2 and Source 3...")
    con.execute(f"""
    CREATE OR REPLACE TABLE candidates AS
    SELECT 
        entity_id,
        normalize_str(business_name) AS norm_name,
        normalize_str(business_address) AS norm_addr,
        normalize_cntry(country) AS norm_country
    FROM read_csv('{S2_FILE}', header=True, sep='\\t', all_varchar=True)
    UNION ALL
    SELECT 
        entity_id,
        normalize_str(business_name) AS norm_name,
        normalize_str(business_address) AS norm_addr,
        normalize_cntry(country) AS norm_country
    FROM read_csv('{S3_FILE}', header=True, sep='\\t', all_varchar=True)
    """)
    cands_count = con.execute("SELECT count(*) FROM candidates").fetchone()[0]
    print(f"Candidates pre-normalized: {cands_count:,} records.")

    print("\n[3/4] Unnesting candidate pairs from TSV...")
    con.execute(f"""
    CREATE OR REPLACE TABLE unnested_pairs AS
    SELECT 
        source1_entity_id, 
        unnest(string_split(candidate_entity_ids, ',')) AS candidate_entity_id
    FROM read_csv('{CANDIDATE_FILE}', header=True, sep='\\t', all_varchar=True)
    WHERE candidate_entity_ids IS NOT NULL AND candidate_entity_ids != ''
    """)
    total_pairs = con.execute("SELECT count(*) FROM unnested_pairs").fetchone()[0]
    print(f"Total Candidate Pairs to score: {total_pairs:,}")

    query = """
    SELECT 
        p.source1_entity_id,
        p.candidate_entity_id,
        s1.norm_name AS n1,
        s1.norm_addr AS a1,
        s1.norm_country AS c1,
        c.norm_name AS n2,
        c.norm_addr AS a2,
        c.norm_country AS c2
    FROM unnested_pairs p
    JOIN s1 ON p.source1_entity_id = s1.entity_id
    JOIN candidates c ON p.candidate_entity_id = c.entity_id
    """

    print("\n[4/4] Streaming candidate pairs & computing features in chunks...")
    res = con.execute(query)

    matches_optimized = defaultdict(list)
    matches_baseline = defaultdict(list)
    processed_count = 0
    opt_match_count = 0
    base_match_count = 0
    vectors_per_chunk = 50  # ~102,400 rows per chunk
    start_scoring = time.time()
    last_log_time = start_scoring

    while True:
        chunk = res.fetch_df_chunk(vectors_per_chunk)
        if chunk.empty:
            break

        chunk.fillna("", inplace=True)
        n1 = chunk["n1"].tolist()
        a1 = chunk["a1"].tolist()
        c1 = chunk["c1"].tolist()

        n2 = chunk["n2"].tolist()
        a2 = chunk["a2"].tolist()
        c2 = chunk["c2"].tolist()

        # Vectorized RapidFuzz features (workers=6)
        features = pd.DataFrame()
        features["name_ratio"] = safe_ratio_vec(n1, n2, ratio)
        features["name_token_sort"] = safe_ratio_vec(n1, n2, token_sort_ratio)
        features["name_token_set"] = safe_ratio_vec(n1, n2, token_set_ratio)
        features["address_ratio"] = safe_ratio_vec(a1, a2, ratio)
        features["address_token_sort"] = safe_ratio_vec(a1, a2, token_sort_ratio)
        features["address_token_set"] = safe_ratio_vec(a1, a2, token_set_ratio)
        features["country_match"] = (np.array(c1) == np.array(c2)).astype(int)
        features["name_length_diff"] = np.abs(np.array([len(x) for x in n1]) - np.array([len(x) for x in n2]))
        features["address_length_diff"] = np.abs(np.array([len(x) for x in a1]) - np.array([len(x) for x in a2]))

        X = features[feature_cols].values

        # Predict with LightGBM
        probs = model.predict_proba(X)[:, 1]

        # 1. Track Optimized Matches (T >= 0.9995)
        mask_opt = probs >= THRESHOLD_OPTIMIZED
        if np.any(mask_opt):
            s1_sub = chunk["source1_entity_id"].values[mask_opt]
            cand_sub = chunk["candidate_entity_id"].values[mask_opt]
            for sid, cid in zip(s1_sub, cand_sub):
                matches_optimized[sid].append(cid)
            opt_match_count += int(mask_opt.sum())

        # 2. Track Baseline Matches (T >= 0.760)
        mask_base = probs >= THRESHOLD_BASELINE
        if np.any(mask_base):
            s1_sub = chunk["source1_entity_id"].values[mask_base]
            cand_sub = chunk["candidate_entity_id"].values[mask_base]
            for sid, cid in zip(s1_sub, cand_sub):
                matches_baseline[sid].append(cid)
            base_match_count += int(mask_base.sum())

        processed_count += len(chunk)
        now = time.time()
        if now - last_log_time >= 30.0 or processed_count >= total_pairs:
            elapsed = now - start_scoring
            speed = processed_count / elapsed if elapsed > 0 else 0
            eta = (total_pairs - processed_count) / speed if speed > 0 else 0
            pct = (processed_count / total_pairs * 100) if total_pairs > 0 else 0
            print(f"Scored {processed_count:>10,} / {total_pairs:,} ({pct:>5.1f}%) | Opt Matches: {opt_match_count:>8,} | Base Matches: {base_match_count:>8,} | Speed: {speed:>6,.0f} pairs/s | ETA: {eta/60:>4.1f}m")
            last_log_time = now

    print(f"\nScoring completed in {(time.time() - start_scoring)/60:.2f} minutes.")
    print(f"Total pairs evaluated: {processed_count:,}")
    print(f"Total optimized matches (prob >= {THRESHOLD_OPTIMIZED}): {opt_match_count:,}")
    print(f"Total baseline matches (prob >= {THRESHOLD_BASELINE}): {base_match_count:,}")

    # Read original Source 1 entity IDs in exact file order
    print("\nReading exact original order of Source-1 IDs from test_source1.tsv...")
    s1_all_ids = []
    with open(S1_FILE, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0]:
                s1_all_ids.append(parts[0])

    con.close()
    if os.path.exists(DB_FILE):
        try:
            os.remove(DB_FILE)
            print(f"Removed temporary scratch database: {DB_FILE}")
        except Exception:
            pass

    # Write 1: Optimized Result File
    print(f"\nWriting primary optimized submission file: {OUTPUT_FILE_OPTIMIZED}...")
    with open(OUTPUT_FILE_OPTIMIZED, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_all_ids:
            matches = matches_optimized.get(sid, [])
            f.write(f"{sid}\t{','.join(matches)}\n")

    # Write 2: Also write as standard output/matching_results.tsv
    print(f"Writing official submission file: {OUTPUT_FILE_FINAL}...")
    with open(OUTPUT_FILE_FINAL, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_all_ids:
            matches = matches_optimized.get(sid, [])
            f.write(f"{sid}\t{','.join(matches)}\n")

    # Write 3: Baseline Comparison File (threshold 0.76)
    baseline_out = "output/matching_results_lightgbm_076.tsv"
    print(f"Writing baseline reference file: {baseline_out}...")
    with open(baseline_out, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_all_ids:
            matches = matches_baseline.get(sid, [])
            f.write(f"{sid}\t{','.join(matches)}\n")

    total_time = time.time() - start_total
    print(f"\nAll files successfully written! Total runtime: {total_time/60:.2f} minutes.")
    print(f"  Optimized: {OUTPUT_FILE_FINAL} ({os.path.getsize(OUTPUT_FILE_FINAL):,} bytes)")
    print(f"  Optimized Backup: {OUTPUT_FILE_OPTIMIZED} ({os.path.getsize(OUTPUT_FILE_OPTIMIZED):,} bytes)")
    print(f"  Baseline 0.76: {baseline_out} ({os.path.getsize(baseline_out):,} bytes)")

    # Integrity verification
    print("\nRunning submission integrity verification...")
    row_count = 0
    non_empty = 0
    with open(OUTPUT_FILE_FINAL, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        assert header == "source1_entity_id\tmatched_entity_ids", f"Invalid header: {header}"
        for line in f:
            row_count += 1
            parts = line.strip().split("\t")
            if len(parts) > 1 and parts[1]:
                non_empty += 1

    print(f"Verified row count: {row_count:,} (Expected: {len(s1_all_ids):,})")
    print(f"Source-1 entities with matches: {non_empty:,} ({non_empty/row_count*100:.2f}%)")
    print(f"Source-1 entities without matches (singletons): {row_count - non_empty:,} ({(row_count - non_empty)/row_count*100:.2f}%)")
    assert row_count == len(s1_all_ids), "Row count mismatch!"
    print("ALL SUBMISSION VERIFICATION CHECKS PASSED!")

if __name__ == "__main__":
    main()
