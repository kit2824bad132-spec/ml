"""
High-Performance Test Inference Pipeline for Amazon ML Challenge 2026
Applies validated LightGBM Model (LGBMClassifier with 11 disambiguation features)
with the optimal threshold T = 0.900 (or 0.975).
Pre-normalizes entity tables once in DuckDB to avoid 73M Python regex calls.
Streams candidate pairs in disk-backed chunks with bounded RAM.
Outputs official submission file: output/matching_results.tsv
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

sys.path.append(os.path.dirname(__file__))
from preprocessing import strip_legal_suffixes, number_match_score

# Input Paths
TEST_DIR = "dataset/test"
S1_FILE = os.path.join(TEST_DIR, "test_source1.tsv")
S2_FILE = os.path.join(TEST_DIR, "test_source2.tsv")
S3_FILE = os.path.join(TEST_DIR, "test_source3.tsv")
CANDIDATE_FILE = os.environ.get("ER_CANDIDATE_FILE", "output/candidate_pairs_final_improved_20260926_230951.tsv")
MODEL_FILE = os.environ.get("ER_MODEL_FILE", "output/model_lightgbm.pkl")
OUTPUT_FILE = os.environ.get("ER_MATCHING_OUTPUT", "output/matching_results.tsv")
OUTPUT_BACKUP = "output/matching_results_improved_model_0975.tsv"
DB_FILE = os.environ.get("ER_INFERENCE_DB", "inference_optimized_run.duckdb")

# Optimal Validated Decision Threshold for F0.5
THRESHOLD = float(os.environ.get("ER_THRESHOLD", "0.900"))

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

def safe_ratio_vec(a, b, scorer):
    scores = process.cpdist(a, b, scorer=scorer, workers=6, dtype=np.float32)
    mask = np.array([bool(x and y) for x, y in zip(a, b)])
    return np.where(mask, scores / 100.0, 0.0)

def main():
    start_total = time.time()
    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026 - OPTIMIZED TEST INFERENCE PIPELINE")
    print("=" * 80)
    print(f"Model File: {MODEL_FILE}")
    print(f"Optimal Threshold: {THRESHOLD}")
    print(f"Candidate File: {CANDIDATE_FILE}")
    print(f"Output File: {OUTPUT_FILE}")
    print(f"DuckDB Scratch Database: {DB_FILE}")

    print("\n[1/4] Loading trained model artifact...")
    model_obj = joblib.load(MODEL_FILE)
    model = model_obj["model"] if isinstance(model_obj, dict) and "model" in model_obj else model_obj
    print("Model loaded successfully.")

    print("\n[2/4] Initializing DuckDB and pre-normalizing entity records...")
    # Clean up old scratch DB if present
    if os.path.exists(DB_FILE):
        try:
            os.remove(DB_FILE)
        except Exception:
            pass

    con = duckdb.connect(database=DB_FILE)
    con.execute("SET memory_limit='10GB'")
    con.execute("SET threads=6")
    con.execute("SET preserve_insertion_order=false")

    # Fast SQL normalization macro
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
    
    valid_matches = defaultdict(list)
    processed_count = 0
    matched_count = 0
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

        # Vectorized RapidFuzz & disambiguation features (workers=6)
        clean_n1 = [strip_legal_suffixes(x) for x in n1]
        clean_n2 = [strip_legal_suffixes(x) for x in n2]
        num_scores = [number_match_score(x, y) for x, y in zip(a1, a2)]

        features = pd.DataFrame()
        features["name_ratio"] = safe_ratio_vec(n1, n2, ratio)
        features["name_token_sort"] = safe_ratio_vec(n1, n2, token_sort_ratio)
        features["name_token_set"] = safe_ratio_vec(n1, n2, token_set_ratio)
        features["clean_name_ratio"] = safe_ratio_vec(clean_n1, clean_n2, ratio)
        features["address_ratio"] = safe_ratio_vec(a1, a2, ratio)
        features["address_token_sort"] = safe_ratio_vec(a1, a2, token_sort_ratio)
        features["address_token_set"] = safe_ratio_vec(a1, a2, token_set_ratio)
        features["number_match_score"] = num_scores
        features["country_match"] = [1 if x == y and x else 0 for x, y in zip(c1, c2)]
        features["name_length_diff"] = [abs(len(x) - len(y)) for x, y in zip(n1, n2)]
        features["address_length_diff"] = [abs(len(x) - len(y)) for x, y in zip(a1, a2)]

        # Enforce feature order
        X = features[FEATURE_COLUMNS].values

        # Predict probabilities
        probs = model.predict_proba(X)[:, 1]

        # Filter by optimal threshold
        mask = probs >= THRESHOLD
        if np.any(mask):
            s1_sub = chunk["source1_entity_id"].values[mask]
            cand_sub = chunk["candidate_entity_id"].values[mask]
            probs_sub = probs[mask]
            for sid, cid, p in zip(s1_sub, cand_sub, probs_sub):
                valid_matches[sid].append((float(p), cid))
            matched_count += int(mask.sum())

        processed_count += len(chunk)
        now = time.time()
        if now - last_log_time >= 30.0 or processed_count >= total_pairs:
            elapsed = now - start_scoring
            speed = processed_count / elapsed if elapsed > 0 else 0
            eta = (total_pairs - processed_count) / speed if speed > 0 else 0
            pct = (processed_count / total_pairs * 100) if total_pairs > 0 else 0
            print(f"Scored {processed_count:>10,} / {total_pairs:,} ({pct:>5.1f}%) | Matches >= {THRESHOLD}: {matched_count:>8,} | Speed: {speed:>6,.0f} pairs/s | ETA: {eta/60:>4.1f}m")
            last_log_time = now

    print(f"\nScoring completed in {(time.time() - start_scoring)/60:.2f} minutes.")
    print(f"Total pairs evaluated: {processed_count:,}")
    print(f"Total matching pairs (prob >= {THRESHOLD}): {matched_count:,}")

    # Read exact original order of Source 1 entity IDs
    print("\nReading exact original order of Source-1 IDs from test_source1.tsv...")
    s1_all_ids = []
    with open(S1_FILE, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0]:
                s1_all_ids.append(parts[0])

    con.close()

    # Clean up scratch database
    if os.path.exists(DB_FILE):
        try:
            os.remove(DB_FILE)
            print(f"Removed temporary scratch database: {DB_FILE}")
        except Exception:
            pass

    print("\nWriting high-precision predictions with Ground-Truth source cardinality capping (S2<=4, S3<=4)...")
    total_written_links = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_all_ids:
            matches = valid_matches.get(sid, [])
            if matches:
                # Sort descending by model probability
                matches.sort(key=lambda x: x[0], reverse=True)
                # Cap to ground-truth cardinality: max 4 for S2, max 4 for S3
                s2_top = [cid for p, cid in matches if cid.startswith("S2-")][:4]
                s3_top = [cid for p, cid in matches if cid.startswith("S3-")][:4]
                selected = s2_top + s3_top
                total_written_links += len(selected)
                f.write(f"{sid}\t{','.join(selected)}\n")
            else:
                f.write(f"{sid}\t\n")

    print(f"Total filtered high-precision links written: {total_written_links:,}")
    import shutil
    shutil.copyfile(OUTPUT_FILE, OUTPUT_BACKUP)
    print(f"Saved dedicated backup: {OUTPUT_BACKUP}")

    total_time = time.time() - start_total
    file_size = os.path.getsize(OUTPUT_FILE)
    print(f"\nDone! Submission file generated: {OUTPUT_FILE} ({file_size:,} bytes)")
    print(f"Total pipeline runtime: {total_time/60:.2f} minutes.")

    # Validation check on final output
    print("\nRunning submission file integrity verification...")
    row_count = 0
    non_empty = 0
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
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
    print("ALL SUBMISSION VALIDATION CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
