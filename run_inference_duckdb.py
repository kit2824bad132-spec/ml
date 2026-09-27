import duckdb
import pandas as pd
import numpy as np
import joblib
import time
import os
import sys
from collections import defaultdict

# Append src to path so we can import preprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import normalize_text, normalize_country
from rapidfuzz import process
from rapidfuzz.fuzz import ratio, token_sort_ratio, token_set_ratio

CANDIDATE_FILE = os.environ.get("ER_CANDIDATE_FILE", "output/candidate_pairs.tsv")
S1_TEST_FILE = "dataset/test/test_source1.tsv"
S2_TEST_FILE = "dataset/test/test_source2.tsv"
S3_TEST_FILE = "dataset/test/test_source3.tsv"
MODEL_FILE = os.environ.get("ER_MODEL_FILE", "dataset/train/model_hard.pkl")
OUTPUT_FILE = os.environ.get("ER_MATCHING_OUTPUT", "output/matching_results.tsv")
INFERENCE_DB = os.environ.get("ER_INFERENCE_DB", "inference_temp.duckdb")

def safe_ratio_vec(a, b):
    scores = process.cpdist(a, b, scorer=ratio, workers=4, dtype=np.float32)
    return np.where(np.logical_and(a, b), scores / 100.0, 0.0)

def safe_token_sort_vec(a, b):
    scores = process.cpdist(a, b, scorer=token_sort_ratio, workers=4, dtype=np.float32)
    return np.where(np.logical_and(a, b), scores / 100.0, 0.0)

def safe_token_set_vec(a, b):
    scores = process.cpdist(a, b, scorer=token_set_ratio, workers=4, dtype=np.float32)
    return np.where(np.logical_and(a, b), scores / 100.0, 0.0)

def main():
    print("Loading model...")
    model_data = joblib.load(MODEL_FILE)
    model = model_data["model"]
    threshold = model_data.get("threshold", 0.70)
    feature_cols = model_data["feature_columns"]
    print(f"Model loaded. Threshold: {threshold}")

    print("Initializing DuckDB...")
    # Using an on-disk database just in case memory pressure is high
    con = duckdb.connect(database=INFERENCE_DB)
    
    print("Loading test data into DuckDB...")
    # all_varchar=True prevents type inference errors on dirty strings
    con.execute(f"CREATE OR REPLACE TABLE s1 AS SELECT * FROM read_csv('{S1_TEST_FILE}', header=True, sep='\\t', all_varchar=True)")
    con.execute(f"CREATE OR REPLACE TABLE s2 AS SELECT * FROM read_csv('{S2_TEST_FILE}', header=True, sep='\\t', all_varchar=True)")
    con.execute(f"CREATE OR REPLACE TABLE s3 AS SELECT * FROM read_csv('{S3_TEST_FILE}', header=True, sep='\\t', all_varchar=True)")
    
    print("Creating unified candidates table...")
    con.execute("""
        CREATE OR REPLACE TABLE candidates AS 
        SELECT * FROM s2 
        UNION ALL 
        SELECT * FROM s3
    """)
    
    print("Loading and unnesting candidate_pairs.tsv... (This might take a minute)")
    con.execute(f"""
        CREATE OR REPLACE TABLE unnested_pairs AS
        SELECT 
            source1_entity_id, 
            unnest(string_split(candidate_entity_ids, ',')) AS candidate_entity_id
        FROM read_csv('{CANDIDATE_FILE}', header=True, sep='\\t', all_varchar=True)
        WHERE candidate_entity_ids IS NOT NULL AND candidate_entity_ids != ''
    """)

    # Join query to fetch raw attributes all at once
    query = """
        SELECT 
            p.source1_entity_id,
            p.candidate_entity_id,
            s1.business_name AS name1,
            s1.business_address AS addr1,
            s1.country AS country1,
            c.business_name AS name2,
            c.business_address AS addr2,
            c.country AS country2
        FROM unnested_pairs p
        JOIN s1 ON p.source1_entity_id = s1.entity_id
        JOIN candidates c ON p.candidate_entity_id = c.entity_id
    """
    
    print("Executing join query and streaming chunks to Python...")
    res = con.execute(query)
    
    start_time = time.time()
    valid_matches = defaultdict(list)
    vectors_per_chunk = 25
    processed_count = 0
    
    while True:
        chunk = res.fetch_df_chunk(vectors_per_chunk)
        if chunk.empty:
            break
            
        chunk.fillna("", inplace=True)
        
        # Text normalization using the same logic as training
        n1 = chunk['name1'].apply(normalize_text).tolist()
        a1 = chunk['addr1'].apply(normalize_text).tolist()
        c1 = chunk['country1'].apply(normalize_country).tolist()
        
        n2 = chunk['name2'].apply(normalize_text).tolist()
        a2 = chunk['addr2'].apply(normalize_text).tolist()
        c2 = chunk['country2'].apply(normalize_country).tolist()
        
        # Feature computation
        features = pd.DataFrame()
        features['name_ratio'] = safe_ratio_vec(n1, n2)
        features['name_token_sort'] = safe_token_sort_vec(n1, n2)
        features['name_token_set'] = safe_token_set_vec(n1, n2)
        features['address_ratio'] = safe_ratio_vec(a1, a2)
        features['address_token_sort'] = safe_token_sort_vec(a1, a2)
        features['address_token_set'] = safe_token_set_vec(a1, a2)
        features['country_match'] = [1 if x == y else 0 for x, y in zip(c1, c2)]
        features['name_length_diff'] = [abs(len(x) - len(y)) for x, y in zip(n1, n2)]
        features['address_length_diff'] = [abs(len(x) - len(y)) for x, y in zip(a1, a2)]
        
        # Enforce exact column order as expected by Random Forest
        features = features[feature_cols]
        
        # Predict
        probs = model.predict_proba(features)[:, 1]
        chunk['prob'] = probs
        
        # Keep predictions passing the threshold
        matches = chunk[chunk['prob'] >= threshold]
        for s1_id, candidate_id in zip(
            matches['source1_entity_id'], matches['candidate_entity_id']
        ):
            valid_matches[s1_id].append(candidate_id)
            
        processed_count += len(chunk)
        elapsed = time.time() - start_time
        print(f"Processed {processed_count:,} pairs... Elapsed: {elapsed:.2f}s ({processed_count/elapsed:,.0f} pairs/sec)")

    print("Writing final formatted matches (including singletons)...")
    s1_ids = con.execute("SELECT entity_id FROM s1").fetchdf()['entity_id'].tolist()
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_ids:
            matches = valid_matches.get(s1_id, [])
            f.write(f"{s1_id}\t{','.join(matches)}\n")
            
    print(f"Done! Results successfully saved to {OUTPUT_FILE}")
    con.close()

if __name__ == "__main__":
    main()
