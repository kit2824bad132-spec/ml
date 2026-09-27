import pandas as pd
import numpy as np
import joblib
import time
import os
import sys

# Append src to path so we can import preprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import normalize_text, normalize_country
from rapidfuzz.fuzz import ratio, token_sort_ratio, token_set_ratio

CANDIDATE_FILE = "output/candidate_pairs.tsv"
S1_TEST_FILE = "dataset/test/test_source1.tsv"
S2_TEST_FILE = "dataset/test/test_source2.tsv"
S3_TEST_FILE = "dataset/test/test_source3.tsv"
MODEL_FILE = "dataset/train/model_hard.pkl"
OUTPUT_FILE = "output/matching_results.tsv"

def safe_ratio_vec(a, b):
    return [ratio(x, y)/100.0 if x and y else 0.0 for x, y in zip(a, b)]

def safe_token_sort_vec(a, b):
    return [token_sort_ratio(x, y)/100.0 if x and y else 0.0 for x, y in zip(a, b)]

def safe_token_set_vec(a, b):
    return [token_set_ratio(x, y)/100.0 if x and y else 0.0 for x, y in zip(a, b)]

def main():
    print("Loading model...")
    model_data = joblib.load(MODEL_FILE)
    model = model_data["model"]
    threshold = model_data.get("threshold", 0.70)
    feature_cols = model_data["feature_columns"]
    print(f"Model loaded. Threshold: {threshold}")

    print("Loading test data...")
    s1 = pd.read_csv(S1_TEST_FILE, sep="\t", dtype=str).fillna("")
    s_cands = pd.concat([
        pd.read_csv(S2_TEST_FILE, sep="\t", dtype=str).fillna(""),
        pd.read_csv(S3_TEST_FILE, sep="\t", dtype=str).fillna("")
    ], ignore_index=True)

    print("Pre-normalizing test strings to save time...")
    s1['name_norm'] = s1['business_name'].apply(normalize_text)
    s1['addr_norm'] = s1['business_address'].apply(normalize_text)
    s1['ctry_norm'] = s1['country'].apply(normalize_country)
    
    s_cands['name_norm'] = s_cands['business_name'].apply(normalize_text)
    s_cands['addr_norm'] = s_cands['business_address'].apply(normalize_text)
    s_cands['ctry_norm'] = s_cands['country'].apply(normalize_country)

    # Convert to dicts for lightning-fast lookup
    # dict format: entity_id -> (name_norm, addr_norm, ctry_norm)
    print("Building lookup dictionaries...")
    s1_dict = {row['entity_id']: (row['name_norm'], row['addr_norm'], row['ctry_norm']) for _, row in s1.iterrows()}
    cand_dict = {row['entity_id']: (row['name_norm'], row['addr_norm'], row['ctry_norm']) for _, row in s_cands.iterrows()}
    
    print("Starting chunked inference...")
    start_time = time.time()
    
    chunk_size = 50000 # number of S1 rows per chunk
    current_chunk = []
    processed_s1_count = 0
    
    with open(OUTPUT_FILE, "w") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        
        with open(CANDIDATE_FILE, "r", encoding="utf-8") as f_in:
            for line in f_in:
                if line.startswith("source1_entity_id"): 
                    continue
                line = line.strip()
                if not line:
                    continue
                current_chunk.append(line)
                
                if len(current_chunk) >= chunk_size:
                    process_chunk_and_write(current_chunk, s1_dict, cand_dict, model, threshold, feature_cols, f_out)
                    processed_s1_count += len(current_chunk)
                    print(f"Processed {processed_s1_count} S1 entities... Elapsed: {time.time()-start_time:.2f}s")
                    current_chunk = []
            
            # Process remaining
            if current_chunk:
                process_chunk_and_write(current_chunk, s1_dict, cand_dict, model, threshold, feature_cols, f_out)
                processed_s1_count += len(current_chunk)
                print(f"Processed {processed_s1_count} S1 entities... Elapsed: {time.time()-start_time:.2f}s")
                
        # Fill in missing singletons (S1 IDs that were completely absent from candidate_pairs.tsv)
        print("Checking for missing S1 entities...")
        for s1_id in s1_dict:
            f_out.write(f"{s1_id}\t\n")
            
    print(f"Done! Total time: {time.time()-start_time:.2f}s")
    print(f"Results saved to {OUTPUT_FILE}")

def process_chunk_and_write(chunk_lines, s1_dict, cand_dict, model, threshold, feature_cols, f_out):
    batch_features = []
    batch_meta = []
    
    # We will track which S1s we've seen so we can remove them from s1_dict
    seen_s1s = []
    
    for line in chunk_lines:
        parts = line.split("\t")
        if len(parts) == 1:
            # No candidates
            seen_s1s.append((parts[0], []))
            continue
        
        s1_id = parts[0]
        cands = parts[1].split(",")
        seen_s1s.append((s1_id, cands))
        
        s1_rec = s1_dict.get(s1_id)
        if not s1_rec: continue
        
        n1, a1, c1 = s1_rec
        
        for cand_id in cands:
            if not cand_id: continue
            c_rec = cand_dict.get(cand_id)
            if not c_rec: continue
            
            n2, a2, c2 = c_rec
            
            features = {
                "name_ratio": (ratio(n1, n2) / 100.0) if n1 and n2 else 0.0,
                "name_token_sort": (token_sort_ratio(n1, n2) / 100.0) if n1 and n2 else 0.0,
                "name_token_set": (token_set_ratio(n1, n2) / 100.0) if n1 and n2 else 0.0,
                "address_ratio": (ratio(a1, a2) / 100.0) if a1 and a2 else 0.0,
                "address_token_sort": (token_sort_ratio(a1, a2) / 100.0) if a1 and a2 else 0.0,
                "address_token_set": (token_set_ratio(a1, a2) / 100.0) if a1 and a2 else 0.0,
                "country_match": 1 if c1 == c2 else 0,
                "name_length_diff": abs(len(n1) - len(n2)),
                "address_length_diff": abs(len(a1) - len(a2))
            }
            batch_features.append([features[c] for c in feature_cols])
            batch_meta.append((s1_id, cand_id))
            
    if batch_features:
        probs = model.predict_proba(batch_features)[:, 1]
        valid_pairs = {}
        for (s1_id, cand_id), p in zip(batch_meta, probs):
            if p >= threshold:
                if s1_id not in valid_pairs:
                    valid_pairs[s1_id] = []
                valid_pairs[s1_id].append(cand_id)
                
        for s1_id, cands in seen_s1s:
            if s1_id in valid_pairs:
                f_out.write(f"{s1_id}\t{','.join(valid_pairs[s1_id])}\n")
            else:
                f_out.write(f"{s1_id}\t\n")
            s1_dict.pop(s1_id, None)
    else:
        for s1_id, cands in seen_s1s:
            f_out.write(f"{s1_id}\t\n")
            s1_dict.pop(s1_id, None)

if __name__ == "__main__":
    main()
