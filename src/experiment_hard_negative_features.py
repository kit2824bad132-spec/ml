import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio, token_sort_ratio

from preprocessing import normalize_country, normalize_text


PAIR_FILE = "dataset/train/hard_negative_pairs_experiment.tsv"
S1_FILE = "dataset/train/train_source1_sample_50k.tsv"
S2_FILE = "dataset/train/train_source2_matches_50k.tsv"
S3_FILE = "dataset/train/train_source3_matches_50k.tsv"
OUTPUT_FILE = "dataset/train/hard_negative_features_experiment.tsv"


def similarity(scorer, value1, value2):
    if not value1 or not value2:
        return 0.0
    return scorer(value1, value2) / 100.0


def main():
    pairs = pd.read_csv(PAIR_FILE, sep="\t", dtype=str).fillna("")
    s1 = pd.read_csv(S1_FILE, sep="\t", dtype=str).fillna("")
    candidates = pd.concat(
        [
            pd.read_csv(S2_FILE, sep="\t", dtype=str).fillna(""),
            pd.read_csv(S3_FILE, sep="\t", dtype=str).fillna(""),
        ],
        ignore_index=True,
    ).drop_duplicates("entity_id")
    source_lookup = s1.set_index("entity_id").to_dict("index")
    candidate_lookup = candidates.set_index("entity_id").to_dict("index")
    rows = []

    print(f"Creating features for {len(pairs):,} experiment pairs...")
    for index, pair in enumerate(pairs.itertuples(index=False), start=1):
        source = source_lookup.get(pair.source1_entity_id)
        candidate = candidate_lookup.get(pair.candidate_entity_id)
        if source is None or candidate is None:
            continue

        name1 = normalize_text(source.get("business_name", ""))
        name2 = normalize_text(candidate.get("business_name", ""))
        address1 = normalize_text(source.get("business_address", ""))
        address2 = normalize_text(candidate.get("business_address", ""))
        country1 = normalize_country(source.get("country", ""))
        country2 = normalize_country(candidate.get("country", ""))
        rows.append(
            {
                "source1_entity_id": pair.source1_entity_id,
                "candidate_entity_id": pair.candidate_entity_id,
                "name_ratio": similarity(ratio, name1, name2),
                "name_token_sort": similarity(token_sort_ratio, name1, name2),
                "name_token_set": similarity(token_set_ratio, name1, name2),
                "address_ratio": similarity(ratio, address1, address2),
                "address_token_sort": similarity(token_sort_ratio, address1, address2),
                "address_token_set": similarity(token_set_ratio, address1, address2),
                "country_match": int(bool(country1) and country1 == country2),
                "name_length_diff": abs(len(name1) - len(name2)),
                "address_length_diff": abs(len(address1) - len(address2)),
                "selection_reason": pair.selection_reason,
                "country_conflict": pair.country_conflict == "True",
                "label": 0,
            }
        )
        if index % 10000 == 0:
            print(f"Processed {index:,} pairs")

    features = pd.DataFrame(rows)
    features.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print("Feature rows:", len(features))
    print("Feature similarity range:", float(features.name_ratio.min()), "to", float(features.name_ratio.max()))
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()