from collections import defaultdict
import math

import pandas as pd
from rapidfuzz.fuzz import ratio

from preprocessing import normalize_country, normalize_text


S1_FILE = "dataset/train/train_source1_sample_50k.tsv"
S2_FILE = "dataset/train/train_source2_matches_50k.tsv"
S3_FILE = "dataset/train/train_source3_matches_50k.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"
OLD_PAIRS_FILE = "dataset/train/hard_negative_pairs_50k.tsv"
OUTPUT_FILE = "dataset/train/hard_negative_pairs_experiment.tsv"
SHORTLIST_SIZE = 300
MAX_TOKEN_FREQUENCY = 5000


def tokens(value):
    return {token for token in value.split() if len(token) >= 2}


def load_truth(path, source_ids):
    ground_truth = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    truth = {}
    for row in ground_truth.itertuples(index=False):
        source_id = row.source1_entity_id
        if source_id in source_ids:
            truth[source_id] = {
                value.strip()
                for value in row.matched_entity_ids.split(",")
                if value.strip()
            }
    return truth


def build_index(records, column):
    index = defaultdict(list)
    frequencies = defaultdict(int)
    record_tokens = {}
    for record in records:
        values = tokens(record[column])
        record_tokens[record["entity_id"]] = values
        for token in values:
            index[token].append(record["entity_id"])
            frequencies[token] += 1
    return index, frequencies, record_tokens


def shortlist(source_tokens, index, frequencies):
    overlap_scores = defaultdict(float)
    for token in source_tokens:
        frequency = frequencies.get(token, 0)
        if frequency == 0 or frequency > MAX_TOKEN_FREQUENCY:
            continue
        weight = 1.0 / math.log2(frequency + 2)
        for candidate_id in index[token]:
            overlap_scores[candidate_id] += weight
    ranked = sorted(overlap_scores, key=lambda key: (-overlap_scores[key], key))
    return ranked[:SHORTLIST_SIZE]


def select_candidates(source, candidates, truth, name_shortlist, address_shortlist):
    source_id = source["entity_id"]
    excluded = truth.get(source_id, set())
    candidate_ids = set(name_shortlist) | set(address_shortlist)
    scored = []

    for candidate_id in candidate_ids:
        if candidate_id in excluded:
            continue
        candidate = candidates[candidate_id]
        name_score = ratio(source["name"], candidate["name"]) / 100 if source["name"] and candidate["name"] else 0.0
        address_score = ratio(source["address"], candidate["address"]) / 100 if source["address"] and candidate["address"] else 0.0
        country_conflict = bool(
            source["country"]
            and candidate["country"]
            and source["country"] != candidate["country"]
        )

        if name_score >= 0.60 and address_score <= 0.70:
            scored.append(("name_confusion", name_score, address_score, candidate_id))
        if address_score >= 0.60 and name_score <= 0.75:
            scored.append(("address_confusion", address_score, name_score, candidate_id))
        if country_conflict and name_score >= 0.60:
            scored.append(("country_confusion", name_score, address_score, candidate_id))

    chosen = {}
    priorities = ("name_confusion", "address_confusion", "country_confusion")
    for category in priorities:
        category_rows = [row for row in scored if row[0] == category]
        category_rows.sort(key=lambda row: (-row[1], row[2], row[3]))
        for _, primary_score, secondary_score, candidate_id in category_rows:
            if candidate_id not in chosen:
                chosen[candidate_id] = (category, primary_score, secondary_score)
                break

    return [
        {
            "source1_entity_id": source_id,
            "candidate_entity_id": candidate_id,
            "name_similarity": round(
                ratio(source["name"], candidates[candidate_id]["name"]) / 100
                if source["name"] and candidates[candidate_id]["name"]
                else 0.0,
                6,
            ),
            "address_similarity": round(
                ratio(source["address"], candidates[candidate_id]["address"]) / 100
                if source["address"] and candidates[candidate_id]["address"]
                else 0.0,
                6,
            ),
            "selection_reason": reason,
            "country_conflict": bool(
                source["country"]
                and candidates[candidate_id]["country"]
                and source["country"] != candidates[candidate_id]["country"]
            ),
            "label": 0,
        }
        for candidate_id, (reason, _, _) in chosen.items()
    ]


def main():
    print("Loading source records and ground truth...")
    s1 = pd.read_csv(S1_FILE, sep="\t", dtype=str).fillna("")
    candidate_frame = pd.concat(
        [
            pd.read_csv(S2_FILE, sep="\t", dtype=str).fillna(""),
            pd.read_csv(S3_FILE, sep="\t", dtype=str).fillna(""),
        ],
        ignore_index=True,
    ).drop_duplicates("entity_id")
    source_ids = set(s1.entity_id)
    truth = load_truth(GT_FILE, source_ids)

    sources = {
        row.entity_id: {
            "entity_id": row.entity_id,
            "name": normalize_text(row.business_name),
            "address": normalize_text(row.business_address),
            "country": normalize_country(row.country),
        }
        for row in s1.itertuples(index=False)
    }
    candidates = {
        row.entity_id: {
            "entity_id": row.entity_id,
            "name": normalize_text(row.business_name),
            "address": normalize_text(row.business_address),
            "country": normalize_country(row.country),
        }
        for row in candidate_frame.itertuples(index=False)
    }

    name_index, name_frequency, _ = build_index(list(candidates.values()), "name")
    address_index, address_frequency, _ = build_index(list(candidates.values()), "address")
    generated = []

    print(f"Indexed {len(candidates):,} candidate records; selecting negatives...")
    for index, source in enumerate(sources.values(), start=1):
        name_shortlist = shortlist(
            tokens(source["name"]), name_index, name_frequency
        )
        address_shortlist = shortlist(
            tokens(source["address"]), address_index, address_frequency
        )
        generated.extend(
            select_candidates(
                source, candidates, truth, name_shortlist, address_shortlist
            )
        )
        if index % 5000 == 0:
            print(f"Processed {index:,} source-1 records")

    result = pd.DataFrame(generated)
    if not result.empty:
        result = result.drop_duplicates(
            ["source1_entity_id", "candidate_entity_id"]
        )
    result.to_csv(OUTPUT_FILE, sep="\t", index=False)

    print("\nNew hard negatives:", len(result))
    print("Selection reasons:")
    print(result.selection_reason.value_counts().to_string() if len(result) else "none")
    print("Unique source-1 records:", result.source1_entity_id.nunique() if len(result) else 0)
    if len(result):
        for column in ("name_similarity", "address_similarity"):
            print(f"{column} quantiles:")
            print(result[column].quantile([0, .25, .5, .75, .9, .95, 1]).round(3).to_string())
        print("Nonempty country-conflict rate:", round(float(result.country_conflict.mean()), 4))

    old = pd.read_csv(OLD_PAIRS_FILE, sep="\t")
    print("\nLegacy hard negatives:", len(old))
    print("Legacy name_similarity quantiles (0-1):")
    print((old.name_similarity / 100).quantile([0, .25, .5, .75, .9, .95, 1]).round(3).to_string())
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()