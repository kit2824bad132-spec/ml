from collections import defaultdict
import math
import random
import re

import pandas as pd
from rapidfuzz.fuzz import ratio
from sklearn.model_selection import GroupShuffleSplit

from preprocessing import normalize_text


S1_FILE = "dataset/train/train_source1_sample_50k.tsv"
S2_FILE = "dataset/train/train_source2_matches_50k.tsv"
S3_FILE = "dataset/train/train_source3_matches_50k.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"
SPLIT_FILE = "dataset/train/clean_validation_split.tsv"
VALIDATION_CANDIDATES_FILE = "dataset/train/clean_validation_candidates.tsv"
BASELINE_PAIRS_FILE = "dataset/train/clean_validation_training_pairs_baseline.tsv"
IMPROVED_PAIRS_FILE = "dataset/train/clean_validation_training_pairs_improved.tsv"

RANDOM_SEED = 42
MAX_BLOCK_CANDIDATES = 250
NEGATIVES_PER_SOURCE = 2
LEGACY_RANDOM_SAMPLE = 100
HARD_SHORTLIST_SIZE = 300
MAX_TOKEN_FREQUENCY = 5000
PAIR_COLUMNS = ["source1_entity_id", "candidate_entity_id", "label", "pair_type"]


def get_block_keys(record):
    country = record["country"].strip().lower()
    name = record["name"]
    address = record["address"]
    compact_name = name.replace(" ", "")
    first_token = name.split(" ", 1)[0] if name else ""
    keys = []
    if name:
        keys.append(("exact", country, name))
    if len(compact_name) >= 6:
        keys.append(("prefix6", country, compact_name[:6]))
    if len(first_token) >= 5:
        keys.append(("first_token", country, first_token))
    for code in set(re.findall(r"(?<!\d)\d{5,6}(?!\d)", address)):
        keys.append(("postal", country, code))
    return keys


def build_block_index(candidates):
    buckets = defaultdict(list)
    for candidate_id, record in candidates.items():
        for key in get_block_keys(record):
            buckets[key].append(candidate_id)
    return {
        key: values
        for key, values in buckets.items()
        if len(values) <= MAX_BLOCK_CANDIDATES
    }


def generate_candidates(source_rows, candidate_records, block_index, output_file):
    output_rows = []
    total = len(source_rows)
    for position, source in enumerate(source_rows, start=1):
        candidate_ids = set()
        for key in get_block_keys(source):
            candidate_ids.update(block_index.get(key, ()))
        output_rows.extend(
            {
                "source1_entity_id": source["entity_id"],
                "candidate_entity_id": candidate_id,
            }
            for candidate_id in sorted(candidate_ids)
        )
        if position % 1000 == 0:
            print(f"Candidate generation: {position:,}/{total:,} source-1 rows")
    frame = pd.DataFrame(
        output_rows,
        columns=["source1_entity_id", "candidate_entity_id"],
    )
    frame.to_csv(output_file, sep="\t", index=False)
    print(f"Frozen candidates saved: {output_file} ({len(frame):,} pairs)")
    return frame


def read_training_truth(path, training_ids):
    truth = {}
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=200000):
        selected = chunk[chunk.source1_entity_id.isin(training_ids)]
        for row in selected.itertuples(index=False):
            truth[row.source1_entity_id] = {
                value.strip()
                for value in row.matched_entity_ids.split(",")
                if value.strip()
            }
    return truth


def token_set(value):
    return {token for token in value.split() if len(token) >= 2}


def build_token_index(candidate_records, field):
    index = defaultdict(list)
    frequencies = defaultdict(int)
    for candidate_id, record in candidate_records.items():
        for token in token_set(record[field]):
            index[token].append(candidate_id)
            frequencies[token] += 1
    return index, frequencies


def token_shortlist(value, index, frequencies):
    scores = defaultdict(float)
    for token in token_set(value):
        frequency = frequencies.get(token, 0)
        if not frequency or frequency > MAX_TOKEN_FREQUENCY:
            continue
        weight = 1 / math.log2(frequency + 2)
        for candidate_id in index[token]:
            scores[candidate_id] += weight
    return sorted(scores, key=lambda cid: (-scores[cid], cid))[:HARD_SHORTLIST_SIZE]


def pair_row(source_id, candidate_id, label, pair_type):
    return {
        "source1_entity_id": source_id,
        "candidate_entity_id": candidate_id,
        "label": int(label),
        "pair_type": pair_type,
    }


def make_training_pairs(training_rows, candidate_records, truth):
    candidate_ids = list(candidate_records)
    candidate_id_set = set(candidate_ids)
    rows_common = []
    old_hard_rows = []
    new_hard_rows = []
    rng = random.Random(RANDOM_SEED)

    name_index, name_frequency = build_token_index(candidate_records, "name")
    address_index, address_frequency = build_token_index(candidate_records, "address")

    for position, source in enumerate(training_rows, start=1):
        source_id = source["entity_id"]
        true_targets = truth.get(source_id, set())
        available_true = sorted(true_targets & candidate_id_set)
        rows_common.extend(
            pair_row(source_id, candidate_id, 1, "positive")
            for candidate_id in available_true
        )

        selected_random = set()
        attempts = 0
        while len(selected_random) < NEGATIVES_PER_SOURCE and attempts < 1000:
            attempts += 1
            candidate_id = rng.choice(candidate_ids)
            if candidate_id not in true_targets:
                selected_random.add(candidate_id)
        rows_common.extend(
            pair_row(source_id, candidate_id, 0, "random_negative")
            for candidate_id in sorted(selected_random)
        )

        # Reproduce the legacy random-100 then top-name-similarity strategy,
        # but only for training source-1 IDs.
        legacy_scored = []
        source_name = source["name"].lower().strip()
        if source_name:
            for candidate_id in rng.sample(candidate_ids, min(LEGACY_RANDOM_SAMPLE, len(candidate_ids))):
                if candidate_id in true_targets:
                    continue
                candidate_name = candidate_records[candidate_id]["name"].lower().strip()
                if candidate_name:
                    legacy_scored.append((ratio(source_name, candidate_name), candidate_id))
        legacy_scored.sort(reverse=True)
        old_hard_rows.extend(
            pair_row(source_id, candidate_id, 0, "legacy_hard_negative")
            for _, candidate_id in legacy_scored[:2]
        )

        name_candidates = token_shortlist(source["name"], name_index, name_frequency)
        address_candidates = token_shortlist(source["address"], address_index, address_frequency)
        category_rows = {"name_confusion": [], "address_confusion": [], "country_confusion": []}
        for candidate_id in set(name_candidates) | set(address_candidates):
            if candidate_id in true_targets:
                continue
            candidate = candidate_records[candidate_id]
            name_score = ratio(source["name"], candidate["name"]) / 100 if source["name"] and candidate["name"] else 0
            address_score = ratio(source["address"], candidate["address"]) / 100 if source["address"] and candidate["address"] else 0
            country_conflict = bool(
                source["country"].strip()
                and candidate["country"].strip()
                and source["country"].strip().lower() != candidate["country"].strip().lower()
            )
            if name_score >= 0.60 and address_score <= 0.70:
                category_rows["name_confusion"].append((name_score, address_score, candidate_id))
            if address_score >= 0.60 and name_score <= 0.75:
                category_rows["address_confusion"].append((address_score, name_score, candidate_id))
            if country_conflict and name_score >= 0.60:
                category_rows["country_confusion"].append((name_score, address_score, candidate_id))

        selected = set()
        for category in ("name_confusion", "address_confusion", "country_confusion"):
            category_rows[category].sort(key=lambda item: (-item[0], item[1], item[2]))
            for _, _, candidate_id in category_rows[category]:
                if candidate_id not in selected:
                    selected.add(candidate_id)
                    new_hard_rows.append(pair_row(source_id, candidate_id, 0, category))
                    break

        if position % 5000 == 0:
            print(f"Training pair construction: {position:,}/{len(training_rows):,}")

    def finalize(extra_rows):
        frame = pd.DataFrame(rows_common + extra_rows, columns=PAIR_COLUMNS)
        frame["priority"] = frame.label
        frame = frame.sort_values("priority", ascending=False).drop_duplicates(
            ["source1_entity_id", "candidate_entity_id"], keep="first"
        )
        return frame[PAIR_COLUMNS].reset_index(drop=True)

    return finalize(old_hard_rows), finalize(new_hard_rows), rows_common, old_hard_rows, new_hard_rows


def main():
    print("Loading source records; no ground truth is loaded in this phase...")
    s1_frame = pd.read_csv(S1_FILE, sep="\t", dtype=str, keep_default_na=False)
    s1_frame = s1_frame.sort_values("entity_id").reset_index(drop=True)
    candidates_frame = pd.concat(
        [
            pd.read_csv(S2_FILE, sep="\t", dtype=str, keep_default_na=False),
            pd.read_csv(S3_FILE, sep="\t", dtype=str, keep_default_na=False),
        ],
        ignore_index=True,
    ).drop_duplicates("entity_id")

    source_rows = [
        {
            "entity_id": row.entity_id,
            "name": normalize_text(row.business_name),
            "address": normalize_text(row.business_address),
            "country": row.country,
        }
        for row in s1_frame.itertuples(index=False)
    ]
    candidate_records = {
        row.entity_id: {
            "name": normalize_text(row.business_name),
            "address": normalize_text(row.business_address),
            "country": row.country,
        }
        for row in candidates_frame.itertuples(index=False)
    }

    train_indices, validation_indices = next(
        GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED).split(
            s1_frame[["entity_id"]], groups=s1_frame.entity_id
        )
    )
    training_ids = set(s1_frame.iloc[train_indices].entity_id)
    validation_ids = set(s1_frame.iloc[validation_indices].entity_id)
    split = s1_frame[["entity_id"]].copy()
    split["split"] = split.entity_id.map(
        lambda entity_id: "train" if entity_id in training_ids else "validation"
    )
    split.to_csv(SPLIT_FILE, sep="\t", index=False)
    print(f"Fixed split saved: {len(training_ids):,} train, {len(validation_ids):,} validation")
    print("Source-1 ID overlap:", len(training_ids & validation_ids))

    block_index = build_block_index(candidate_records)
    print(f"Built label-independent blocker with {len(block_index):,} allowed blocks")
    validation_rows = [row for row in source_rows if row["entity_id"] in validation_ids]
    validation_candidates = generate_candidates(
        validation_rows,
        candidate_records,
        block_index,
        VALIDATION_CANDIDATES_FILE,
    )

    # The frozen validation candidate file is written above before this first GT read.
    print("Loading ground truth only now, after validation candidates are frozen...")
    truth = read_training_truth(GT_FILE, training_ids)
    training_rows = [row for row in source_rows if row["entity_id"] in training_ids]
    baseline_pairs, improved_pairs, common_rows, old_hard_rows, new_hard_rows = make_training_pairs(
        training_rows, candidate_records, truth
    )
    baseline_pairs.to_csv(BASELINE_PAIRS_FILE, sep="\t", index=False)
    improved_pairs.to_csv(IMPROVED_PAIRS_FILE, sep="\t", index=False)

    print("\nCLEAN PREPARATION COMPLETE")
    print("Frozen validation candidates:", len(validation_candidates))
    print("Training positives:", sum(row["label"] == 1 for row in common_rows))
    print("Training normal negatives:", sum(row["pair_type"] == "random_negative" for row in common_rows))
    print("Legacy hard negatives generated on train IDs:", len(old_hard_rows))
    print("Improved hard negatives generated on train IDs:", len(new_hard_rows))
    print("Baseline training pairs:", len(baseline_pairs), baseline_pairs.label.value_counts().to_dict())
    print("Improved training pairs:", len(improved_pairs), improved_pairs.label.value_counts().to_dict())
    print("Validation candidate IDs in validation split:", validation_candidates.source1_entity_id.nunique())
    print("Saved split:", SPLIT_FILE)
    print("Saved frozen candidates:", VALIDATION_CANDIDATES_FILE)
    print("Saved baseline pairs:", BASELINE_PAIRS_FILE)
    print("Saved improved pairs:", IMPROVED_PAIRS_FILE)


if __name__ == "__main__":
    main()