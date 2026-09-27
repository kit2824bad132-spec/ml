from collections import defaultdict
import gzip
import json
import math
import re
import csv

import pandas as pd

from preprocessing import normalize_text


S1_FILE = "dataset/train/train_source1_sample_50k.tsv"
S2_FILE = "dataset/train/train_source2_matches_50k.tsv"
S3_FILE = "dataset/train/train_source3_matches_50k.tsv"
SPLIT_FILE = "dataset/train/clean_validation_split.tsv"
CURRENT_FILE = "dataset/train/clean_validation_candidates.tsv"
PASS_FILE = "dataset/train/clean_blocking_pass_candidates.tsv.gz"
UNION_FILE = "dataset/train/clean_blocking_candidates_mult_pass.tsv"
GENERATION_REPORT = "output/clean_blocking_generation_report.json"

MAX_BLOCK_SIZE = 250
MAX_TOKEN_PASS_CANDIDATES_PER_SOURCE = 200
RARE_TOKEN_FREQUENCY = 250

NAME_STOPWORDS = {
    "and", "the", "for", "with", "from", "inc", "llc", "ltd", "limited",
    "company", "co", "corp", "corporation", "group", "services", "service",
    "healthcare", "health", "medical", "clinic", "center", "centre", "hospital",
}
ADDRESS_STOPWORDS = {
    "and", "the", "road", "rd", "street", "st", "avenue", "ave", "drive",
    "dr", "boulevard", "blvd", "lane", "ln", "suite", "ste", "unit", "floor",
    "fl", "building", "bldg", "highway", "hwy", "north", "south", "east", "west",
}


def read_records():
    s1 = pd.read_csv(S1_FILE, sep="\t", dtype=str, keep_default_na=False)
    split = pd.read_csv(SPLIT_FILE, sep="\t", dtype=str)
    validation_ids = set(split.loc[split.split == "validation", "entity_id"])
    source_rows = []
    for row in s1[s1.entity_id.isin(validation_ids)].itertuples(index=False):
        source_rows.append(
            {
                "entity_id": row.entity_id,
                "name": normalize_text(row.business_name),
                "address": normalize_text(row.business_address),
                "country": row.country.strip().lower(),
            }
        )

    s2 = pd.read_csv(S2_FILE, sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(S3_FILE, sep="\t", dtype=str, keep_default_na=False)
    target_rows = []
    target_by_source = {"source2": {}, "source3": {}}
    for source_name, frame in (("source2", s2), ("source3", s3)):
        for row in frame.itertuples(index=False):
            record = {
                "entity_id": row.entity_id,
                "name": normalize_text(row.business_name),
                "address": normalize_text(row.business_address),
                "country": row.country.strip().lower(),
                "source": source_name,
            }
            target_by_source[source_name][record["entity_id"]] = record
            target_rows.append(record)
    targets = {record["entity_id"]: record for record in target_rows}
    return source_rows, targets, target_by_source, validation_ids


def words(value):
    return [word for word in value.split() if word]


def current_keys(record, pass_name):
    country = record["country"]
    name = record["name"]
    address = record["address"]
    compact = name.replace(" ", "")
    first = name.split(" ", 1)[0] if name else ""
    if pass_name == "current_exact_name_country":
        return [(country, name)] if name else []
    if pass_name == "current_prefix6_country":
        return [(country, compact[:6])] if len(compact) >= 6 else []
    if pass_name == "current_first_name_token_country":
        return [(country, first)] if len(first) >= 5 else []
    if pass_name == "current_postal_pin_country":
        return [(country, code) for code in set(re.findall(r"(?<!\d)\d{5,6}(?!\d)", address))]
    raise ValueError(pass_name)


def tokens_for(record, field, stopwords):
    return {
        token
        for token in words(record[field])
        if len(token) >= 4 and not token.isdigit() and token not in stopwords
    }


def token_document_frequency(records, field, stopwords):
    frequency = defaultdict(int)
    for record in records:
        for token in tokens_for(record, field, stopwords):
            frequency[token] += 1
    return frequency


def make_rule_keys(record, pass_name, name_frequency, address_frequency):
    country = record["country"]
    if pass_name == "exact_name_any_country":
        return [(record["name"],)] if record["name"] else []
    if pass_name == "rare_name_token_country":
        return [
            (country, token)
            for token in tokens_for(record, "name", NAME_STOPWORDS)
            if name_frequency.get(token, 0) <= RARE_TOKEN_FREQUENCY
        ]
    if pass_name == "rare_address_token_country":
        return [
            (country, token)
            for token in tokens_for(record, "address", ADDRESS_STOPWORDS)
            if address_frequency.get(token, 0) <= RARE_TOKEN_FREQUENCY
        ]
    if pass_name == "house_number_address_token_country":
        house_numbers = set(re.findall(r"(?<!\w)\d{1,6}(?!\w)", record["address"]))
        address_tokens = [
            token for token in tokens_for(record, "address", ADDRESS_STOPWORDS)
            if address_frequency.get(token, 0) <= RARE_TOKEN_FREQUENCY
        ]
        address_tokens.sort(key=lambda token: (address_frequency.get(token, 0), token))
        return [
            (country, number, token)
            for number in house_numbers
            for token in address_tokens[:3]
        ]
    if pass_name == "name_token_address_token_country":
        name_tokens = [
            token for token in tokens_for(record, "name", NAME_STOPWORDS)
            if name_frequency.get(token, 0) <= RARE_TOKEN_FREQUENCY
        ]
        address_tokens = [
            token for token in tokens_for(record, "address", ADDRESS_STOPWORDS)
            if address_frequency.get(token, 0) <= RARE_TOKEN_FREQUENCY
        ]
        name_tokens.sort(key=lambda token: (name_frequency.get(token, 0), token))
        address_tokens.sort(key=lambda token: (address_frequency.get(token, 0), token))
        return [
            (country, name_token, address_token)
            for name_token in name_tokens[:3]
            for address_token in address_tokens[:3]
        ]
    if pass_name == "address_tail_city_token_country":
        address_tokens = words(record["address"])
        tail = address_tokens[-3:]
        return [
            (country, token)
            for token in set(tail)
            if len(token) >= 4
            and not token.isdigit()
            and token not in ADDRESS_STOPWORDS
            and address_frequency.get(token, 0) <= RARE_TOKEN_FREQUENCY
        ]
    raise ValueError(pass_name)


def build_index(records, key_function):
    buckets = defaultdict(list)
    for candidate_id, record in records.items():
        for key in key_function(record):
            buckets[key].append(candidate_id)
    return {key: ids for key, ids in buckets.items() if len(ids) <= MAX_BLOCK_SIZE}


def execute_pass(
    name,
    source_rows,
    target_records,
    target_by_source,
    name_frequency,
    address_frequency,
    output_writer,
    cumulative_union,
):
    token_rule = name in {
        "rare_name_token_country",
        "rare_address_token_country",
        "house_number_address_token_country",
        "name_token_address_token_country",
        "address_tail_city_token_country",
    }
    if name.startswith("current_"):
        key_function = lambda record: current_keys(record, name)
        records = target_records
        source_records = source_rows
    elif name.startswith("separate_source2_") or name.startswith("separate_source3_"):
        source_name = "source2" if "source2" in name else "source3"
        underlying_rule = name.split("_", 2)[2]
        key_function = lambda record: current_keys(record, underlying_rule)
        records = target_by_source[source_name]
        source_records = source_rows
    else:
        key_function = lambda record: make_rule_keys(
            record, name, name_frequency, address_frequency
        )
        records = target_records
        source_records = source_rows

    index = build_index(records, key_function)
    pass_count = 0
    additional_pairs = 0
    max_per_source = 0
    for position, source in enumerate(source_records, start=1):
        scores = defaultdict(float)
        for key in key_function(source):
            candidate_ids = index.get(key, ())
            weight = 1.0 / math.log2(len(candidate_ids) + 2) if token_rule else 1.0
            for candidate_id in candidate_ids:
                scores[candidate_id] += weight
        if token_rule and len(scores) > MAX_TOKEN_PASS_CANDIDATES_PER_SOURCE:
            selected = sorted(scores, key=lambda cid: (-scores[cid], cid))[
                :MAX_TOKEN_PASS_CANDIDATES_PER_SOURCE
            ]
        else:
            selected = sorted(scores)
        max_per_source = max(max_per_source, len(selected))
        for candidate_id in selected:
            output_writer.writerow((name, source["entity_id"], candidate_id))
            pair = (source["entity_id"], candidate_id)
            pass_count += 1
            if pair not in cumulative_union:
                additional_pairs += 1
                cumulative_union.add(pair)
        if position % 1000 == 0:
            print(f"{name}: {position:,}/{len(source_records):,} source-1 rows")

    return {
        "strategy": name,
        "candidate_pairs": pass_count,
        "average_candidates_per_source1": pass_count / len(source_records) if source_records else 0,
        "maximum_candidates_per_source1": max_per_source,
        "additional_pairs_to_union": additional_pairs,
        "token_strategy_per_source_cap": MAX_TOKEN_PASS_CANDIDATES_PER_SOURCE if token_rule else None,
        "maximum_block_size": MAX_BLOCK_SIZE,
    }


def main():
    source_rows, targets, target_by_source, validation_ids = read_records()
    current_frame = pd.read_csv(
        CURRENT_FILE,
        sep="\t",
        dtype={"source1_entity_id": str, "candidate_entity_id": str},
    )
    current_pairs = set(zip(current_frame.source1_entity_id, current_frame.candidate_entity_id))
    if set(current_frame.source1_entity_id) - validation_ids:
        raise ValueError("Current validation candidates contain non-validation source IDs")
    cumulative_union = set(current_pairs)
    name_frequency = token_document_frequency(list(targets.values()), "name", NAME_STOPWORDS)
    address_frequency = token_document_frequency(list(targets.values()), "address", ADDRESS_STOPWORDS)

    passes = [
        "current_exact_name_country",
        "current_prefix6_country",
        "current_first_name_token_country",
        "current_postal_pin_country",
        "exact_name_any_country",
        "rare_name_token_country",
        "rare_address_token_country",
        "house_number_address_token_country",
        "name_token_address_token_country",
        "address_tail_city_token_country",
    ]
    for source_name in ("source2", "source3"):
        for rule_name in (
            "current_exact_name_country",
            "current_prefix6_country",
            "current_first_name_token_country",
            "current_postal_pin_country",
        ):
            passes.append(f"separate_{source_name}_{rule_name}")

    generation_stats = []
    with gzip.open(PASS_FILE, "wt", encoding="utf-8", newline="") as compressed:
        writer = csv.writer(compressed, delimiter="\t", lineterminator="\n")
        writer.writerow(("blocking_pass", "source1_entity_id", "candidate_entity_id"))
        for pass_name in passes:
            stats = execute_pass(
                pass_name,
                source_rows,
                targets,
                target_by_source,
                name_frequency,
                address_frequency,
                writer,
                cumulative_union,
            )
            generation_stats.append(stats)

    reconstructed_current = set()
    for stats in generation_stats[:4]:
        pass_name = stats["strategy"]
        key_function = lambda record, rule=pass_name: current_keys(record, rule)
        index = build_index(targets, key_function)
        for source in source_rows:
            for key in key_function(source):
                reconstructed_current.update(
                    (source["entity_id"], candidate_id)
                    for candidate_id in index.get(key, ())
                )
    if reconstructed_current != current_pairs:
        raise ValueError(
            "Reconstructed four-pass blocker differs from clean_validation_candidates.tsv: "
            f"missing={len(current_pairs - reconstructed_current)}, "
            f"extra={len(reconstructed_current - current_pairs)}"
        )

    union_frame = pd.DataFrame(
        sorted(cumulative_union),
        columns=["source1_entity_id", "candidate_entity_id"],
    )
    union_frame.to_csv(UNION_FILE, sep="\t", index=False)
    report = {
        "validation_source1_ids": len(validation_ids),
        "candidate_generation_used_ground_truth": False,
        "current_candidate_pairs": len(current_pairs),
        "current_candidate_average_per_source1": len(current_pairs) / len(validation_ids),
        "current_candidate_maximum_per_source1": int(current_frame.groupby("source1_entity_id").size().max()),
        "current_rules_reconstructed_exactly": True,
        "new_union_candidate_pairs": len(cumulative_union),
        "new_union_additional_pairs": len(cumulative_union - current_pairs),
        "new_union_growth_ratio": len(cumulative_union) / len(current_pairs) if current_pairs else None,
        "new_union_average_per_source1": len(cumulative_union) / len(validation_ids),
        "new_union_maximum_per_source1": int(union_frame.groupby("source1_entity_id").size().max()),
        "max_block_size": MAX_BLOCK_SIZE,
        "max_token_pass_candidates_per_source": MAX_TOKEN_PASS_CANDIDATES_PER_SOURCE,
        "pass_file": PASS_FILE,
        "union_file": UNION_FILE,
        "passes": generation_stats,
    }
    with open(GENERATION_REPORT, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    print("\nBLOCKING CANDIDATES FROZEN; ground truth was not loaded")
    print("Current pairs:", len(current_pairs))
    print("New union pairs:", len(cumulative_union))
    print("Additional pairs:", len(cumulative_union - current_pairs))
    print("Per-pass file:", PASS_FILE)
    print("Union file:", UNION_FILE)
    print("Generation report:", GENERATION_REPORT)


if __name__ == "__main__":
    main()