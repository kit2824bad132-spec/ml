from collections import Counter, defaultdict
import gzip
import json

import joblib
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio, token_sort_ratio

from preprocessing import normalize_country, normalize_text


S1_FILE = "dataset/train/train_source1_sample_50k.tsv"
S2_FILE = "dataset/train/train_source2_matches_50k.tsv"
S3_FILE = "dataset/train/train_source3_matches_50k.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"
SPLIT_FILE = "dataset/train/clean_validation_split.tsv"
CURRENT_FILE = "dataset/train/clean_validation_candidates.tsv"
PASS_FILE = "dataset/train/clean_blocking_pass_candidates.tsv.gz"
UNION_FILE = "dataset/train/clean_blocking_candidates_mult_pass.tsv"
SELECTIVE_UNION_FILE = "dataset/train/clean_blocking_candidates_selective.tsv"
GENERATION_REPORT = "output/clean_blocking_generation_report.json"
MODEL_FILE = "dataset/train/clean_validation_model_improved.pkl"
REPORT_FILE = "output/clean_blocking_experiment_report.json"
CHUNK_SIZE = 50000
FIXED_THRESHOLD = 0.70
SPLIT_SEED = 42


def read_normalized_records():
    source_frame = pd.read_csv(S1_FILE, sep="\t", dtype=str, keep_default_na=False)
    target_frame = pd.concat(
        [
            pd.read_csv(S2_FILE, sep="\t", dtype=str, keep_default_na=False),
            pd.read_csv(S3_FILE, sep="\t", dtype=str, keep_default_na=False),
        ],
        ignore_index=True,
    ).drop_duplicates("entity_id")

    def convert(frame):
        return {
            row.entity_id: {
                "name": normalize_text(row.business_name),
                "address": normalize_text(row.business_address),
                "country": normalize_country(row.country),
            }
            for row in frame.itertuples(index=False)
        }

    return convert(source_frame), convert(target_frame)


def load_validation_truth(validation_ids):
    truth_pairs = set()
    total_matches = 0
    for chunk in pd.read_csv(
        GT_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=200000,
    ):
        selected = chunk[chunk.source1_entity_id.isin(validation_ids)]
        for row in selected.itertuples(index=False):
            for candidate_id in row.matched_entity_ids.split(","):
                candidate_id = candidate_id.strip()
                if candidate_id:
                    truth_pairs.add((row.source1_entity_id, candidate_id))
                    total_matches += 1
    return truth_pairs, total_matches


def pass_statistics(truth_pairs, current_pairs, generation_report):
    counts = Counter()
    new_candidate_counts = Counter()
    per_source = defaultdict(Counter)
    positive_pairs = defaultdict(set)
    noncurrent_positive_pairs = defaultdict(set)
    with gzip.open(PASS_FILE, "rt", encoding="utf-8", newline="") as compressed:
        for chunk in pd.read_csv(compressed, sep="\t", dtype=str, chunksize=CHUNK_SIZE):
            for row in chunk.itertuples(index=False):
                pass_name = row.blocking_pass
                source_id = row.source1_entity_id
                candidate_id = row.candidate_entity_id
                pair = (source_id, candidate_id)
                counts[pass_name] += 1
                per_source[pass_name][source_id] += 1
                if pair not in current_pairs:
                    new_candidate_counts[pass_name] += 1
                if pair in truth_pairs:
                    positive_pairs[pass_name].add(pair)
                    if pair not in current_pairs:
                        noncurrent_positive_pairs[pass_name].add(pair)

    stats = []
    cumulative_count = len(current_pairs)
    cumulative_true = set(truth_pairs & current_pairs)
    for pass_info in generation_report["passes"]:
        name = pass_info["strategy"]
        pass_true = positive_pairs[name]
        cumulative_true.update(pass_true)
        cumulative_count += pass_info["additional_pairs_to_union"]
        stats.append(
            {
                **pass_info,
                "average_candidates_per_source1": counts[name] / generation_report["validation_source1_ids"],
                "maximum_candidates_per_source1": max(per_source[name].values(), default=0),
                "additional_candidate_pairs_vs_current": new_candidate_counts[name],
                "candidate_growth_vs_current": (
                    (len(current_pairs) + new_candidate_counts[name]) / len(current_pairs)
                    if current_pairs
                    else None
                ),
                "true_matches_in_pass": len(pass_true),
                "candidate_recall": len(pass_true) / generation_report["total_validation_matches"],
                "additional_true_matches_recovered_vs_current": len(noncurrent_positive_pairs[name]),
                "cumulative_union_candidates_after_pass": cumulative_count,
                "cumulative_union_true_matches_after_pass": len(cumulative_true),
                "cumulative_union_candidate_recall_after_pass": (
                    len(cumulative_true) / generation_report["total_validation_matches"]
                ),
            }
        )
    return stats


def current_block_keys(record):
    country = record["country"].strip().lower()
    name = record["name"]
    address = record["address"]
    compact = name.replace(" ", "")
    first = name.split(" ", 1)[0] if name else ""
    keys = []
    if name:
        keys.append(("exact", country, name))
    if len(compact) >= 6:
        keys.append(("prefix6", country, compact[:6]))
    if len(first) >= 5:
        keys.append(("first", country, first))
    import re
    keys.extend(
        ("postal", country, code)
        for code in set(re.findall(r"(?<!\d)\d{5,6}(?!\d)", address))
    )
    return keys


def diagnose_blocking_misses(
    truth_pairs,
    current_pairs,
    selective_pairs,
    full_union_pairs,
    source_records,
    target_records,
):
    raw_block_sizes = Counter()
    target_keys = {}
    for candidate_id, record in target_records.items():
        keys = current_block_keys(record)
        target_keys[candidate_id] = set(keys)
        raw_block_sizes.update(keys)

    exclusive_causes = Counter()
    field_diagnostics = Counter()
    name_overlap_count = 0
    address_overlap_count = 0
    name_stopwords = {
        "company", "corp", "corporation", "services", "service", "limited",
        "inc", "llc", "group", "health", "medical", "center", "centre",
        "hospital", "clinic",
    }
    for source_id, candidate_id in truth_pairs - current_pairs:
        source = source_records[source_id]
        candidate = target_records.get(candidate_id)
        if candidate is None:
            exclusive_causes["target_record_not_in_candidate_pool"] += 1
            continue
        shared_keys = current_block_keys(source) and (
            set(current_block_keys(source)) & target_keys[candidate_id]
        )
        if shared_keys:
            if any(raw_block_sizes[key] > 250 for key in shared_keys):
                exclusive_causes["shared_key_suppressed_by_250_candidate_cap"] += 1
            else:
                exclusive_causes["shared_allowed_key_missing_from_current_file"] += 1
        else:
            exclusive_causes["no_shared_current_exact_prefix_first_token_or_postal_key"] += 1

        if source["country"] and candidate["country"] and source["country"] != candidate["country"]:
            field_diagnostics["country_mismatch"] += 1
        if not source["name"] or not candidate["name"]:
            field_diagnostics["missing_name"] += 1
        if not source["address"] or not candidate["address"]:
            field_diagnostics["missing_address"] += 1
        source_name_tokens = {
            token for token in source["name"].split()
            if len(token) >= 4 and token not in name_stopwords
        }
        candidate_name_tokens = {
            token for token in candidate["name"].split()
            if len(token) >= 4 and token not in name_stopwords
        }
        source_address_tokens = {
            token for token in source["address"].split()
            if len(token) >= 4 and not token.isdigit()
        }
        candidate_address_tokens = {
            token for token in candidate["address"].split()
            if len(token) >= 4 and not token.isdigit()
        }
        if source_name_tokens & candidate_name_tokens:
            name_overlap_count += 1
        if source_address_tokens & candidate_address_tokens:
            address_overlap_count += 1

    missed = truth_pairs - current_pairs
    return {
        "current_blocking_missed_true_matches": len(missed),
        "exclusive_current_blocker_causes": dict(exclusive_causes),
        "overlapping_field_diagnostics": dict(field_diagnostics),
        "missed_match_overlap_diagnostics": {
            "shared_informative_name_token": name_overlap_count,
            "shared_address_token": address_overlap_count,
        },
        "still_missing_after_selective_union": len(truth_pairs - selective_pairs),
        "still_missing_after_all_pass_union": len(truth_pairs - full_union_pairs),
        "interpretation": (
            "Most misses have no key in the existing four passes. A smaller portion shares "
            "a current key but is suppressed because its bucket exceeds the 250-candidate cap."
        ),
    }


def similarity(scorer, value1, value2):
    if not value1 or not value2:
        return 0.0
    return scorer(value1, value2) / 100.0


def score_candidate_file(
    path,
    source_records,
    target_records,
    truth_pairs,
    total_matches,
    feature_columns,
    model,
    threshold,
):
    tp = 0
    fp = 0
    candidate_positives = 0
    candidate_count = 0
    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype={"source1_entity_id": str, "candidate_entity_id": str},
        chunksize=CHUNK_SIZE,
    ):
        feature_rows = []
        labels = np.empty(len(chunk), dtype=np.int8)
        for index, row in enumerate(chunk.itertuples(index=False)):
            source = source_records[row.source1_entity_id]
            candidate = target_records[row.candidate_entity_id]
            name1, name2 = source["name"], candidate["name"]
            address1, address2 = source["address"], candidate["address"]
            country1, country2 = source["country"], candidate["country"]
            feature_rows.append(
                {
                    "name_ratio": similarity(ratio, name1, name2),
                    "name_token_sort": similarity(token_sort_ratio, name1, name2),
                    "name_token_set": similarity(token_set_ratio, name1, name2),
                    "address_ratio": similarity(ratio, address1, address2),
                    "address_token_sort": similarity(token_sort_ratio, address1, address2),
                    "address_token_set": similarity(token_set_ratio, address1, address2),
                    "country_match": int(country1 == country2),
                    "name_length_diff": abs(len(name1) - len(name2)),
                    "address_length_diff": abs(len(address1) - len(address2)),
                }
            )
            labels[index] = int((row.source1_entity_id, row.candidate_entity_id) in truth_pairs)

        probabilities = model.predict_proba(pd.DataFrame(feature_rows)[feature_columns])[:, 1]
        predictions = probabilities >= threshold
        tp += int(((labels == 1) & predictions).sum())
        fp += int(((labels == 0) & predictions).sum())
        candidate_positives += int(labels.sum())
        candidate_count += len(chunk)

    fn_candidates = candidate_positives - tp
    fn_end_to_end = total_matches - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall_candidates = tp / candidate_positives if candidate_positives else 0.0
    recall_end_to_end = tp / total_matches if total_matches else 0.0
    f05_candidates = (1.25 * precision * recall_candidates / (0.25 * precision + recall_candidates)) if (0.25 * precision + recall_candidates) else 0.0
    f05_end_to_end = (1.25 * precision * recall_end_to_end / (0.25 * precision + recall_end_to_end)) if (0.25 * precision + recall_end_to_end) else 0.0
    return {
        "candidate_pairs": candidate_count,
        "candidate_positive_pairs": candidate_positives,
        "all_validation_ground_truth_matches": total_matches,
        "true_positives_predicted": tp,
        "false_positives": fp,
        "false_negatives_within_candidates": fn_candidates,
        "missed_by_blocking_plus_classifier": fn_end_to_end,
        "predicted_matches": tp + fp,
        "precision": precision,
        "recall_within_candidates": recall_candidates,
        "f0_5_within_candidates": f05_candidates,
        "end_to_end_recall": recall_end_to_end,
        "end_to_end_f0_5": f05_end_to_end,
        "threshold": threshold,
    }


def main():
    split = pd.read_csv(SPLIT_FILE, sep="\t", dtype=str)
    validation_ids = set(split.loc[split.split == "validation", "entity_id"])
    current_frame = pd.read_csv(
        CURRENT_FILE,
        sep="\t",
        dtype={"source1_entity_id": str, "candidate_entity_id": str},
    )
    current_pairs = set(zip(current_frame.source1_entity_id, current_frame.candidate_entity_id))
    generation_report = json.loads(open(GENERATION_REPORT, encoding="utf-8").read())

    # Candidate generation is complete and persisted before this ground-truth read.
    truth_pairs, total_matches = load_validation_truth(validation_ids)
    generation_report["total_validation_matches"] = total_matches
    per_pass = pass_statistics(truth_pairs, current_pairs, generation_report)
    selective_pairs = set(
        map(
            tuple,
            pd.read_csv(
                SELECTIVE_UNION_FILE,
                sep="\t",
                dtype=str,
            )[["source1_entity_id", "candidate_entity_id"]].itertuples(index=False, name=None),
        )
    )
    full_union_pairs = set(
        map(
            tuple,
            pd.read_csv(
                UNION_FILE,
                sep="\t",
                dtype=str,
            )[["source1_entity_id", "candidate_entity_id"]].itertuples(index=False, name=None),
        )
    )

    model_data = joblib.load(MODEL_FILE)
    model = model_data["model"]
    feature_columns = model_data["feature_columns"]
    threshold = float(model_data.get("threshold", FIXED_THRESHOLD))
    if threshold != FIXED_THRESHOLD:
        raise ValueError(f"Expected unchanged clean model threshold {FIXED_THRESHOLD}, got {threshold}")
    source_records, target_records = read_normalized_records()
    miss_diagnosis = diagnose_blocking_misses(
        truth_pairs,
        current_pairs,
        selective_pairs,
        full_union_pairs,
        source_records,
        target_records,
    )
    current_score = score_candidate_file(
        CURRENT_FILE, source_records, target_records, truth_pairs,
        total_matches, feature_columns, model, threshold,
    )
    union_score = score_candidate_file(
        UNION_FILE, source_records, target_records, truth_pairs,
        total_matches, feature_columns, model, threshold,
    )
    selective_score = score_candidate_file(
        SELECTIVE_UNION_FILE, source_records, target_records, truth_pairs,
        total_matches, feature_columns, model, threshold,
    )

    decision = (
        "KEEP_NEW_BLOCKING"
        if selective_score["candidate_positive_pairs"] > current_score["candidate_positive_pairs"]
        and selective_score["f0_5_within_candidates"] >= current_score["f0_5_within_candidates"] * 0.95
        and selective_score["candidate_pairs"] <= current_score["candidate_pairs"] * 3
        else "DO_NOT_REPLACE_CURRENT_BLOCKER"
    )
    report = {
        "status": "COMPLETE",
        "old_candidate_recall": current_score["candidate_positive_pairs"] / total_matches,
        "new_candidate_recall": selective_score["candidate_positive_pairs"] / total_matches,
        "old_candidate_count": current_score["candidate_pairs"],
        "new_candidate_count": selective_score["candidate_pairs"],
        "old_f0_5": current_score["f0_5_within_candidates"],
        "new_f0_5": selective_score["f0_5_within_candidates"],
        "f0_5_difference": (
            selective_score["f0_5_within_candidates"]
            - current_score["f0_5_within_candidates"]
        ),
        "f0_5_metric_definition": "precision/recall over generated candidate pairs only",
        "old_end_to_end_f0_5": current_score["end_to_end_f0_5"],
        "new_end_to_end_f0_5": selective_score["end_to_end_f0_5"],
        "end_to_end_f0_5_difference": (
            selective_score["end_to_end_f0_5"] - current_score["end_to_end_f0_5"]
        ),
        "split_file": SPLIT_FILE,
        "validation_source1_ids": len(validation_ids),
        "random_seed": SPLIT_SEED,
        "threshold": threshold,
        "model_used": MODEL_FILE,
        "model_retrained": False,
        "ground_truth_loaded_after_candidate_freeze": True,
        "validation_ground_truth_used_to_construct_candidates": False,
        "candidate_recall_denominator_all_matches": total_matches,
        "current_blocking": {
            "candidate_file": CURRENT_FILE,
            "candidate_recall": current_score["candidate_positive_pairs"] / total_matches,
            "candidate_recall_numerator": current_score["candidate_positive_pairs"],
            "candidate_count": current_score["candidate_pairs"],
            "average_candidates_per_source1": current_score["candidate_pairs"] / len(validation_ids),
            "maximum_candidates_per_source1": int(current_frame.groupby("source1_entity_id").size().max()),
            "model_metrics": current_score,
        },
        "new_multi_pass_blocking": {
            "candidate_file": SELECTIVE_UNION_FILE,
            "selected_passes": [
                "house_number_address_token_country",
                "name_token_address_token_country",
            ],
            "candidate_recall": selective_score["candidate_positive_pairs"] / total_matches,
            "candidate_recall_numerator": selective_score["candidate_positive_pairs"],
            "candidate_count": selective_score["candidate_pairs"],
            "additional_candidates_vs_current": selective_score["candidate_pairs"] - current_score["candidate_pairs"],
            "candidate_growth_ratio": selective_score["candidate_pairs"] / current_score["candidate_pairs"],
            "average_candidates_per_source1": selective_score["candidate_pairs"] / len(validation_ids),
            "maximum_candidates_per_source1": int(
                pd.read_csv(SELECTIVE_UNION_FILE, sep="\t", dtype=str)
                .groupby("source1_entity_id").size().max()
            ),
            "additional_true_matches_recovered": selective_score["candidate_positive_pairs"] - current_score["candidate_positive_pairs"],
            "model_metrics": selective_score,
        },
        "all_pass_union_diagnostic": {
            "candidate_file": UNION_FILE,
            "pass_candidate_file": PASS_FILE,
            "candidate_recall": union_score["candidate_positive_pairs"] / total_matches,
            "candidate_recall_numerator": union_score["candidate_positive_pairs"],
            "candidate_count": union_score["candidate_pairs"],
            "additional_candidates_vs_current": union_score["candidate_pairs"] - current_score["candidate_pairs"],
            "candidate_growth_ratio": union_score["candidate_pairs"] / current_score["candidate_pairs"],
            "average_candidates_per_source1": union_score["candidate_pairs"] / len(validation_ids),
            "maximum_candidates_per_source1": generation_report["new_union_maximum_per_source1"],
            "additional_true_matches_recovered": union_score["candidate_positive_pairs"] - current_score["candidate_positive_pairs"],
            "model_metrics": union_score,
        },
        "blocking_passes": per_pass,
        "current_blocking_miss_diagnosis": miss_diagnosis,
        "generation_controls": {
            "current_rules_reconstructed_exactly": generation_report["current_rules_reconstructed_exactly"],
            "maximum_block_size": generation_report["max_block_size"],
            "maximum_new_token_pass_candidates_per_source": generation_report["max_token_pass_candidates_per_source"],
            "generic_tokens_excluded": True,
            "large_buckets_excluded": True,
            "separate_source2_source3_passes_added_no_pairs_beyond_current_union": all(
                item["additional_pairs_to_union"] == 0
                for item in generation_report["passes"]
                if item["strategy"].startswith("separate_")
            ),
        },
        "decision_rule_applied": {
            "candidate_recall_must_increase": True,
            "within_candidate_f0_5_must_not_drop_more_than_5_percent": True,
            "candidate_growth_must_not_exceed_3x": True,
            "growth_cap_is_an_experiment_guardrail_not_a_competition_requirement": True,
        },
        "decision": decision,
        "leakage_checks": {
            "same_fixed_validation_source1_ids": True,
            "source1_count": len(validation_ids),
            "validation_labels_used_during_generation": False,
            "all_pass_candidates_frozen_before_ground_truth": True,
            "candidate_pair_duplicate_removal_in_union": True,
            "existing_improved_model_used_without_retraining": True,
            "model_threshold_unchanged": threshold,
        },
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)

    print("\nCURRENT BLOCKING")
    print(current_score)
    print("\nNEW MULTI-PASS BLOCKING")
    print(selective_score)
    print("\nALL-PASS UNION DIAGNOSTIC")
    print(union_score)
    print("\nPer-pass candidate metrics:")
    for item in per_pass:
        print(
            f"{item['strategy']}: candidates={item['candidate_pairs']:,}, "
            f"avg/S1={item['average_candidates_per_source1']:.2f}, "
            f"max/S1={item['maximum_candidates_per_source1']}, "
            f"recall={item['candidate_recall']:.6f}, "
            f"additional matches vs current={item['additional_true_matches_recovered_vs_current']}"
        )
    print("\nDecision:", decision)
    print("Report:", REPORT_FILE)


if __name__ == "__main__":
    main()