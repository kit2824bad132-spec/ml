import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio, token_sort_ratio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

from preprocessing import normalize_country, normalize_text


S1_FILE = "dataset/train/train_source1_sample_50k.tsv"
S2_FILE = "dataset/train/train_source2_matches_50k.tsv"
S3_FILE = "dataset/train/train_source3_matches_50k.tsv"
SPLIT_FILE = "dataset/train/clean_validation_split.tsv"
BASELINE_PAIRS_FILE = "dataset/train/clean_validation_training_pairs_baseline.tsv"
IMPROVED_PAIRS_FILE = "dataset/train/clean_validation_training_pairs_improved.tsv"
VALIDATION_FILE = "dataset/train/clean_validation_labeled_candidates.tsv"
AUDIT_FILE = "output/clean_validation_label_audit.json"
REPORT_FILE = "output/clean_validation_report.json"
CALIBRATION_SPLIT_FILE = "dataset/train/clean_validation_calibration_assessment_split.tsv"
ERROR_FILE = "output/clean_validation_error_cases.tsv"
BASELINE_MODEL_FILE = "dataset/train/clean_validation_model_baseline.pkl"
IMPROVED_MODEL_FILE = "dataset/train/clean_validation_model_improved.pkl"

RANDOM_SEED = 42
CALIBRATION_SEED = 271828
FIXED_THRESHOLD = 0.70
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
FEATURE_COLUMNS = [
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "address_ratio",
    "address_token_sort",
    "address_token_set",
    "country_match",
    "name_length_diff",
    "address_length_diff",
]
MODEL_PARAMETERS = {
    "n_estimators": 250,
    "max_depth": 14,
    "min_samples_leaf": 3,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


def read_pairs(path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype={"source1_entity_id": str, "candidate_entity_id": str},
    )


def prepare_records():
    source = pd.read_csv(S1_FILE, sep="\t", dtype=str, keep_default_na=False)
    targets = pd.concat(
        [
            pd.read_csv(S2_FILE, sep="\t", dtype=str, keep_default_na=False),
            pd.read_csv(S3_FILE, sep="\t", dtype=str, keep_default_na=False),
        ],
        ignore_index=True,
    ).drop_duplicates("entity_id")

    def normalize_frame(frame):
        result = {}
        for row in frame.itertuples(index=False):
            result[row.entity_id] = {
                "name": normalize_text(row.business_name),
                "address": normalize_text(row.business_address),
                "country": normalize_country(row.country),
            }
        return result

    return normalize_frame(source), normalize_frame(targets)


def safe_similarity(scorer, value1, value2):
    if not value1 or not value2:
        return 0.0
    return scorer(value1, value2) / 100.0


def make_features(pairs, source_records, candidate_records, include_raw=False):
    rows = []
    for pair in pairs.itertuples(index=False):
        source = source_records.get(pair.source1_entity_id)
        candidate = candidate_records.get(pair.candidate_entity_id)
        if source is None or candidate is None:
            continue
        name1, name2 = source["name"], candidate["name"]
        address1, address2 = source["address"], candidate["address"]
        country1, country2 = source["country"], candidate["country"]
        row = {
            "source1_entity_id": pair.source1_entity_id,
            "candidate_entity_id": pair.candidate_entity_id,
            "name_ratio": safe_similarity(ratio, name1, name2),
            "name_token_sort": safe_similarity(token_sort_ratio, name1, name2),
            "name_token_set": safe_similarity(token_set_ratio, name1, name2),
            "address_ratio": safe_similarity(ratio, address1, address2),
            "address_token_sort": safe_similarity(token_sort_ratio, address1, address2),
            "address_token_set": safe_similarity(token_set_ratio, address1, address2),
            "country_match": int(country1 == country2),
            "name_length_diff": abs(len(name1) - len(name2)),
            "address_length_diff": abs(len(address1) - len(address2)),
            "label": int(pair.label),
        }
        if hasattr(pair, "pair_type"):
            row["pair_type"] = pair.pair_type
        if hasattr(pair, "selection_reason"):
            row["selection_reason"] = pair.selection_reason
        if include_raw:
            row.update(
                {
                    "source_name": name1,
                    "candidate_name": name2,
                    "source_address": address1,
                    "candidate_address": address2,
                    "source_country": country1,
                    "candidate_country": country2,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def metric_row(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f0_5": float(fbeta_score(labels, predictions, beta=0.5, zero_division=0)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "predicted_matches": int(predictions.sum()),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


def threshold_metrics(labels, probabilities):
    return [metric_row(labels, probabilities, threshold) for threshold in THRESHOLDS]


def classify_errors(scored, source_records, candidate_records):
    categories = {
        "similar_business_names": 0,
        "different_addresses": 0,
        "same_address_different_business": 0,
        "country_mismatch": 0,
        "missing_name": 0,
        "missing_address": 0,
        "numeric_address_mismatch": 0,
        "duplicate_entities": 0,
        "other": 0,
    }
    examples = []
    for row in scored.itertuples(index=False):
        source = source_records[row.source1_entity_id]
        candidate = candidate_records[row.candidate_entity_id]
        name1, name2 = source["name"], candidate["name"]
        address1, address2 = source["address"], candidate["address"]
        country1, country2 = source["country"], candidate["country"]
        flags = {
            "similar_business_names": bool(name1 and name2 and row.name_ratio >= 0.80),
            "different_addresses": bool(address1 and address2 and row.address_ratio < 0.50),
            "same_address_different_business": bool(
                address1 and address2 and name1 and name2
                and row.address_ratio >= 0.80 and row.name_ratio < 0.50
            ),
            "country_mismatch": bool(country1 and country2 and country1 != country2),
            "missing_name": not name1 or not name2,
            "missing_address": not address1 or not address2,
            "numeric_address_mismatch": False,
            "duplicate_entities": bool(
                name1 and address1 and name1 == name2 and address1 == address2
                and country1 == country2
            ),
        }
        digits1 = set(re.findall(r"\d+", address1))
        digits2 = set(re.findall(r"\d+", address2))
        flags["numeric_address_mismatch"] = bool(digits1 and digits2 and digits1 != digits2)
        matched = False
        for category, is_match in flags.items():
            if is_match:
                categories[category] += 1
                matched = True
        if not matched:
            categories["other"] += 1
        examples.append(
            {
                "source1_entity_id": row.source1_entity_id,
                "candidate_entity_id": row.candidate_entity_id,
                "actual_label": int(row.label),
                "predicted_label": int(row.prediction),
                "probability": float(row.probability),
                "name_ratio": float(row.name_ratio),
                "address_ratio": float(row.address_ratio),
                "source_country": country1,
                "candidate_country": country2,
                "categories": ",".join(name for name, value in flags.items() if value) or "other",
                "source_name": name1,
                "candidate_name": name2,
                "source_address": address1,
                "candidate_address": address2,
            }
        )
    return categories, examples


def main():
    split = pd.read_csv(SPLIT_FILE, sep="\t", dtype=str)
    train_ids = set(split.loc[split.split == "train", "entity_id"])
    validation_ids = sorted(split.loc[split.split == "validation", "entity_id"])
    baseline_pairs = read_pairs(BASELINE_PAIRS_FILE)
    improved_pairs = read_pairs(IMPROVED_PAIRS_FILE)
    validation_pairs = pd.read_csv(
        VALIDATION_FILE,
        sep="\t",
        dtype={"source1_entity_id": str, "candidate_entity_id": str},
    )
    label_audit = json.loads(Path(AUDIT_FILE).read_text(encoding="utf-8"))

    assert set(baseline_pairs.source1_entity_id) <= train_ids
    assert set(improved_pairs.source1_entity_id) <= train_ids
    assert not (set(validation_pairs.source1_entity_id) & train_ids)
    assert validation_pairs.duplicated(["source1_entity_id", "candidate_entity_id"]).sum() == 0

    source_records, candidate_records = prepare_records()
    baseline_features = make_features(baseline_pairs, source_records, candidate_records)
    improved_features = make_features(improved_pairs, source_records, candidate_records)
    validation_features = make_features(
        validation_pairs, source_records, candidate_records, include_raw=True
    )
    baseline_features.to_csv(
        "dataset/train/clean_validation_training_features_baseline.tsv", sep="\t", index=False
    )
    improved_features.to_csv(
        "dataset/train/clean_validation_training_features_improved.tsv", sep="\t", index=False
    )
    validation_features.to_csv(
        "dataset/train/clean_validation_validation_features.tsv", sep="\t", index=False
    )

    assert set(FEATURE_COLUMNS) <= set(validation_features.columns)
    assert all(validation_features[column].notna().all() for column in FEATURE_COLUMNS)
    assert validation_features[FEATURE_COLUMNS].select_dtypes(exclude=[np.number]).empty

    train_groups = set(baseline_features.source1_entity_id) | set(improved_features.source1_entity_id)
    validation_groups = set(validation_features.source1_entity_id)
    assert not train_groups & validation_groups

    baseline_model = RandomForestClassifier(**MODEL_PARAMETERS)
    improved_model = RandomForestClassifier(**MODEL_PARAMETERS)
    print(f"Training rows: baseline={len(baseline_features):,}, improved={len(improved_features):,}")
    print(f"Frozen validation rows: {len(validation_features):,}")
    print("Training baseline model...")
    baseline_model.fit(baseline_features[FEATURE_COLUMNS], baseline_features.label)
    print("Training improved hard-negative model...")
    improved_model.fit(improved_features[FEATURE_COLUMNS], improved_features.label)

    labels = validation_features.label.to_numpy(dtype=np.int8)
    baseline_probabilities = baseline_model.predict_proba(validation_features[FEATURE_COLUMNS])[:, 1]
    improved_probabilities = improved_model.predict_proba(validation_features[FEATURE_COLUMNS])[:, 1]

    # Primary comparison uses the prespecified 0.70 threshold on every frozen validation pair.
    fixed_baseline = metric_row(labels, baseline_probabilities, FIXED_THRESHOLD)
    fixed_improved = metric_row(labels, improved_probabilities, FIXED_THRESHOLD)

    # Threshold selection occurs only after the fixed-threshold comparison. Half of validation
    # source IDs calibrate thresholds; the other half remains untouched for assessment.
    calibration_indices, assessment_indices = next(
        GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=CALIBRATION_SEED).split(
            pd.DataFrame({"entity_id": validation_ids}),
            groups=validation_ids,
        )
    )
    calibration_ids = {validation_ids[index] for index in calibration_indices}
    assessment_ids = {validation_ids[index] for index in assessment_indices}
    calibration_split = pd.DataFrame(
        {
            "entity_id": validation_ids,
            "role": [
                "calibration" if entity_id in calibration_ids else "assessment"
                for entity_id in validation_ids
            ],
        }
    )
    calibration_split.to_csv(CALIBRATION_SPLIT_FILE, sep="\t", index=False)
    calibration_mask = validation_features.source1_entity_id.isin(calibration_ids).to_numpy()
    assessment_mask = validation_features.source1_entity_id.isin(assessment_ids).to_numpy()
    y_calibration = labels[calibration_mask]
    y_assessment = labels[assessment_mask]

    baseline_sweep = threshold_metrics(y_calibration, baseline_probabilities[calibration_mask])
    improved_sweep = threshold_metrics(y_calibration, improved_probabilities[calibration_mask])
    best_calibrated = max(improved_sweep, key=lambda row: row["f0_5"])
    selected_threshold = best_calibrated["threshold"]

    assessment_baseline_fixed = metric_row(
        y_assessment, baseline_probabilities[assessment_mask], FIXED_THRESHOLD
    )
    assessment_improved_fixed = metric_row(
        y_assessment, improved_probabilities[assessment_mask], FIXED_THRESHOLD
    )
    assessment_baseline_selected = metric_row(
        y_assessment, baseline_probabilities[assessment_mask], selected_threshold
    )
    assessment_improved_selected = metric_row(
        y_assessment, improved_probabilities[assessment_mask], selected_threshold
    )

    assessment_scored = validation_features.loc[assessment_mask].copy()
    assessment_scored["probability"] = improved_probabilities[assessment_mask]
    assessment_scored["prediction"] = (
        improved_probabilities[assessment_mask] >= selected_threshold
    ).astype(np.int8)
    false_positives = assessment_scored[
        (assessment_scored.label == 0) & (assessment_scored.prediction == 1)
    ]
    false_negatives = assessment_scored[
        (assessment_scored.label == 1) & (assessment_scored.prediction == 0)
    ]
    fp_categories, fp_examples = classify_errors(false_positives, source_records, candidate_records)
    fn_categories, fn_examples = classify_errors(false_negatives, source_records, candidate_records)
    pd.DataFrame(fp_examples + fn_examples).to_csv(ERROR_FILE, sep="\t", index=False)

    decision_difference = fixed_improved["f0_5"] - fixed_baseline["f0_5"]
    decision = (
        "IMPROVED"
        if decision_difference > 0
        else "WORSE"
        if decision_difference < 0
        else "NO SIGNIFICANT IMPROVEMENT"
    )
    report = {
        "status": "CLEAN",
        "split": {
            "file": "dataset/train/clean_validation_split.tsv",
            "random_seed": RANDOM_SEED,
            "training_source1_ids": len(train_ids),
            "validation_source1_ids": len(validation_ids),
            "source1_id_overlap": 0,
            "training_pair_source1_ids": len(train_groups),
            "validation_pair_source1_ids": len(validation_groups),
            "train_validation_pair_source1_overlap": len(train_groups & validation_groups),
            "calibration_assessment_seed": CALIBRATION_SEED,
            "calibration_source1_ids": len(calibration_ids),
            "assessment_source1_ids": len(assessment_ids),
            "calibration_assessment_source1_overlap": len(calibration_ids & assessment_ids),
            "calibration_assessment_split_file": CALIBRATION_SPLIT_FILE,
        },
        "target_entity_overlap": {
            "training_unique_matched_target_ids": label_audit["training_unique_matched_target_ids"],
            "validation_unique_matched_target_ids": label_audit["validation_unique_matched_target_ids"],
            "overlap_count": label_audit["matched_target_id_overlap_count"],
        },
        "training": {
            "baseline_rows": len(baseline_features),
            "baseline_positives": int(baseline_features.label.sum()),
            "baseline_negatives": int((baseline_features.label == 0).sum()),
            "baseline_legacy_hard_negatives": int((baseline_features.pair_type == "legacy_hard_negative").sum()),
            "improved_rows": len(improved_features),
            "improved_positives": int(improved_features.label.sum()),
            "improved_negatives": int((improved_features.label == 0).sum()),
            "improved_hard_negatives": int(improved_features.pair_type.isin(["name_confusion", "address_confusion", "country_confusion"]).sum()),
            "training_source1_ids_only": True,
        },
        "validation": {
            "candidate_generation": "country-scoped bounded exact name, name prefix6, first token >=5 chars, and postal code blocks",
            "candidate_pairs": label_audit["validation_candidate_pairs"],
            "validation_positive_candidates": label_audit["labeled_validation_positive_candidate_pairs"],
            "validation_negative_candidates": label_audit["labeled_validation_negative_candidate_pairs"],
            "total_ground_truth_matches": label_audit["validation_truth_match_count"],
            "true_matches_in_candidates": label_audit["validation_truth_pairs_present_in_candidates"],
            "candidate_recall": label_audit["candidate_recall"],
            "candidate_recall_denominator": "all ground-truth matched target IDs for validation source-1 IDs",
            "candidate_pair_file_frozen_before_ground_truth": label_audit["candidate_file_frozen_before_ground_truth_read"],
        },
        "features": {
            "used": FEATURE_COLUMNS,
            "same_builder_for_train_and_validation": True,
            "uses_ground_truth_fields": False,
            "uses_label_or_ids_as_model_features": False,
            "similarity_range": [
                float(validation_features[FEATURE_COLUMNS[:6]].min().min()),
                float(validation_features[FEATURE_COLUMNS[:6]].max().max()),
            ],
        },
        "fixed_threshold_comparison": {
            "threshold": FIXED_THRESHOLD,
            "validation_rows": len(validation_features),
            "positive_samples": int(labels.sum()),
            "negative_samples": int((labels == 0).sum()),
            "baseline": fixed_baseline,
            "improved": fixed_improved,
            "improved_minus_baseline_f0_5": decision_difference,
            "decision": decision,
        },
        "threshold_sweep": {
            "performed_after_fixed_threshold_comparison": True,
            "threshold_selection_subset": "calibration source-1 IDs only",
            "calibration_candidate_rows": int(calibration_mask.sum()),
            "assessment_candidate_rows": int(assessment_mask.sum()),
            "baseline": baseline_sweep,
            "improved": improved_sweep,
            "selected_threshold_from_improved_calibration_f0_5": selected_threshold,
            "independent_assessment_at_fixed_threshold": {
                "baseline": assessment_baseline_fixed,
                "improved": assessment_improved_fixed,
            },
            "independent_assessment_at_selected_threshold": {
                "baseline": assessment_baseline_selected,
                "improved": assessment_improved_selected,
            },
        },
        "error_analysis": {
            "subset": "assessment source-1 IDs, improved model, calibration-selected threshold",
            "threshold": selected_threshold,
            "false_positive_count": len(false_positives),
            "false_negative_count": len(false_negatives),
            "false_positive_category_counts_nonexclusive": fp_categories,
            "false_negative_category_counts_nonexclusive": fn_categories,
            "blocking_failures": {
                "count": label_audit["validation_truth_match_count"] - label_audit["validation_truth_pairs_present_in_candidates"],
                "definition": "ground-truth matches absent from the frozen candidate set; not model false negatives",
            },
            "cases_file": ERROR_FILE,
            "category_definitions": {
                "similar_business_names": "both names present and name_ratio >= 0.80",
                "different_addresses": "both addresses present and address_ratio < 0.50",
                "same_address_different_business": "address_ratio >= 0.80 and name_ratio < 0.50",
                "country_mismatch": "both country fields present and unequal",
                "missing_name_or_address": "either pair side has an empty normalized field",
                "numeric_address_mismatch": "both addresses contain digits and digit-token sets differ",
                "duplicate_entities": "same nonempty normalized name/address/country across the pair",
                "other": "none of the listed measurable conditions apply",
            },
        },
        "leakage_checks": {
            "source1_train_validation_overlap": 0,
            "training_pairs_use_training_source1_only": True,
            "hard_negatives_generated_for_validation_source1": 0,
            "validation_candidate_generation_loaded_ground_truth": False,
            "validation_labels_used_to_construct_candidate_set": False,
            "validation_ground_truth_used_only_after_candidate_file_freeze": True,
            "validation_rows_used_in_training": False,
            "pair_overlap_between_train_and_validation_source1_groups": 0,
            "duplicate_validation_candidate_pairs": label_audit["duplicate_validation_candidate_pairs"],
            "matched_target_entity_overlap": label_audit["matched_target_id_overlap_count"],
            "threshold_fixed_comparison_is_prespecified_0_70": True,
            "threshold_sweep_uses_separate_calibration_and_assessment_source_ids": True,
        },
        "model": {
            "type": "RandomForestClassifier",
            "parameters": MODEL_PARAMETERS,
            "baseline_file": BASELINE_MODEL_FILE,
            "improved_file": IMPROVED_MODEL_FILE,
        },
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    joblib.dump(
        {"model": baseline_model, "feature_columns": FEATURE_COLUMNS, "threshold": FIXED_THRESHOLD},
        BASELINE_MODEL_FILE,
    )
    joblib.dump(
        {"model": improved_model, "feature_columns": FEATURE_COLUMNS, "threshold": FIXED_THRESHOLD},
        IMPROVED_MODEL_FILE,
    )

    print("\nCLEAN VALIDATION RESULTS")
    print("Baseline @0.70:", fixed_baseline)
    print("Improved @0.70:", fixed_improved)
    print("Fixed-threshold F0.5 difference:", f"{decision_difference:+.6f}")
    print("Decision:", decision)
    print("Calibration-selected threshold:", selected_threshold)
    print("Improved independent assessment @ selected threshold:", assessment_improved_selected)
    print("FP categories:", fp_categories)
    print("FN categories:", fn_categories)
    print("Candidate recall:", f"{label_audit['candidate_recall']:.6f}")
    print("Report:", REPORT_FILE)


if __name__ == "__main__":
    main()