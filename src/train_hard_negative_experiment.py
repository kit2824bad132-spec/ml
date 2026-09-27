import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit


S1_FILE = "dataset/train/train_source1_sample_50k.tsv"
BASE_FILE = "dataset/train/features_50k.tsv"
OLD_HARD_FILE = "dataset/train/hard_negative_features_50k.tsv"
NEW_HARD_FILE = "dataset/train/hard_negative_features_experiment.tsv"
MODEL_OUTPUT = "dataset/train/model_hard_negative_experiment.pkl"
REPORT_OUTPUT = "output/hard_negative_experiment_report.json"
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
THRESHOLD = 0.70
MODEL_PARAMETERS = {
    "n_estimators": 250,
    "max_depth": 14,
    "min_samples_leaf": 3,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


def read_features(path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype={"source1_entity_id": str, "candidate_entity_id": str},
    )


def score_model(name, model, validation):
    labels = validation.label.astype(int)
    probabilities = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    return {
        "name": name,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f05": float(fbeta_score(labels, predictions, beta=0.5, zero_division=0)),
        "predictions": int(predictions.sum()),
        "false_positives": int(false_positive),
        "false_negatives": int(false_negative),
        "true_positives": int(true_positive),
        "true_negatives": int(true_negative),
    }


def similarity_summary(frame, scale):
    name_similarity = frame.name_ratio / scale
    address_similarity = frame.address_ratio / scale
    return {
        "rows": len(frame),
        "name_similarity_quantiles": {
            str(key): float(value)
            for key, value in name_similarity.quantile(
                [0, .25, .5, .75, .9, .95, 1]
            ).items()
        },
        "address_similarity_quantiles": {
            str(key): float(value)
            for key, value in address_similarity.quantile(
                [0, .25, .5, .75, .9, .95, 1]
            ).items()
        },
        "name_at_least_0_80_rate": float((name_similarity >= 0.80).mean()),
        "address_at_least_0_80_rate": float((address_similarity >= 0.80).mean()),
    }


def main():
    source_ids = pd.read_csv(S1_FILE, sep="\t", dtype={"entity_id": str})
    source_ids = source_ids[["entity_id"]].sort_values("entity_id").reset_index(drop=True)
    train_indices, validation_indices = next(
        GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42).split(
            source_ids, groups=source_ids.entity_id
        )
    )
    train_ids = set(source_ids.iloc[train_indices].entity_id)
    validation_ids = set(source_ids.iloc[validation_indices].entity_id)
    base = read_features(BASE_FILE)
    old_hard = read_features(OLD_HARD_FILE)
    new_hard = read_features(NEW_HARD_FILE)

    base_train = base[base.source1_entity_id.isin(train_ids)]
    old_train = old_hard[old_hard.source1_entity_id.isin(train_ids)]
    new_train = new_hard[new_hard.source1_entity_id.isin(train_ids)]
    base_validation = base[base.source1_entity_id.isin(validation_ids)]
    new_validation = new_hard[new_hard.source1_entity_id.isin(validation_ids)]
    validation = pd.concat([base_validation, new_validation], ignore_index=True)

    baseline_training = pd.concat([base_train, old_train], ignore_index=True)
    improved_training = pd.concat([base_train, new_train], ignore_index=True)
    baseline_model = RandomForestClassifier(**MODEL_PARAMETERS)
    improved_model = RandomForestClassifier(**MODEL_PARAMETERS)

    print(f"Fixed source-1 split: {len(train_ids):,} train, {len(validation_ids):,} validation")
    print(f"Shared validation rows: {len(validation):,}; labels={validation.label.value_counts().to_dict()}")
    print("Training legacy-generation baseline...")
    baseline_model.fit(baseline_training[FEATURE_COLUMNS], baseline_training.label)
    print("Training improved-generation model...")
    improved_model.fit(improved_training[FEATURE_COLUMNS], improved_training.label)

    baseline_metrics = score_model("Baseline", baseline_model, validation)
    improved_metrics = score_model("Improved", improved_model, validation)
    difference = improved_metrics["f05"] - baseline_metrics["f05"]
    decision = (
        "IMPROVED"
        if difference > 0
        else "WORSE"
        if difference < 0
        else "NO SIGNIFICANT IMPROVEMENT"
    )

    print("\nSimilarity characteristics, hard negatives:")
    hard_negative_comparison = {}
    for name, frame, scale in (
        ("Legacy", old_hard, 100.0),
        ("Improved", new_hard, 1.0),
    ):
        summary = similarity_summary(frame, scale)
        hard_negative_comparison[name.lower()] = summary
        print(
            f"{name}: rows={len(frame):,}, name>=0.80="
            f"{summary['name_at_least_0_80_rate']:.4f}, "
            f"address>=0.80={summary['address_at_least_0_80_rate']:.4f}"
        )
        print("  name quantiles:", summary["name_similarity_quantiles"])
        print("  address quantiles:", summary["address_similarity_quantiles"])
    if "selection_reason" in new_hard:
        print("Selection reasons:", new_hard.selection_reason.value_counts().to_dict())
        print("Actual nonempty country-conflict rate:", float(new_hard.country_conflict.mean()))

    report = {
        "threshold": THRESHOLD,
        "fixed_split_random_state": 42,
        "train_source1_count": len(train_ids),
        "validation_source1_count": len(validation_ids),
        "validation_rows": len(validation),
        "validation_positive_rows": int(validation.label.sum()),
        "validation_negative_rows": int((validation.label == 0).sum()),
        "baseline_training_rows": len(baseline_training),
        "improved_training_rows": len(improved_training),
        "baseline": baseline_metrics,
        "improved": improved_metrics,
        "hard_negative_comparison": hard_negative_comparison,
        "f05_difference": difference,
        "decision": decision,
        "model_parameters": MODEL_PARAMETERS,
    }
    Path(REPORT_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUTPUT, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    joblib.dump(
        {
            "model": improved_model,
            "feature_columns": FEATURE_COLUMNS,
            "threshold": THRESHOLD,
            "validation_metrics": improved_metrics,
            "validation_f05_difference": difference,
            "decision": decision,
            "model_parameters": MODEL_PARAMETERS,
        },
        MODEL_OUTPUT,
    )

    print("\nBaseline:", baseline_metrics)
    print("Improved:", improved_metrics)
    print(f"F0.5 difference: {difference:+.6f}")
    print("Decision:", decision)
    print("Experiment model:", MODEL_OUTPUT)
    print("Report:", REPORT_OUTPUT)


if __name__ == "__main__":
    main()