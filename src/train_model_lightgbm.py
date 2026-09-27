import os

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupShuffleSplit


FEATURE_FILE = "dataset/train/features_combined_50k.tsv"
MODEL_FILE = "output/model_lightgbm.pkl"
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
    "objective": "binary",
    "n_estimators": 500,
    "learning_rate": 0.04,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 2.0,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": 6,
    "verbosity": -1,
}


def macro_f05(y_true, y_pred, group_codes, group_count):
    true_positive = np.bincount(
        group_codes, weights=y_true * y_pred, minlength=group_count
    )
    false_positive = np.bincount(
        group_codes, weights=(1 - y_true) * y_pred, minlength=group_count
    )
    false_negative = np.bincount(
        group_codes, weights=y_true * (1 - y_pred), minlength=group_count
    )
    denominator = 1.25 * true_positive + 0.25 * false_negative + false_positive
    scores = np.divide(
        1.25 * true_positive,
        denominator,
        out=np.ones_like(denominator, dtype=float),
        where=denominator > 0,
    )
    return float(scores.mean())


def main():
    data = pd.read_csv(FEATURE_FILE, sep="\t")
    features = data[FEATURE_COLUMNS].fillna(0)
    labels = data["label"].astype(int).to_numpy()
    groups = data["source1_entity_id"].astype(str).to_numpy()

    train_indices, validation_indices = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
            features, labels, groups
        )
    )
    validation_groups, unique_groups = pd.factorize(
        groups[validation_indices], sort=False
    )
    validation_labels = labels[validation_indices]

    validation_model = LGBMClassifier(**MODEL_PARAMETERS)
    validation_model.fit(features.iloc[train_indices], labels[train_indices])
    validation_probabilities = validation_model.predict_proba(
        features.iloc[validation_indices]
    )[:, 1]

    threshold_scores = []
    for threshold in np.arange(0.05, 1.0, 0.01):
        predictions = (validation_probabilities >= threshold).astype(np.int8)
        score = macro_f05(
            validation_labels,
            predictions,
            validation_groups,
            len(unique_groups),
        )
        threshold_scores.append((score, float(threshold)))

    validation_score, best_threshold = max(threshold_scores)
    print(
        f"Grouped holdout: {len(validation_indices):,} pairs, "
        f"{len(unique_groups):,} S1 groups"
    )
    print(
        f"Best macro F0.5: {validation_score:.6f} "
        f"at threshold {best_threshold:.2f}"
    )

    final_model = LGBMClassifier(**MODEL_PARAMETERS)
    final_model.fit(features, labels)
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "feature_columns": FEATURE_COLUMNS,
            "threshold": best_threshold,
            "validation_macro_f05": validation_score,
            "model_parameters": MODEL_PARAMETERS,
        },
        MODEL_FILE,
    )
    print(f"Trained on all {len(data):,} available labeled pairs")
    print(f"Saved model: {MODEL_FILE}")


if __name__ == "__main__":
    main()