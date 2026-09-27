import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score
)


FEATURE_FILE = "dataset/train/features_50k.tsv"


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(FEATURE_FILE, sep="\t")

print("Dataset:", df.shape)


# ============================================================
# 2. FEATURES
# ============================================================

feature_columns = [
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "address_ratio",
    "address_token_sort",
    "address_token_set",
    "country_match",
    "name_length_diff",
    "address_length_diff"
]

X = df[feature_columns]
y = df["label"]

# Important:
# Keep all pairs belonging to the same Source-1 entity
# in either train OR validation, not both.
groups = df["source1_entity_id"]


# ============================================================
# 3. TRAIN / VALIDATION SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]


print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_val))


# ============================================================
# 4. TRAIN MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")

model.fit(X_train, y_train)


# ============================================================
# 5. PREDICT PROBABILITIES
# ============================================================

probabilities = model.predict_proba(X_val)[:, 1]


# ============================================================
# 6. TEST MULTIPLE THRESHOLDS
# ============================================================

print("\n================================")
print("THRESHOLD ANALYSIS")
print("================================")

best_threshold = None
best_f05 = -1

for threshold in [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f05 = fbeta_score(
        y_val,
        predictions,
        beta=0.5,
        zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F0.5: {f05:.4f}"
    )

    if f05 > best_f05:
        best_f05 = f05
        best_threshold = threshold


# ============================================================
# 7. BEST RESULT
# ============================================================

print("\n================================")
print("BEST THRESHOLD")
print("================================")

print("Best threshold:", best_threshold)
print("Best F0.5:", round(best_f05, 4))