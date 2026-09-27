import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier

FEATURE_FILE = "dataset/train/features_combined_50k.tsv"
MODEL_FILE = "dataset/train/model_hard.pkl"

print("Loading dataset...")
df = pd.read_csv(FEATURE_FILE, sep="\t")

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

print("Rows:", len(df))
print("Training final model...")

model = RandomForestClassifier(
    n_estimators=250,
    max_depth=14,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train on ALL available training pairs
model.fit(X, y)

joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns,
        "threshold": 0.70
    },
    MODEL_FILE
)

print()
print("==============================")
print("MODEL SAVED")
print("==============================")
print("Model:", MODEL_FILE)
print("Threshold: 0.70")
print("Training rows:", len(df))