import pandas as pd


BASE_FILE = "dataset/train/features_50k.tsv"
HARD_FILE = "dataset/train/hard_negative_features_50k.tsv"

OUTPUT_FILE = "dataset/train/features_combined_50k.tsv"


print("Loading base features...")

base = pd.read_csv(
    BASE_FILE,
    sep="\t"
)

print("Base rows:", len(base))


print("Loading hard-negative features...")

hard = pd.read_csv(
    HARD_FILE,
    sep="\t"
)

print("Hard-negative rows:", len(hard))


# ============================================================
# KEEP SAME COLUMNS
# ============================================================

feature_columns = [
    "source1_entity_id",
    "candidate_entity_id",
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "address_ratio",
    "address_token_sort",
    "address_token_set",
    "country_match",
    "name_length_diff",
    "address_length_diff",
    "label"
]


base = base[feature_columns]

hard = hard[feature_columns]


# ============================================================
# COMBINE
# ============================================================

combined = pd.concat(
    [base, hard],
    ignore_index=True
)


# Remove exact duplicate pairs if any
combined = combined.drop_duplicates(
    subset=[
        "source1_entity_id",
        "candidate_entity_id"
    ]
)


# ============================================================
# SHUFFLE
# ============================================================

combined = combined.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

combined.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("================================")
print("COMBINED DATASET")
print("================================")

print(
    "Total rows:",
    len(combined)
)

print()
print("Label distribution:")

print(
    combined["label"].value_counts()
)

print()
print("Saved:")
print(OUTPUT_FILE)
