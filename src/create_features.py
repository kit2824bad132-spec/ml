import pandas as pd
import numpy as np

from rapidfuzz.fuzz import (
    ratio,
    token_sort_ratio,
    token_set_ratio
)

from preprocessing import (
    normalize_text,
    normalize_country
)


# ==================================================
# FILES
# ==================================================

PAIR_FILE = "dataset/train/training_pairs_50k.tsv"

S1_FILE = "dataset/train/train_source1_sample_50k.tsv"

S2_FILE = "dataset/train/train_source2_matches_50k.tsv"

S3_FILE = "dataset/train/train_source3_matches_50k.tsv"

OUTPUT = "dataset/train/features_50k.tsv"


# ==================================================
# 1. LOAD DATA
# ==================================================

print("Loading training pairs...")

pairs = pd.read_csv(
    PAIR_FILE,
    sep="\t"
)

print("Training pairs:", len(pairs))


print("Loading S1...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t"
)


print("Loading S2...")

s2 = pd.read_csv(
    S2_FILE,
    sep="\t"
)


print("Loading S3...")

s3 = pd.read_csv(
    S3_FILE,
    sep="\t"
)


# ==================================================
# 2. COMBINE S2 + S3
# ==================================================

candidates = pd.concat(
    [s2, s3],
    ignore_index=True
)

print(
    "Candidate records:",
    len(candidates)
)


# ==================================================
# 3. CREATE LOOKUPS
# ==================================================

s1_lookup = s1.set_index("entity_id")

candidate_lookup = candidates.set_index(
    "entity_id"
)


# ==================================================
# 4. SIMILARITY FUNCTIONS
# ==================================================

def safe_ratio(a, b):

    if not a or not b:
        return 0.0

    return ratio(a, b) / 100.0


def safe_token_sort(a, b):

    if not a or not b:
        return 0.0

    return token_sort_ratio(a, b) / 100.0


def safe_token_set(a, b):

    if not a or not b:
        return 0.0

    return token_set_ratio(a, b) / 100.0


# ==================================================
# 5. CREATE FEATURES
# ==================================================

features = []

print("\nCreating features...")
print("=" * 50)


for i, row in pairs.iterrows():

    s1_id = row["source1_entity_id"]

    candidate_id = row["candidate_entity_id"]


    # ----------------------------------------------
    # Get records
    # ----------------------------------------------

    source1 = s1_lookup.loc[s1_id]

    candidate = candidate_lookup.loc[candidate_id]


    # ----------------------------------------------
    # Get raw values
    # ----------------------------------------------

    name1 = source1["business_name"]
    name2 = candidate["business_name"]

    address1 = source1["business_address"]
    address2 = candidate["business_address"]

    country1 = source1["country"]
    country2 = candidate["country"]


    # ----------------------------------------------
    # Normalize
    # ----------------------------------------------

    name1 = normalize_text(name1)
    name2 = normalize_text(name2)

    address1 = normalize_text(address1)
    address2 = normalize_text(address2)

    country1 = normalize_country(country1)
    country2 = normalize_country(country2)


    # ==================================================
    # NAME FEATURES
    # ==================================================

    name_ratio = safe_ratio(
        name1,
        name2
    )

    name_token_sort = safe_token_sort(
        name1,
        name2
    )

    name_token_set = safe_token_set(
        name1,
        name2
    )


    # ==================================================
    # ADDRESS FEATURES
    # ==================================================

    address_ratio = safe_ratio(
        address1,
        address2
    )

    address_token_sort = safe_token_sort(
        address1,
        address2
    )

    address_token_set = safe_token_set(
        address1,
        address2
    )


    # ==================================================
    # COUNTRY FEATURE
    # ==================================================

    country_match = int(
        country1 == country2
    )


    # ==================================================
    # LENGTH FEATURES
    # ==================================================

    name_length_diff = abs(
        len(name1) - len(name2)
    )

    address_length_diff = abs(
        len(address1) - len(address2)
    )


    # ==================================================
    # SAVE
    # ==================================================

    features.append({

        "source1_entity_id":
            s1_id,

        "candidate_entity_id":
            candidate_id,

        "name_ratio":
            name_ratio,

        "name_token_sort":
            name_token_sort,

        "name_token_set":
            name_token_set,

        "address_ratio":
            address_ratio,

        "address_token_sort":
            address_token_sort,

        "address_token_set":
            address_token_set,

        "country_match":
            country_match,

        "name_length_diff":
            name_length_diff,

        "address_length_diff":
            address_length_diff,

        "label":
            row["label"]
    })


    # Progress
    if (i + 1) % 10000 == 0:

        print(
            f"Processed {i + 1:,} / {len(pairs):,}"
        )


# ==================================================
# 6. SAVE FEATURES
# ==================================================

features_df = pd.DataFrame(features)

features_df.to_csv(
    OUTPUT,
    sep="\t",
    index=False
)


# ==================================================
# 7. DISPLAY RESULT
# ==================================================

print("\n")
print("=" * 50)
print("FEATURE CREATION COMPLETE")
print("=" * 50)

print(
    "Rows:",
    len(features_df)
)

print(
    "Columns:",
    len(features_df.columns)
)

print("\nFeature columns:")

for column in features_df.columns:
    print(" -", column)

print("\nLabel distribution:")

print(
    features_df["label"].value_counts()
)

print("\nSaved to:")

print(OUTPUT)