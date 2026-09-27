import pandas as pd

from rapidfuzz.fuzz import (
    ratio,
    token_sort_ratio,
    token_set_ratio
)

from preprocessing import (
    normalize_text,
    normalize_country
)


PAIR_FILE = "dataset/train/hard_negative_pairs_50k.tsv"

S1_FILE = "dataset/train/train_source1_sample_50k.tsv"

S2_FILE = "dataset/train/train_source2_matches_50k.tsv"

S3_FILE = "dataset/train/train_source3_matches_50k.tsv"

OUTPUT_FILE = "dataset/train/hard_negative_features_50k.tsv"


print("Loading data...")

pairs = pd.read_csv(
    PAIR_FILE,
    sep="\t"
)

s1 = pd.read_csv(
    S1_FILE,
    sep="\t"
)

s2 = pd.read_csv(
    S2_FILE,
    sep="\t"
)

s3 = pd.read_csv(
    S3_FILE,
    sep="\t"
)


print("Hard negative pairs:", len(pairs))
print("S1:", len(s1))
print("S2:", len(s2))
print("S3:", len(s3))


# ============================================================
# COMBINE S2 + S3
# ============================================================

candidates = pd.concat(
    [s2, s3],
    ignore_index=True
)


# ============================================================
# CREATE LOOKUPS
# ============================================================

s1_lookup = s1.set_index(
    "entity_id"
).to_dict("index")


candidate_lookup = candidates.set_index(
    "entity_id"
).to_dict("index")


# ============================================================
# FEATURE CREATION
# ============================================================

features = []

print()
print("Creating features...")


for index, row in pairs.iterrows():

    s1_id = row["source1_entity_id"]

    candidate_id = row["candidate_entity_id"]


    source1 = s1_lookup.get(
        s1_id
    )

    candidate = candidate_lookup.get(
        candidate_id
    )


    if source1 is None or candidate is None:
        continue


    # --------------------------------------------------------
    # Normalize values
    # --------------------------------------------------------

    name1 = normalize_text(
        source1.get("business_name", "")
    )

    name2 = normalize_text(
        candidate.get("business_name", "")
    )


    address1 = normalize_text(
        source1.get("business_address", "")
    )

    address2 = normalize_text(
        candidate.get("business_address", "")
    )


    country1 = normalize_country(
        source1.get("country", "")
    )

    country2 = normalize_country(
        candidate.get("country", "")
    )


    # --------------------------------------------------------
    # Similarity features
    # --------------------------------------------------------

    name_ratio = ratio(
        name1,
        name2
    )

    name_token_sort = token_sort_ratio(
        name1,
        name2
    )

    name_token_set = token_set_ratio(
        name1,
        name2
    )


    address_ratio = ratio(
        address1,
        address2
    )

    address_token_sort = token_sort_ratio(
        address1,
        address2
    )

    address_token_set = token_set_ratio(
        address1,
        address2
    )


    # --------------------------------------------------------
    # Other features
    # --------------------------------------------------------

    country_match = int(
        country1 != ""
        and country1 == country2
    )


    name_length_diff = abs(
        len(name1) - len(name2)
    )


    address_length_diff = abs(
        len(address1) - len(address2)
    )


    features.append(
        {
            "source1_entity_id": s1_id,

            "candidate_entity_id": candidate_id,

            "name_ratio": name_ratio,

            "name_token_sort": name_token_sort,

            "name_token_set": name_token_set,

            "address_ratio": address_ratio,

            "address_token_sort": address_token_sort,

            "address_token_set": address_token_set,

            "country_match": country_match,

            "name_length_diff": name_length_diff,

            "address_length_diff": address_length_diff,

            "label": 0
        }
    )


    if (index + 1) % 10000 == 0:

        print(
            "Processed:",
            index + 1
        )


# ============================================================
# SAVE
# ============================================================

feature_df = pd.DataFrame(
    features
)


feature_df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)


print()
print("================================")
print("HARD NEGATIVE FEATURES CREATED")
print("================================")

print(
    "Rows:",
    len(feature_df)
)

print(
    "Saved:",
    OUTPUT_FILE
)
