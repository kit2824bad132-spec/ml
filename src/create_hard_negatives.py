import pandas as pd
from rapidfuzz.fuzz import ratio
import random


S1_FILE = "dataset/train/train_source1_sample_50k.tsv"
S2_FILE = "dataset/train/train_source2_matches_50k.tsv"
S3_FILE = "dataset/train/train_source3_matches_50k.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"

OUTPUT_FILE = "dataset/train/hard_negative_pairs_50k.tsv"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading data...")

s1 = pd.read_csv(S1_FILE, sep="\t")
s2 = pd.read_csv(S2_FILE, sep="\t")
s3 = pd.read_csv(S3_FILE, sep="\t")

gt = pd.read_csv(GT_FILE, sep="\t")


print("S1:", len(s1))
print("S2:", len(s2))
print("S3:", len(s3))


# ============================================================
# 2. COMBINE S2 + S3
# ============================================================

candidates = pd.concat(
    [s2, s3],
    ignore_index=True
)


# ============================================================
# 3. NORMALIZE NAMES
# ============================================================

def clean_name(value):

    if pd.isna(value):
        return ""

    return str(value).lower().strip()


s1["clean_name"] = s1["business_name"].apply(
    clean_name
)

candidates["clean_name"] = candidates["business_name"].apply(
    clean_name
)


# ============================================================
# 4. GROUND TRUTH DICTIONARY
# ============================================================

gt_dict = {}

for _, row in gt.iterrows():

    source1_id = row["source1_entity_id"]

    value = row["matched_entity_ids"]

    if pd.isna(value) or str(value).strip() == "":

        gt_dict[source1_id] = set()

    else:

        gt_dict[source1_id] = set(
            str(value).split(",")
        )


print("Ground truth loaded.")


# ============================================================
# 5. CANDIDATE RECORDS
# ============================================================

candidate_records = list(
    zip(
        candidates["entity_id"],
        candidates["clean_name"]
    )
)


print(
    "Total candidate records:",
    len(candidate_records)
)


# ============================================================
# 6. GENERATE HARD NEGATIVES
# ============================================================

hard_negatives = []

random.seed(42)

print()
print("Generating hard negatives...")


for index, row in s1.iterrows():

    source1_id = row["entity_id"]

    source1_name = row["clean_name"]

    if source1_name == "":
        continue


    # True matches for this Source-1 entity

    true_matches = gt_dict.get(
        source1_id,
        set()
    )


    # Random sample of candidates

    sample_size = min(
        100,
        len(candidate_records)
    )

    sampled_candidates = random.sample(
        candidate_records,
        sample_size
    )


    scored = []


    # --------------------------------------------------------
    # Compare names
    # --------------------------------------------------------

    for candidate_id, candidate_name in sampled_candidates:

        # Never select a true match as a negative

        if candidate_id in true_matches:
            continue


        if candidate_name == "":
            continue


        similarity = ratio(
            source1_name,
            candidate_name
        )


        scored.append(
            (
                similarity,
                candidate_id
            )
        )


    # --------------------------------------------------------
    # Select hardest negatives
    # --------------------------------------------------------

    scored.sort(
        reverse=True
    )


    # Select top 2 difficult negatives

    for similarity, candidate_id in scored[:2]:

        hard_negatives.append(
            {
                "source1_entity_id": source1_id,
                "candidate_entity_id": candidate_id,
                "name_similarity": similarity,
                "label": 0
            }
        )


    # Progress

    if (index + 1) % 5000 == 0:

        print(
            "Processed:",
            index + 1,
            "S1 records"
        )


# ============================================================
# 7. SAVE
# ============================================================

hard_df = pd.DataFrame(
    hard_negatives
)


hard_df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)


# ============================================================
# 8. SUMMARY
# ============================================================

print()
print("================================")
print("HARD NEGATIVES CREATED")
print("================================")

print(
    "Hard negative pairs:",
    len(hard_df)
)

print(
    "Saved:",
    OUTPUT_FILE
)