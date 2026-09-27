import pandas as pd
import random

S1_FILE = "dataset/train/train_source1_sample_50k.tsv"
S2_FILE = "dataset/train/train_source2_matches_50k.tsv"
S3_FILE = "dataset/train/train_source3_matches_50k.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"

OUTPUT = "dataset/train/training_pairs_50k.tsv"

RANDOM_STATE = 42
NEGATIVES_PER_S1 = 2

random.seed(RANDOM_STATE)


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

print("Loading S1...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t"
)

print("S1:", len(s1))

print("Loading S2...")

s2 = pd.read_csv(
    S2_FILE,
    sep="\t"
)

print("S2:", len(s2))

print("Loading S3...")

s3 = pd.read_csv(
    S3_FILE,
    sep="\t"
)

print("S3:", len(s3))


# Combine S2 + S3
candidates = pd.concat(
    [s2, s3],
    ignore_index=True
)

print("Total S2/S3:", len(candidates))


# --------------------------------------------------
# 2. Load ground truth
# --------------------------------------------------

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t"
)

gt = gt[
    gt["source1_entity_id"].isin(
        set(s1["entity_id"])
    )
].copy()

gt["matched_entity_ids"] = (
    gt["matched_entity_ids"]
    .fillna("")
)


# --------------------------------------------------
# 3. Create dictionary:
#    S1 ID -> true matches
# --------------------------------------------------

true_matches = {}

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    value = row["matched_entity_ids"]

    if value.strip() == "":
        true_matches[s1_id] = set()
    else:
        true_matches[s1_id] = set(
            x.strip()
            for x in value.split(",")
            if x.strip()
        )


# --------------------------------------------------
# 4. Candidate lookup
# --------------------------------------------------

candidate_dict = {}

for _, row in candidates.iterrows():

    entity_id = row["entity_id"]

    candidate_dict[entity_id] = row


all_candidate_ids = list(candidate_dict.keys())


# --------------------------------------------------
# 5. Create positive pairs
# --------------------------------------------------

pairs = []

print("\nCreating positive pairs...")

for s1_id, matches in true_matches.items():

    for match_id in matches:

        if match_id not in candidate_dict:
            continue

        pairs.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": match_id,
            "label": 1
        })


print("Positive pairs:", len(pairs))


# --------------------------------------------------
# 6. Create negative pairs
# --------------------------------------------------

print("Creating negative pairs...")

for s1_id, matches in true_matches.items():

    # Select random non-matching records
    negative_count = 0

    attempts = 0

    while negative_count < NEGATIVES_PER_S1:

        attempts += 1

        candidate_id = random.choice(
            all_candidate_ids
        )

        if candidate_id in matches:
            continue

        pairs.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": candidate_id,
            "label": 0
        })

        negative_count += 1

        if attempts > 100:
            break


# --------------------------------------------------
# 7. Save
# --------------------------------------------------

pairs_df = pd.DataFrame(pairs)

pairs_df = pairs_df.drop_duplicates(
    subset=[
        "source1_entity_id",
        "candidate_entity_id"
    ]
)

pairs_df.to_csv(
    OUTPUT,
    sep="\t",
    index=False
)


# --------------------------------------------------
# 8. Statistics
# --------------------------------------------------

print("\n================================")
print("TRAINING PAIRS CREATED")
print("================================")

print("Total pairs:", len(pairs_df))

print("\nLabel distribution:")
print(
    pairs_df["label"].value_counts()
)

print("\nSaved:")
print(OUTPUT)