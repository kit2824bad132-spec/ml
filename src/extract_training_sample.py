import pandas as pd
import numpy as np

SOURCE1 = "dataset/train/train_source1.tsv"
GROUND_TRUTH = "dataset/train/train_ground_truth.tsv"

OUTPUT = "dataset/train/train_source1_sample_50k.tsv"

TARGET = 50_000
RANDOM_STATE = 42

# --------------------------------------------------
# 1. Read ONLY the ground truth
# --------------------------------------------------

print("Reading ground truth...")

gt = pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    usecols=["source1_entity_id", "matched_entity_ids"]
)

# Empty = no match
gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

gt["match_count"] = gt["matched_entity_ids"].apply(
    lambda x: 0 if x.strip() == "" else len(x.split(","))
)

print("\nGround truth distribution:")
print(gt["match_count"].value_counts().sort_index())


# --------------------------------------------------
# 2. Create balanced sample of S1 IDs
# --------------------------------------------------

# We want examples from:
# 0 matches
# 1 match
# 2 matches
# 3+ matches

gt["group"] = np.where(
    gt["match_count"] == 0,
    "zero",
    np.where(
        gt["match_count"] == 1,
        "one",
        np.where(
            gt["match_count"] == 2,
            "two",
            "three_plus"
        )
    )
)

print("\nGroups:")
print(gt["group"].value_counts())


# Sample roughly equally from each group
groups = []

group_names = ["zero", "one", "two", "three_plus"]

per_group = TARGET // len(group_names)

for group_name in group_names:

    group = gt[gt["group"] == group_name]

    n = min(per_group, len(group))

    sample = group.sample(
        n=n,
        random_state=RANDOM_STATE
    )

    groups.append(sample)

sample_gt = pd.concat(groups)

# If we don't have exactly 50k, fill the remaining
# from all remaining records.

remaining_needed = TARGET - len(sample_gt)

if remaining_needed > 0:

    remaining = gt[
        ~gt["source1_entity_id"].isin(
            sample_gt["source1_entity_id"]
        )
    ]

    extra = remaining.sample(
        n=min(remaining_needed, len(remaining)),
        random_state=RANDOM_STATE
    )

    sample_gt = pd.concat([sample_gt, extra])


# Shuffle
sample_gt = sample_gt.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


print("\nSelected S1 records:", len(sample_gt))


# --------------------------------------------------
# 3. Extract only those rows from the huge S1 file
# --------------------------------------------------

wanted_ids = set(sample_gt["source1_entity_id"])

print("\nScanning Source 1 file in chunks...")

chunks = []

for chunk in pd.read_csv(
    SOURCE1,
    sep="\t",
    chunksize=100_000
):

    selected = chunk[
        chunk["entity_id"].isin(wanted_ids)
    ]

    if len(selected) > 0:
        chunks.append(selected)

    print(
        f"Found {sum(len(x) for x in chunks)} / {TARGET}",
        end="\r"
    )


sample_s1 = pd.concat(chunks, ignore_index=True)


# --------------------------------------------------
# 4. Save
# --------------------------------------------------

sample_s1.to_csv(
    OUTPUT,
    sep="\t",
    index=False
)

print("\n\nDone!")

print("Rows:", len(sample_s1))
print("Saved:", OUTPUT)