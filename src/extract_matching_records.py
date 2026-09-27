import pandas as pd

GT_FILE = "dataset/train/train_ground_truth.tsv"
SAMPLE_FILE = "dataset/train/train_source1_sample_50k.tsv"

S2_FILE = "dataset/train/train_source2.tsv"
S3_FILE = "dataset/train/train_source3.tsv"

S2_OUTPUT = "dataset/train/train_source2_matches_50k.tsv"
S3_OUTPUT = "dataset/train/train_source3_matches_50k.tsv"


# --------------------------------------------------
# 1. Get the 50K S1 IDs
# --------------------------------------------------

print("Reading 50K S1 sample...")

sample_s1 = pd.read_csv(
    SAMPLE_FILE,
    sep="\t",
    usecols=["entity_id"]
)

sample_ids = set(sample_s1["entity_id"])

print("S1 sample:", len(sample_ids))


# --------------------------------------------------
# 2. Read ground truth
# --------------------------------------------------

print("\nReading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t"
)

gt = gt[
    gt["source1_entity_id"].isin(sample_ids)
].copy()

gt["matched_entity_ids"] = gt[
    "matched_entity_ids"
].fillna("")


# --------------------------------------------------
# 3. Extract all S2/S3 IDs
# --------------------------------------------------

s2_ids = set()
s3_ids = set()

for value in gt["matched_entity_ids"]:

    if not value.strip():
        continue

    for entity_id in value.split(","):

        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            s3_ids.add(entity_id)


print("\nRequired matching records:")
print("S2:", len(s2_ids))
print("S3:", len(s3_ids))


# --------------------------------------------------
# 4. Extract S2 records in chunks
# --------------------------------------------------

def extract_records(
    input_file,
    output_file,
    wanted_ids,
    source_name
):

    print(f"\nScanning {source_name}...")

    found = []

    for chunk in pd.read_csv(
        input_file,
        sep="\t",
        chunksize=100_000
    ):

        selected = chunk[
            chunk["entity_id"].isin(wanted_ids)
        ]

        if len(selected) > 0:
            found.append(selected)

        current = sum(len(x) for x in found)

        print(
            f"{source_name}: {current} / {len(wanted_ids)}",
            end="\r"
        )

    if found:
        result = pd.concat(
            found,
            ignore_index=True
        )
    else:
        result = pd.DataFrame()

    result.to_csv(
        output_file,
        sep="\t",
        index=False
    )

    print(
        f"\n{source_name} extracted: {len(result)}"
    )

    print(
        f"Saved: {output_file}"
    )


# --------------------------------------------------
# 5. Extract S2
# --------------------------------------------------

extract_records(
    S2_FILE,
    S2_OUTPUT,
    s2_ids,
    "S2"
)


# --------------------------------------------------
# 6. Extract S3
# --------------------------------------------------

extract_records(
    S3_FILE,
    S3_OUTPUT,
    s3_ids,
    "S3"
)


print("\n================================")
print("DONE")
print("================================")