import json

import pandas as pd


GT_FILE = "dataset/train/train_ground_truth.tsv"
SPLIT_FILE = "dataset/train/clean_validation_split.tsv"
CANDIDATES_FILE = "dataset/train/clean_validation_candidates.tsv"
LABELED_FILE = "dataset/train/clean_validation_labeled_candidates.tsv"
AUDIT_FILE = "output/clean_validation_label_audit.json"
CHUNK_SIZE = 200000


def parse_targets(value):
    return {
        target.strip()
        for target in value.split(",")
        if target.strip()
    }


def main():
    split = pd.read_csv(SPLIT_FILE, sep="\t", dtype=str)
    training_ids = set(split.loc[split.split == "train", "entity_id"])
    validation_ids = set(split.loc[split.split == "validation", "entity_id"])

    # The candidate file is already frozen; only now is ground truth opened.
    candidates = pd.read_csv(
        CANDIDATES_FILE,
        sep="\t",
        dtype={"source1_entity_id": str, "candidate_entity_id": str},
    )
    candidate_pairs = set(
        zip(candidates.source1_entity_id, candidates.candidate_entity_id)
    )
    validation_targets = {}
    training_targets = set()
    total_validation_targets = 0

    for chunk in pd.read_csv(
        GT_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):
        selected = chunk[chunk.source1_entity_id.isin(training_ids | validation_ids)]
        for row in selected.itertuples(index=False):
            targets = parse_targets(row.matched_entity_ids)
            if row.source1_entity_id in validation_ids:
                validation_targets[row.source1_entity_id] = targets
                total_validation_targets += len(targets)
            else:
                training_targets.update(targets)

    positive_pairs = {
        (source_id, target_id)
        for source_id, targets in validation_targets.items()
        for target_id in targets
    }
    candidate_positive_pairs = positive_pairs & candidate_pairs
    candidate_pair_set = candidate_pairs
    labeled = candidates.copy()
    labeled["label"] = [
        int(pair in positive_pairs)
        for pair in zip(labeled.source1_entity_id, labeled.candidate_entity_id)
    ]
    labeled.to_csv(LABELED_FILE, sep="\t", index=False)

    validation_targets_union = set().union(*validation_targets.values()) if validation_targets else set()
    target_overlap = training_targets & validation_targets_union
    candidate_recall = (
        len(candidate_positive_pairs) / total_validation_targets
        if total_validation_targets
        else 0.0
    )
    audit = {
        "candidate_file_frozen_before_ground_truth_read": True,
        "validation_labels_used_to_construct_candidates": False,
        "validation_source1_count": len(validation_ids),
        "training_source1_count": len(training_ids),
        "source1_id_overlap": len(training_ids & validation_ids),
        "validation_candidate_pairs": len(candidates),
        "unique_validation_candidate_pairs": len(candidate_pair_set),
        "duplicate_validation_candidate_pairs": len(candidates) - len(candidate_pair_set),
        "validation_source1_with_candidates": candidates.source1_entity_id.nunique(),
        "validation_truth_match_count": total_validation_targets,
        "validation_truth_pairs_present_in_candidates": len(candidate_positive_pairs),
        "candidate_recall": candidate_recall,
        "training_unique_matched_target_ids": len(training_targets),
        "validation_unique_matched_target_ids": len(validation_targets_union),
        "matched_target_id_overlap_count": len(target_overlap),
        "matched_target_overlap_fraction_of_validation_targets": (
            len(target_overlap) / len(validation_targets_union)
            if validation_targets_union
            else 0.0
        ),
        "labeled_validation_positive_candidate_pairs": int(labeled.label.sum()),
        "labeled_validation_negative_candidate_pairs": int((labeled.label == 0).sum()),
    }
    with open(AUDIT_FILE, "w", encoding="utf-8") as output:
        json.dump(audit, output, indent=2)

    print("Labeled frozen validation candidate pairs:", len(labeled))
    print("Positive candidate pairs:", int(labeled.label.sum()))
    print("Negative candidate pairs:", int((labeled.label == 0).sum()))
    print("Total validation GT matches:", total_validation_targets)
    print("True matches in frozen candidates:", len(candidate_positive_pairs))
    print("Candidate recall:", f"{candidate_recall:.6f}")
    print("Matched target ID overlap train/validation:", len(target_overlap))
    print("Saved labeled candidates:", LABELED_FILE)
    print("Saved audit:", AUDIT_FILE)


if __name__ == "__main__":
    main()