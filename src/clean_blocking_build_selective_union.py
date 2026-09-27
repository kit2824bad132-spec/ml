import gzip
import json

import pandas as pd


CURRENT_FILE = "dataset/train/clean_validation_candidates.tsv"
PASS_FILE = "dataset/train/clean_blocking_pass_candidates.tsv.gz"
OUTPUT_FILE = "dataset/train/clean_blocking_candidates_selective.tsv"
REPORT_FILE = "output/clean_blocking_selective_generation_report.json"
SELECTED_PASSES = {
    "house_number_address_token_country",
    "name_token_address_token_country",
}


def main():
    # This step consumes only candidate pairs frozen without ground truth.
    current = pd.read_csv(
        CURRENT_FILE,
        sep="\t",
        dtype={"source1_entity_id": str, "candidate_entity_id": str},
    )
    current_pairs = set(zip(current.source1_entity_id, current.candidate_entity_id))
    selected_pairs = set()
    with gzip.open(PASS_FILE, "rt", encoding="utf-8", newline="") as compressed:
        for chunk in pd.read_csv(compressed, sep="\t", dtype=str, chunksize=100000):
            selected = chunk[chunk.blocking_pass.isin(SELECTED_PASSES)]
            selected_pairs.update(zip(selected.source1_entity_id, selected.candidate_entity_id))

    union = current_pairs | selected_pairs
    output = pd.DataFrame(
        sorted(union),
        columns=["source1_entity_id", "candidate_entity_id"],
    )
    output.to_csv(OUTPUT_FILE, sep="\t", index=False)
    counts = output.groupby("source1_entity_id").size()
    report = {
        "ground_truth_loaded": False,
        "selected_passes": sorted(SELECTED_PASSES),
        "current_candidate_pairs": len(current_pairs),
        "selected_pass_candidate_pairs": len(selected_pairs),
        "new_union_candidate_pairs": len(union),
        "additional_candidate_pairs_vs_current": len(union - current_pairs),
        "candidate_growth_ratio": len(union) / len(current_pairs) if current_pairs else None,
        "average_candidates_per_source1": len(union) / 10000,
        "maximum_candidates_per_source1": int(counts.max()),
        "candidate_file": OUTPUT_FILE,
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as target:
        json.dump(report, target, indent=2)
    print("Selective union frozen without ground truth")
    print("Candidate pairs:", len(union))
    print("Additional pairs:", len(union - current_pairs))
    print("Growth:", f"{len(union) / len(current_pairs):.4f}x")
    print("Maximum candidates per Source-1:", int(counts.max()))
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()