#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from instance_generator import generate_instance, sha256_file


def main():
    parser = argparse.ArgumentParser(description="Validate phase-2 dataset splits and reproducibility")
    parser.add_argument("--config", default="configs/dataset/phase2.json")
    parser.add_argument("--manifest", default="data/instances/manifest.csv")
    parser.add_argument("--scip-binary")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    with Path(args.manifest).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    errors = []
    expected_count = 0
    for split_config in config["splits"].values():
        expected_count += len(split_config.get("real", []))
        expected_count += sum(len(split_seeds) for split_seeds in split_config.get("synthetic", {}).values())
    if len(rows) != expected_count:
        errors.append(f"expected {expected_count} manifest rows, found {len(rows)}")
    ids = [row["instance_id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("instance IDs are not unique")

    scenarios = defaultdict(set)
    seeds = defaultdict(set)
    sizes = defaultdict(set)
    for row in rows:
        scenarios[row["source_scenario"]].add(row["split"])
        sizes[row["split"]].add(row["size"])
        if row["source_type"] == "synthetic":
            key = (row["size"], row["seed"])
            if key in seeds[row["split"]]:
                errors.append(f"duplicate synthetic seed within split: {row['split']} {key}")
            seeds[row["split"]].add(key)
        cip_path = Path(row["cip_path"])
        metadata_path = Path(row["metadata_path"])
        if not cip_path.is_file() or cip_path.stat().st_size == 0:
            errors.append(f"missing or empty CIP: {cip_path}")
            continue
        if not metadata_path.is_file():
            errors.append(f"missing metadata: {metadata_path}")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["instance_id"] != row["instance_id"] or metadata["split"] != row["split"]:
            errors.append(f"manifest/metadata identity mismatch: {metadata_path}")
        if sha256_file(cip_path) != metadata["milp"]["cip_sha256"]:
            errors.append(f"CIP hash mismatch: {cip_path}")
        for source_name in ("edges", "pairs"):
            source_path = Path(metadata["source_files"][source_name])
            expected_hash = metadata["source_files"][f"{source_name}_sha256"]
            if not source_path.is_file() or sha256_file(source_path) != expected_hash:
                errors.append(f"source hash mismatch: {source_path}")
        for field in ("variables", "integer_variables", "constraints", "commodities"):
            if int(metadata["milp"][field]) <= 0:
                errors.append(f"non-positive MILP field {field}: {row['instance_id']}")

    for scenario, split_set in scenarios.items():
        if len(split_set) > 1:
            errors.append(f"scenario leakage: {scenario} appears in {sorted(split_set)}")
    for split in ("train", "validation", "test"):
        missing_sizes = {"small", "medium", "large"} - sizes[split]
        if missing_sizes:
            errors.append(f"{split} lacks sizes: {sorted(missing_sizes)}")
    real_rows = [row for row in rows if row["source_type"] == "real"]
    largest_real = max(real_rows, key=lambda row: int(row["variables"]))
    if largest_real["split"] != "transfer":
        errors.append(f"largest real instance {largest_real['instance_id']} is not in transfer")

    split_seed_pairs = {
        split: {(row["size"], row["seed"]) for row in rows if row["split"] == split and row["source_type"] == "synthetic"}
        for split in ("train", "validation", "test")
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_seed_pairs[left] & split_seed_pairs[right]
        if overlap:
            errors.append(f"synthetic seed leakage between {left} and {right}: {sorted(overlap)}")

    if args.scip_binary:
        for row in rows:
            command = [args.scip_binary, "-q", "-c", f"read {row['cip_path']}", "-c", "quit"]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            output = completed.stdout + completed.stderr
            if completed.returncode != 0 or "error" in output.lower():
                errors.append(f"SCIP failed to read {row['cip_path']}")

    synthetic_rows = [row for row in rows if row["source_type"] == "synthetic"]
    with tempfile.TemporaryDirectory(prefix="harness-dataset-check-") as temp_dir:
        for row in synthetic_rows:
            metadata = json.loads(Path(row["metadata_path"]).read_text(encoding="utf-8"))
            if metadata["generator"]["parameters"] != config["synthetic_presets"][row["size"]]:
                errors.append(f"generator preset mismatch: {row['instance_id']}")
            regenerated = generate_instance(
                row["instance_id"],
                int(row["seed"]),
                metadata["generator"]["parameters"],
                temp_dir,
            )
            if regenerated["source_files"]["edges_sha256"] != metadata["source_files"]["edges_sha256"]:
                errors.append(f"edge reproduction failed: {row['instance_id']}")
            if regenerated["source_files"]["pairs_sha256"] != metadata["source_files"]["pairs_sha256"]:
                errors.append(f"pair reproduction failed: {row['instance_id']}")

    if errors:
        print("Dataset validation FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    counts = defaultdict(int)
    for row in rows:
        counts[row["split"]] += 1
    print(f"Dataset validation passed: {len(rows)} instances")
    print("Split counts: " + ", ".join(f"{key}={counts[key]}" for key in ("train", "validation", "test", "transfer")))


if __name__ == "__main__":
    main()
