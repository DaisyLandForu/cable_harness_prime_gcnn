#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

from instance_generator import generate_instance, sha256_file


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_logged(command, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with code {completed.returncode}; see {log_path}")


def is_center(name):
    return name.startswith("E") or re.fullmatch(r"[NM][0-9]+", name) is not None


def inspect_real_source(edges_path, pairs_path):
    nodes = set()
    center_nodes = set()
    center_edges = 0
    scenarios = set()
    edge_count = 0
    first_row = True
    with Path(edges_path).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.reader(stream):
            if first_row:
                first_row = False
                continue
            if len(row) < 5:
                continue
            source, target = row[2].strip(), row[3].strip()
            scenarios.add(row[1].strip())
            nodes.update((source, target))
            if is_center(source):
                center_nodes.add(source)
            if is_center(target):
                center_nodes.add(target)
            if is_center(source) and is_center(target):
                center_edges += 1
            edge_count += 1
    cable_count = -1
    with Path(pairs_path).open(newline="", encoding="utf-8-sig") as stream:
        cable_count = sum(1 for _ in csv.reader(stream)) - 1
    return {
        "nodes": len(nodes),
        "edges": edge_count,
        "center_nodes": len(center_nodes),
        "center_edges": center_edges,
        "cables": cable_count,
        "scenario": "+".join(sorted(item for item in scenarios if item)),
    }


def size_from_variables(n_vars):
    if n_vars <= 15000:
        return "small"
    if n_vars <= 60000:
        return "medium"
    return "large"


def manifest_row(metadata):
    graph = metadata["business_parameters"]
    milp = metadata["milp"]
    baseline = metadata["baseline"]
    generation_seed = metadata["generation_seed"]
    return {
        "split": metadata["split"],
        "instance_id": metadata["instance_id"],
        "source_type": metadata["source_type"],
        "source_scenario": metadata["source_scenario"],
        "seed": generation_seed if generation_seed is not None else metadata["baseline_seed"],
        "size": metadata["size"],
        "graph_nodes": graph["nodes"],
        "graph_edges": graph["edges"],
        "center_nodes": milp["center_nodes"],
        "center_edges": milp["center_edges"],
        "cables": graph["cables"],
        "commodities": milp["commodities"],
        "copy_num": metadata["copy_num"],
        "variables": milp["variables"],
        "integer_variables": milp["integer_variables"],
        "constraints": milp["constraints"],
        "baseline_status": baseline["status"],
        "baseline_time": baseline["wall_time"],
        "baseline_nodes": baseline["nodes"],
        "baseline_gap": baseline["final_gap"],
        "cip_path": milp["cip"],
        "metadata_path": str(Path(milp["cip"]).with_suffix(".json")),
    }


def main():
    parser = argparse.ArgumentParser(description="Build phase-2 train/validation/test/transfer dataset")
    parser.add_argument("--config", default="configs/dataset/phase2.json")
    parser.add_argument("--binary", default="build/scip_tree")
    parser.add_argument("--instances-dir", default="data/instances")
    parser.add_argument("--generated-dir", default="data/generated")
    parser.add_argument("--results-dir", default="results/dataset")
    parser.add_argument("--manifest", default="data/instances/manifest.csv")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = read_json(args.config)
    binary = str(Path(args.binary).resolve())
    if not Path(binary).is_file():
        parser.error(f"binary not found: {binary}")
    instances_dir = Path(args.instances_dir)
    generated_dir = Path(args.generated_dir)
    results_dir = Path(args.results_dir)
    rows = []

    planned = []
    for split, split_config in config["splits"].items():
        for real_id in split_config.get("real", []):
            planned.append(
                {
                    "split": split,
                    "instance_id": f"real_{real_id:02d}",
                    "source_type": "real",
                    "real_id": real_id,
                    "generation_seed": None,
                }
            )
        for size, seeds in split_config.get("synthetic", {}).items():
            for seed in seeds:
                planned.append(
                    {
                        "split": split,
                        "instance_id": f"syn_{size}_s{seed}",
                        "source_type": "synthetic",
                        "size": size,
                        "generation_seed": seed,
                    }
                )

    seen_ids = set()
    for index, item in enumerate(planned, start=1):
        instance_id = item["instance_id"]
        if instance_id in seen_ids:
            raise RuntimeError(f"duplicate instance id: {instance_id}")
        seen_ids.add(instance_id)
        split = item["split"]
        print(f"[{index}/{len(planned)}] {split}/{instance_id}", flush=True)

        split_dir = instances_dir / split
        cip_path = split_dir / f"{instance_id}.cip"
        metadata_path = split_dir / f"{instance_id}.json"
        if args.resume and cip_path.is_file() and metadata_path.is_file():
            metadata = read_json(metadata_path)
            rows.append(manifest_row(metadata))
            print("  resume: complete instance", flush=True)
            continue

        if item["source_type"] == "synthetic":
            source_dir = generated_dir / split
            generator_metadata = generate_instance(
                instance_id,
                item["generation_seed"],
                config["synthetic_presets"][item["size"]],
                source_dir,
            )
            edges_path = Path(generator_metadata["source_files"]["edges"])
            pairs_path = Path(generator_metadata["source_files"]["pairs"])
            source_scenario = generator_metadata["scenario"]
            graph = generator_metadata["graph"]
        else:
            real_id = item["real_id"]
            edges_path = Path(f"code/data/edges-{real_id}.csv")
            pairs_path = Path(f"code/data/pairs-{real_id}.csv")
            graph = inspect_real_source(edges_path, pairs_path)
            source_scenario = graph.pop("scenario")
            generator_metadata = None

        build_json = results_dir / "raw" / "build" / f"{instance_id}.json"
        build_log = results_dir / "raw" / "build" / f"{instance_id}.log"
        build_command = [
            binary,
            "--instance-id", instance_id,
            "--edges", str(edges_path),
            "--pairs", str(pairs_path),
            "--copy-num", str(config["copy_num"]),
            "--branching", "default",
            "--seed", str(config["baseline_seed"]),
            "--threads", str(config["threads"]),
            "--time-limit", str(config["baseline_time_limit"]),
            "--node-limit", str(config["baseline_node_limit"]),
            "--export-milp", str(cip_path),
            "--build-only",
            "--output-json", str(build_json),
        ]
        if not (args.resume and cip_path.is_file() and build_json.is_file()):
            run_logged(build_command, build_log)
        else:
            print("  resume: existing CIP/build metrics", flush=True)
        build = read_json(build_json)

        if item["source_type"] == "real":
            baseline_json = Path(f"results/baseline/raw/instance_{item['real_id']}_default_seed0.json")
            baseline_log = Path(f"results/baseline/raw/instance_{item['real_id']}_default_seed0.log")
            if not baseline_json.is_file():
                raise RuntimeError(f"missing phase-1 baseline: {baseline_json}")
        else:
            baseline_json = results_dir / "raw" / "baseline" / f"{instance_id}.json"
            baseline_log = results_dir / "raw" / "baseline" / f"{instance_id}.log"
            baseline_command = [
                binary,
                "--instance-id", instance_id,
                "--edges", str(edges_path),
                "--pairs", str(pairs_path),
                "--copy-num", str(config["copy_num"]),
                "--branching", "default",
                "--seed", str(config["baseline_seed"]),
                "--threads", str(config["threads"]),
                "--time-limit", str(config["baseline_time_limit"]),
                "--node-limit", str(config["baseline_node_limit"]),
                "--output-json", str(baseline_json),
            ]
            if not (args.resume and baseline_json.is_file()):
                run_logged(baseline_command, baseline_log)
            else:
                print("  resume: existing baseline", flush=True)
        baseline = read_json(baseline_json)

        n_vars = build["number_of_variables"]
        size = item.get("size", size_from_variables(n_vars))
        metadata = {
            "instance_id": instance_id,
            "split": split,
            "size": size,
            "source_type": item["source_type"],
            "source_scenario": source_scenario,
            "generation_seed": item["generation_seed"],
            "baseline_seed": config["baseline_seed"],
            "copy_num": config["copy_num"],
            "business_parameters": graph,
            "generator": generator_metadata,
            "source_files": {
                "edges": str(edges_path),
                "pairs": str(pairs_path),
                "edges_sha256": sha256_file(edges_path),
                "pairs_sha256": sha256_file(pairs_path),
            },
            "milp": {
                "cip": str(cip_path),
                "cip_sha256": sha256_file(cip_path),
                "variables": n_vars,
                "integer_variables": build["number_of_integer_variables"],
                "constraints": build["number_of_constraints"],
                "center_nodes": build["number_of_center_nodes"],
                "center_edges": build["number_of_center_edges"],
                "commodities": build["number_of_commodities"],
                "constraint_variable_ratio": build["number_of_constraints"] / n_vars,
            },
            "baseline": {
                "method": "default",
                "status": baseline["status"],
                "objective": baseline["objective"],
                "dual_bound": baseline["dual_bound"],
                "final_gap": baseline["final_gap"],
                "wall_time": baseline["wall_clock_time"],
                "solving_time": baseline["solving_time"],
                "nodes": baseline["nodes"],
                "time_limit": config["baseline_time_limit"],
                "threads": config["threads"],
                "raw_json": str(baseline_json),
                "raw_log": str(baseline_log),
            },
        }
        write_json(metadata_path, metadata)
        rows.append(manifest_row(metadata))

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} instances to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
