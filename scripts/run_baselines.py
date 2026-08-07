#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
import subprocess
from pathlib import Path


RUN_FIELDS = [
    "instance_id",
    "method",
    "seed",
    "scip_version",
    "status",
    "objective",
    "business_objective",
    "primal_bound",
    "dual_bound",
    "final_gap",
    "wall_clock_time",
    "presolve_time",
    "solving_time",
    "solve_time_after_presolve",
    "nodes",
    "lp_iterations",
    "primal_dual_integral",
    "first_solution_time",
    "number_of_variables",
    "number_of_integer_variables",
    "number_of_constraints",
    "active_branching_rule",
    "branchrule_calls",
    "node_selection_rule",
    "threads",
    "time_limit",
    "node_limit",
    "has_solution",
    "solution_feasible",
    "return_code",
    "json_path",
    "log_path",
]


def parse_list(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def finite_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    summary = []
    methods = sorted({row["method"] for row in rows if row.get("method")})
    for method in methods:
        selected = [row for row in rows if row.get("method") == method]
        solved = [row for row in selected if row.get("status") == "optimal"]
        times = finite_values(selected, "wall_clock_time")
        nodes = finite_values(selected, "nodes")
        gaps = finite_values(selected, "final_gap")
        summary.append(
            {
                "method": method,
                "runs": len(selected),
                "optimal_runs": len(solved),
                "timeout_runs": sum(row.get("status") == "time_limit" for row in selected),
                "solved_rate": len(solved) / len(selected) if selected else 0.0,
                "mean_wall_time": statistics.fmean(times) if times else None,
                "median_wall_time": statistics.median(times) if times else None,
                "mean_nodes": statistics.fmean(nodes) if nodes else None,
                "median_nodes": statistics.median(nodes) if nodes else None,
                "mean_final_gap": statistics.fmean(gaps) if gaps else None,
                "finite_gap_runs": len(gaps),
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run reproducible SCIP branching baselines")
    parser.add_argument("--binary", default="build/scip_tree")
    parser.add_argument("--instances", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--methods", default="default,relpscost,random,mostinf")
    parser.add_argument("--strong-instances", default="9")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--node-limit", type=int, default=-1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", default="results/baseline")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    binary = Path(args.binary).resolve()
    if not binary.is_file():
        parser.error(f"binary not found: {binary}")

    instances = parse_list(args.instances)
    methods = parse_list(args.methods)
    seeds = parse_list(args.seeds, int)
    strong_instances = set(parse_list(args.strong_instances))
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    planned = [(instance, method, seed) for instance in instances for method in methods for seed in seeds]
    planned.extend(
        (instance, "strong", seed)
        for instance in instances
        if instance in strong_instances
        for seed in seeds
    )

    rows = []
    for index, (instance, method, seed) in enumerate(planned, start=1):
        stem = f"instance_{instance}_{method}_seed{seed}"
        json_path = raw_dir / f"{stem}.json"
        log_path = raw_dir / f"{stem}.log"
        command = [
            str(binary),
            "--instance-id", instance,
            "--branching", method,
            "--seed", str(seed),
            "--time-limit", str(args.time_limit),
            "--node-limit", str(args.node_limit),
            "--threads", str(args.threads),
            "--output-json", str(json_path),
        ]
        print(f"[{index}/{len(planned)}] instance={instance} method={method} seed={seed}", flush=True)
        return_code = 0
        if not (args.resume and json_path.is_file()):
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            return_code = completed.returncode

        if json_path.is_file():
            with json_path.open(encoding="utf-8") as stream:
                row = json.load(stream)
        else:
            row = {"instance_id": instance, "method": method, "seed": seed, "status": "process_error"}
        row["return_code"] = return_code
        row["json_path"] = str(json_path)
        row["log_path"] = str(log_path)
        rows.append(row)

    write_csv(output_dir / "runs.csv", RUN_FIELDS, rows)
    summary_fields = [
        "method", "runs", "optimal_runs", "timeout_runs", "solved_rate", "mean_wall_time",
        "median_wall_time", "mean_nodes", "median_nodes", "mean_final_gap", "finite_gap_runs",
    ]
    write_csv(output_dir / "summary.csv", summary_fields, aggregate(rows))

    failures = [row for row in rows if row.get("return_code") != 0]
    if failures:
        print(f"Completed with {len(failures)} process failures", flush=True)
        raise SystemExit(1)
    print(f"Completed {len(rows)} runs; summary: {output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
