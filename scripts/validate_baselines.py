#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


REQUIRED_FIELDS = {
    "instance_id",
    "method",
    "seed",
    "scip_version",
    "status",
    "objective",
    "primal_bound",
    "dual_bound",
    "final_gap",
    "wall_clock_time",
    "presolve_time",
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
    "solution_feasible",
}


def close(left, right, tolerance=1e-8):
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def main():
    parser = argparse.ArgumentParser(description="Validate baseline JSON results")
    parser.add_argument("--raw-dir", default="results/baseline/raw")
    parser.add_argument("--expected-runs", type=int, default=37)
    args = parser.parse_args()

    files = sorted(Path(args.raw_dir).glob("instance_*_seed*.json"))
    errors = []
    if len(files) != args.expected_runs:
        errors.append(f"expected {args.expected_runs} JSON files, found {len(files)}")

    rows = []
    for path in files:
        with path.open(encoding="utf-8") as stream:
            row = json.load(stream)
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            errors.append(f"{path}: missing fields {sorted(missing)}")
        if row.get("status") == "optimal" and not row.get("solution_feasible"):
            errors.append(f"{path}: optimal solution failed feasibility check")
        if row.get("nodes", 0) > 1 and row.get("branchrule_calls", 0) == 0:
            errors.append(f"{path}: processed multiple nodes but selected branchrule has no calls")
        rows.append(row)

    optimal_by_instance = defaultdict(list)
    by_key = {}
    for row in rows:
        by_key[(row["instance_id"], row["method"], row["seed"])] = row
        if row.get("status") == "optimal" and isinstance(row.get("objective"), (int, float)):
            optimal_by_instance[row["instance_id"]].append(row)

    for instance, solved in optimal_by_instance.items():
        reference = solved[0]["objective"]
        for row in solved[1:]:
            if not close(reference, row["objective"]):
                errors.append(
                    f"instance {instance}: objective mismatch {reference} vs {row['objective']} ({row['method']})"
                )

    for instance in sorted({row["instance_id"] for row in rows}):
        default = by_key.get((instance, "default", 0))
        relpscost = by_key.get((instance, "relpscost", 0))
        if default is None or relpscost is None:
            continue
        fields = ["status", "nodes", "objective", "dual_bound", "active_branching_rule"]
        if default.get("status") == "optimal" and relpscost.get("status") == "optimal":
            fields.append("lp_iterations")
        for field in fields:
            left = default.get(field)
            right = relpscost.get(field)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                equal = close(float(left), float(right))
            else:
                equal = left == right
            if not equal:
                errors.append(f"instance {instance}: default/relpscost differ in {field}: {left} vs {right}")

    if errors:
        print("Baseline validation FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"Baseline validation passed: {len(rows)} runs")
    print(f"Instances with at least one optimal run: {','.join(sorted(optimal_by_instance))}")


if __name__ == "__main__":
    main()
