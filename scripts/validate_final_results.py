#!/usr/bin/env python3
"""Validate phase-8 raw experiment results and RL branch logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


REQUIRED_FIELDS = {
    "instance_id",
    "split",
    "size",
    "protocol",
    "method",
    "seed",
    "status",
    "objective",
    "primal_bound",
    "dual_bound",
    "gap",
    "wall_time",
    "solve_time",
    "nodes",
    "branch_decisions",
    "rl_inference_total",
    "fallback_count",
    "illegal_actions",
    "legality_checks",
    "active_branching_rule",
    "node_selection_rule",
    "threads",
    "solution_feasible",
    "return_code",
}

EXPECTED_BRANCHRULE = {
    "default": "relpscost",
    "relpscost": "relpscost",
    "random": "random",
    "mostinf": "mostinf",
    "strong": "fullstrong",
    "rl-mlp": "rlmlp",
    "rl-gcnn": "rlgcnn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective-tolerance", type=float, default=1e-8)
    return parser.parse_args()


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def as_int(value: str, default: int = 0) -> int:
    return int(float(value)) if value.strip() else default


def as_float(value: str) -> float | None:
    if not value.strip():
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def close(left: float, right: float, tolerance: float) -> bool:
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def validate_branch_log(row: dict[str, str], errors: list[str]) -> None:
    if not row["method"].startswith("rl-"):
        return
    path_text = row.get("branch_log_path", "")
    path = Path(path_text) if path_text else None
    if path is None or not path.is_file():
        errors.append(f"missing RL branch log for {row['protocol']} {row['instance_id']} seed={row['seed']}")
        return
    with path.open(newline="", encoding="utf-8") as stream:
        decisions = list(csv.DictReader(stream))
    legal_decisions = [
        item
        for item in decisions
        if item.get("result") in {"branched", "reduced_domain"}
    ]
    if len(legal_decisions) != as_int(row["branch_decisions"]):
        errors.append(f"branch log count mismatch: {path}")
    if any(item.get("selected_is_candidate") != "true" for item in legal_decisions):
        errors.append(f"illegal selected candidate in {path}")


def main() -> None:
    args = parse_args()
    with args.input.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        missing_columns = REQUIRED_FIELDS - set(reader.fieldnames or ())

    errors: list[str] = []
    if missing_columns:
        errors.append(f"missing columns: {sorted(missing_columns)}")
    if args.expected_runs is not None and len(rows) != args.expected_runs:
        errors.append(f"expected {args.expected_runs} runs, found {len(rows)}")
    keys = [(row["protocol"], row["instance_id"], row["method"], row["seed"]) for row in rows]
    if len(keys) != len(set(keys)):
        errors.append("duplicate protocol/instance/method/seed rows")

    reference_objectives: dict[str, float] = {}
    status_counts: dict[str, int] = {}
    rl_decisions = 0
    for row in rows:
        label = f"{row['protocol']} {row['instance_id']} {row['method']} seed={row['seed']}"
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        if as_int(row["return_code"], 1) != 0:
            errors.append(f"nonzero return code: {label}")
        if as_int(row["threads"], -1) != 1:
            errors.append(f"non-single-thread run: {label}")
        expected_selector = "dfs" if row["protocol"] == "controlled-bbmdp" else "estimate"
        if row["node_selection_rule"] != expected_selector:
            errors.append(f"wrong node selector: {label}: {row['node_selection_rule']}")
        expected_rule = EXPECTED_BRANCHRULE.get(row["method"])
        if expected_rule is not None and row["active_branching_rule"] != expected_rule:
            errors.append(f"wrong branching rule: {label}: {row['active_branching_rule']}")
        if row["status"] == "optimal":
            if not as_bool(row["solution_feasible"]):
                errors.append(f"optimal solution is infeasible: {label}")
            objective = as_float(row["objective"])
            if objective is None:
                errors.append(f"optimal run has no objective: {label}")
            elif row["instance_id"] not in reference_objectives:
                reference_objectives[row["instance_id"]] = objective
            elif not close(objective, reference_objectives[row["instance_id"]], args.objective_tolerance):
                errors.append(
                    f"objective mismatch: {label}: {objective} vs "
                    f"{reference_objectives[row['instance_id']]}"
                )
        if row["method"].startswith("rl-"):
            if as_int(row["illegal_actions"]) != 0:
                errors.append(f"RL illegal action count is nonzero: {label}")
            decisions = as_int(row["branch_decisions"])
            checks = as_int(row["legality_checks"])
            if checks < decisions:
                errors.append(f"fewer legality checks than decisions: {label}")
            rl_decisions += decisions
            validate_branch_log(row, errors)

    result = {
        "passed": not errors,
        "runs": len(rows),
        "status_counts": status_counts,
        "instances_with_optimal_reference": sorted(reference_objectives),
        "rl_branch_decisions": rl_decisions,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
