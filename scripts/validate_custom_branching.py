#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Validate custom branching run artifacts")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    return parser.parse_args()


def branch_log_path(run_path):
    return run_path.with_name(run_path.stem + "_branches.csv")


def main():
    args = parse_args()
    reference = json.loads(args.reference.read_text())
    reference_objective = reference["objective"]
    errors = []

    for run_path in args.runs:
        run = json.loads(run_path.read_text())
        method = run["method"]
        expected_rule = {
            "custom-random": "rlcustomrandom",
            "custom-mostinf": "rlcustommostinf",
        }.get(method)
        if expected_rule is None:
            errors.append(f"{run_path}: unsupported method {method}")
            continue
        if run["status"] != "optimal" or not run["solution_feasible"]:
            errors.append(f"{run_path}: solve is not optimal and feasible")
        scale = max(1.0, abs(reference_objective))
        if not math.isclose(run["objective"], reference_objective, rel_tol=0.0,
                            abs_tol=args.tolerance * scale):
            errors.append(f"{run_path}: objective differs from reference")
        if run["active_branching_rule"] != expected_rule:
            errors.append(f"{run_path}: wrong active branching rule")
        if run["custom_illegal_actions"] != 0:
            errors.append(f"{run_path}: illegal custom actions recorded")
        if run["custom_branch_decisions"] <= 0:
            errors.append(f"{run_path}: custom branch rule made no decisions")
        if run["branchrule_calls"] != run["custom_branch_lp_calls"]:
            errors.append(f"{run_path}: SCIP and plugin call counters disagree")

        log_path = branch_log_path(run_path)
        with log_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != run["custom_branch_lp_calls"]:
            errors.append(f"{log_path}: row count does not match LP calls")
        for row_number, row in enumerate(rows, start=2):
            selected = int(row["selected_candidate_index"])
            candidate_count = int(row["candidate_count"])
            if row["result"] == "branched":
                if row["selected_is_candidate"] != "true":
                    errors.append(f"{log_path}:{row_number}: selected action is illegal")
                if not 0 <= selected < candidate_count:
                    errors.append(f"{log_path}:{row_number}: candidate index is outside mask")

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"validated {len(args.runs)} custom branching runs")


if __name__ == "__main__":
    main()
