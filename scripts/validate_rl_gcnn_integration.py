#!/usr/bin/env python3
"""Validate phase-7 C++ RL-GCNN integration artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


TRACE_FIELDS = (
    "node_id",
    "depth",
    "candidate_count",
    "selected_candidate_index",
    "selected_variable_index",
    "selected_variable_name",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--default-json", type=Path, required=True)
    parser.add_argument("--rl-json", type=Path, required=True)
    parser.add_argument("--rl-log", type=Path, required=True)
    parser.add_argument("--repeat-json", type=Path, required=True)
    parser.add_argument("--repeat-log", type=Path, required=True)
    parser.add_argument("--cpu-json", type=Path, required=True)
    parser.add_argument("--model-fallback-json", type=Path, required=True)
    parser.add_argument("--gate-fallback-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective-tolerance", type=float, default=1e-8)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_log(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def decision_trace(rows: list[dict[str, str]]) -> list[tuple[str, ...]]:
    return [tuple(row[field] for field in TRACE_FIELDS) for row in rows]


def solved_and_feasible(result: dict) -> bool:
    return (
        result["status"] == "optimal"
        and result["has_solution"]
        and result["solution_feasible"]
    )


def objective_close(left: float, right: float, tolerance: float) -> bool:
    scale = max(1.0, abs(right))
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance * scale)


def inference_summary(result: dict) -> dict:
    solve_time = float(result["solving_time"])
    total = float(result["rl_inference_total"])
    return {
        "nodes": result["nodes"],
        "branch_decisions": result["branch_decisions"],
        "inference_total_seconds": total,
        "inference_mean_seconds": result["rl_inference_mean"],
        "inference_max_seconds": result["rl_inference_max"],
        "inference_fraction_of_solving_time": total / solve_time if solve_time > 0 else None,
    }


def main() -> None:
    args = parse_args()
    default = load_json(args.default_json)
    rl = load_json(args.rl_json)
    repeat = load_json(args.repeat_json)
    cpu = load_json(args.cpu_json)
    model_fallback = load_json(args.model_fallback_json)
    gate_fallback = load_json(args.gate_fallback_json)
    rows = load_log(args.rl_log)
    repeat_rows = load_log(args.repeat_log)

    checks = {
        "all_runs_optimal_and_feasible": all(
            solved_and_feasible(result)
            for result in (default, rl, repeat, cpu, model_fallback, gate_fallback)
        ),
        "all_objectives_match_default": all(
            objective_close(result["objective"], default["objective"], args.objective_tolerance)
            for result in (rl, repeat, cpu, model_fallback, gate_fallback)
        ),
        "all_rl_actions_legal": (
            rl["custom_illegal_actions"] == 0
            and rl["custom_legality_checks"] == rl["branch_decisions"]
            and len(rows) == rl["branch_decisions"]
            and all(row["selected_is_candidate"] == "true" for row in rows)
        ),
        "repeat_decision_sequence_equal": (
            decision_trace(rows) == decision_trace(repeat_rows)
        ),
        "missing_model_fallback_used": (
            model_fallback["branch_decisions"] == 0
            and model_fallback["fallback_count"] > 0
        ),
        "depth_gate_fallback_used": (
            gate_fallback["branch_decisions"] > 0
            and gate_fallback["fallback_count"] > 0
        ),
    }
    summary = {
        "passed": all(checks.values()),
        "checks": checks,
        "default": {
            "objective": default["objective"],
            "nodes": default["nodes"],
            "solving_time": default["solving_time"],
        },
        "rl_cuda": inference_summary(rl),
        "rl_cuda_repeat": inference_summary(repeat),
        "rl_cpu": inference_summary(cpu),
        "repeat_trace_length": len(repeat_rows),
        "model_fallback_count": model_fallback["fallback_count"],
        "gate_rl_decisions": gate_fallback["branch_decisions"],
        "gate_fallback_count": gate_fallback["fallback_count"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
