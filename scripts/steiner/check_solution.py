#!/usr/bin/env python3
"""Solve rooted_mcf_v1 under P0 and verify the selected-edge solution independently."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

from steiner_branching.data.load import load_graph
from steiner_branching.milp.mcf import build_mcf
from steiner_branching.milp.validate import solve_and_validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--known-objective", type=float)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    graph = load_graph(args.instance)
    build = build_mcf(graph)
    result = solve_and_validate(build)
    checker_feasible = bool(result.check and result.check.feasible)
    known_objective_matches = (
        args.known_objective is None
        or result.objective is not None
        and math.isclose(result.objective, args.known_objective, rel_tol=1.0e-7, abs_tol=1.0e-7)
    )
    payload = {
        "checker_errors": list(result.check.errors) if result.check else [],
        "checker_feasible": checker_feasible,
        "graph_sha256": graph.graph_sha256,
        "known_objective": args.known_objective,
        "known_objective_matches": known_objective_matches,
        "nodes": int(build.model.getNNodes()),
        "objective": result.objective,
        "selected_edge_ids": list(result.selected_edge_ids),
        "solving_time_seconds": float(build.model.getSolvingTime()),
        "status": result.status,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result.status == "optimal" and checker_feasible and known_objective_matches else 2


if __name__ == "__main__":
    raise SystemExit(main())
