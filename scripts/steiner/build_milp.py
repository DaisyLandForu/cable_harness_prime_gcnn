#!/usr/bin/env python3
"""Build rooted_mcf_v1 and emit an LP plus canonical problem metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

from steiner_branching.contracts import canonical_json
from steiner_branching.data.load import load_graph
from steiner_branching.milp.mcf import build_mcf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--lp-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    graph = load_graph(args.instance)
    build = build_mcf(graph)
    args.lp_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    build.model.writeProblem(str(args.lp_output))
    args.metadata_output.write_text(
        canonical_json(build.metadata.to_dict()) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "binary_edge_variables": build.counts.binary_edge_variables,
                "continuous_flow_variables": build.counts.continuous_flow_variables,
                "graph_sha256": graph.graph_sha256,
                "lp_output": str(args.lp_output),
                "metadata_output": str(args.metadata_output),
                "metadata_sha256": build.metadata.sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
