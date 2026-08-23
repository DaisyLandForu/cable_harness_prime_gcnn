#!/usr/bin/env python3
"""Measure actual union-two-hop transition bytes on real_02 warmup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_branching import BBMDPBranchingEnv, BBMDPConfig
from rl_branching.graph_features import (
    candidate_twohop_state,
    extract_graph_state,
    graph_state_storage_bytes,
    transition_storage_bytes,
)
from rl_branching.graph_replay import LARGE_BYTE_LIMIT, LARGE_SAMPLE_QUOTA, DualPoolGraphReplay
from rl_branching.replay import ReplayExperience
from rl_branching.scip_profile import sha256_file

INSTANCE = Path("data/instances/transfer/real_02.cip")
OUTPUT = Path("results/probes/real02_transition_bytes.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, default=INSTANCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--max-transitions", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.instance.is_file():
        raise FileNotFoundError(args.instance)

    env = BBMDPBranchingEnv(
        BBMDPConfig(seed=args.seed, time_limit=args.time_limit, node_limit=-1)
    )
    state = env.reset(args.instance)
    report = {
        "instance": str(args.instance),
        "instance_sha256": sha256_file(args.instance),
        "seed": args.seed,
        "time_limit": args.time_limit,
        "status": state.info.get("status"),
        "terminated": state.terminated,
        "truncated": state.truncated,
        "lp_iterations": state.info.get("lp_iterations"),
        "effective_search_params_sha256": None,
        "transitions": [],
        "large_pool_can_hold_4": False,
        "gate_passed": False,
    }
    if state.terminated or state.truncated or state.observation is None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        env.close()
        raise SystemExit("real_02 warmup did not reach a live branching state")

    report["effective_search_params_sha256"] = env.effective_search_params_sha256()
    replay = DualPoolGraphReplay(seed=args.seed)
    measured: list[int] = []
    current = candidate_twohop_state(extract_graph_state(state.observation, state.action_set))
    collected = 0
    while collected < args.max_transitions and not state.terminated and not state.truncated:
        action = int(state.action_set[0])
        transition = env.step(action)
        next_state = None
        if transition.next_observation is not None and transition.next_action_set.size:
            next_state = candidate_twohop_state(
                extract_graph_state(transition.next_observation, transition.next_action_set)
            )
        nbytes = transition_storage_bytes(current, next_state)
        replay.add(
            ReplayExperience(current, 0, float(transition.reward), next_state, 1.0, 1),
            "large",
        )
        measured.append(nbytes)
        report["transitions"].append(
            {
                "state_bytes": graph_state_storage_bytes(current),
                "next_state_bytes": (
                    graph_state_storage_bytes(next_state) if next_state is not None else 0
                ),
                "transition_bytes": nbytes,
                "state_variables": int(current.variable_features.shape[0]),
                "state_rows": int(current.row_features.shape[0]),
                "state_edges": int(current.edge_indices.shape[1]),
                "candidates": int(current.actions.size),
            }
        )
        collected += 1
        if next_state is None:
            break
        current = next_state
        state = env.current_state

    typical = max(measured) if measured else 0
    hold4 = replay.can_hold_large(LARGE_SAMPLE_QUOTA, typical)
    snapshot = replay.snapshot()
    report.update(
        {
            "transition_count": collected,
            "transition_bytes_max": typical,
            "transition_bytes_mean": int(sum(measured) / len(measured)) if measured else 0,
            "large_byte_budget": LARGE_BYTE_LIMIT,
            "large_pool_can_hold_4": hold4,
            "large_pool_count": snapshot.large_count,
            "large_pool_bytes": snapshot.large_bytes,
            "gate_passed": bool(collected > 0 and hold4 and typical > 0),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    env.close()
    if not report["gate_passed"]:
        raise SystemExit(
            f"real_02 large-pool gate failed: max_transition_bytes={typical} "
            f"hold4={hold4} count={collected}"
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
