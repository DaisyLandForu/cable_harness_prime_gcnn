#!/usr/bin/env python3
"""Compare Ecole and C++ first-state tensors on syn_medium_s101."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from rl_branching import BBMDPBranchingEnv, BBMDPConfig
from rl_branching.gcnn_dqn import graph_state_tensors, stable_graph_argmax
from rl_branching.gcnn_model import BipartiteGCNNQNetwork
from rl_branching.graph_features import (
    GraphState,
    candidate_twohop_state,
    extract_graph_state,
)
from rl_branching.scip_profile import dump_effective_search_params, sha256_text

INSTANCE = Path("data/instances/train/syn_medium_s101.cip")
OUTPUT = Path("results/probes/syn_medium_s101_parity.json")
CPP_STATE = Path("results/probes/syn_medium_s101_cpp_first_state.json")
TIME_LIMIT = 60.0
NODE_LIMIT = -1
FEATURE_TOL = 1.0e-6
Q_TOL = 1.0e-5


def _reshape(values, rows, cols):
    array = np.asarray(values, dtype=np.float32)
    return array.reshape(rows, cols) if cols > 1 else array


def graph_from_dump(block: dict) -> GraphState:
    n_vars = int(block["variable_count"])
    n_rows = int(block["row_count"])
    n_edges = int(block["edge_count"])
    return GraphState(
        row_features=np.asarray(block["row_features"], dtype=np.float32).reshape(n_rows, 14),
        variable_features=np.asarray(block["variable_features"], dtype=np.float32).reshape(n_vars, 25),
        edge_indices=np.asarray(
            [block["edge_row_indices"], block["edge_variable_indices"]], dtype=np.int64
        ),
        edge_features=np.asarray(block["edge_features"], dtype=np.float32).reshape(n_edges, 3),
        global_features=np.asarray(block["global_features"], dtype=np.float32),
        variable_categories=np.asarray(block["variable_categories"], dtype=np.float32).reshape(n_vars, 6),
        row_categories=np.asarray(block["row_categories"], dtype=np.float32).reshape(n_rows, 6),
        actions=np.asarray(block["candidate_indices"], dtype=np.int64),
        candidate_names=tuple(block["candidate_names"]),
        variable_names=tuple(block["variable_names"]),
    )


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return float("inf")
    return float(np.max(np.abs(left_array - right_array))) if left_array.size else 0.0


def compare_graphs(python_state: GraphState, cpp_state: GraphState) -> dict:
    return {
        "candidate_names_equal": list(python_state.candidate_names) == list(cpp_state.candidate_names),
        "variable_names_equal": list(python_state.variable_names) == list(cpp_state.variable_names),
        "variable_features": max_abs(python_state.variable_features, cpp_state.variable_features),
        "dsu_features": max_abs(
            python_state.variable_features[:, 19:25], cpp_state.variable_features[:, 19:25]
        ),
        "row_features": max_abs(python_state.row_features, cpp_state.row_features),
        "edge_features": max_abs(python_state.edge_features, cpp_state.edge_features),
        "global_features": max_abs(python_state.global_features, cpp_state.global_features),
        "edge_indices_equal": np.array_equal(python_state.edge_indices, cpp_state.edge_indices),
        "actions_equal": np.array_equal(python_state.actions, cpp_state.actions),
    }


def q_values(model: BipartiteGCNNQNetwork, state: GraphState) -> np.ndarray:
    with torch.no_grad():
        values = model(*graph_state_tensors(state, torch.device("cpu"))).cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def main() -> int:
    env = BBMDPBranchingEnv(
        BBMDPConfig(seed=0, time_limit=TIME_LIMIT, node_limit=NODE_LIMIT)
    )
    state = env.reset(INSTANCE)
    if state.terminated or state.truncated or state.observation is None:
        raise SystemExit("Ecole did not reach a live branching state")
    seeds = {
        "randomization/randomseedshift": int(env.scip_parameter("randomization/randomseedshift")),
        "randomization/permutationseed": int(env.scip_parameter("randomization/permutationseed")),
        "randomization/lpseed": int(env.scip_parameter("randomization/lpseed")),
        "randomization/permuteconss": bool(env.scip_parameter("randomization/permuteconss")),
        "randomization/permutevars": bool(env.scip_parameter("randomization/permutevars")),
    }
    python_dump = env.effective_search_params_dump()
    python_core = env.effective_search_params_sha256(include_seeds=False)
    python_full = sha256_text(python_dump)
    full = extract_graph_state(state.observation, state.action_set)
    twohop = candidate_twohop_state(full)
    env.close()

    subprocess.run(
        [
            "./build/graph_probe",
            "--instance",
            str(INSTANCE),
            "--scip-profile",
            "configs/scip/project-production-v1.set",
            "--output",
            str(CPP_STATE),
            "--dump-state",
            "--time-limit",
            str(TIME_LIMIT),
            "--node-limit",
            str(NODE_LIMIT),
            "--threads",
            "1",
            "--randomseedshift",
            str(seeds["randomization/randomseedshift"]),
            "--permutationseed",
            str(seeds["randomization/permutationseed"]),
            "--lpseed",
            str(seeds["randomization/lpseed"]),
        ],
        check=True,
    )
    cpp = json.loads(CPP_STATE.read_text())
    cpp_full = graph_from_dump(cpp["full"])
    cpp_twohop = graph_from_dump(cpp["twohop"])

    torch.manual_seed(0)
    model = BipartiteGCNNQNetwork(embedding_dim=16, hidden_dim=32)
    model.eval()
    python_full_q = q_values(model, full)
    python_twohop_q = q_values(model, twohop)
    cpp_full_q = q_values(model, cpp_full)
    cpp_twohop_q = q_values(model, cpp_twohop)

    report = {
        "instance": str(INSTANCE),
        "time_limit": TIME_LIMIT,
        "node_limit": NODE_LIMIT,
        "ecole_remapped_seeds": seeds,
        "python_effective_search_params_sha256": python_full,
        "python_effective_search_params_core_sha256": python_core,
        "cpp_effective_search_params_sha256": cpp["effective_search_params_sha256"],
        "cpp_effective_search_params_core_sha256": cpp["effective_search_params_core_sha256"],
        "effective_dump_equal": python_dump == cpp["effective_dump"].replace("\\n", "\n"),
        "lp_iterations": {
            "python": int(state.info.get("lp_iterations", -1)),
            "cpp": int(cpp["lp_iterations"]),
        },
        "local_lower_bounds_max_abs": max_abs(
            state.observation.local_lower_bounds,
            np.asarray(cpp["full"]["local_lower_bounds"], dtype=np.float32),
        ),
        "full": compare_graphs(full, cpp_full),
        "twohop": compare_graphs(twohop, cpp_twohop),
        "q": {
            "python_full_vs_twohop": max_abs(python_full_q, python_twohop_q),
            "cpp_full_vs_twohop": max_abs(cpp_full_q, cpp_twohop_q),
            "python_vs_cpp_full": max_abs(python_full_q, cpp_full_q),
            "python_vs_cpp_twohop": max_abs(python_twohop_q, cpp_twohop_q),
            "python_full_argmax": stable_graph_argmax(python_full_q, full),
            "python_twohop_argmax": stable_graph_argmax(python_twohop_q, twohop),
            "cpp_full_argmax": stable_graph_argmax(cpp_full_q, cpp_full),
            "cpp_twohop_argmax": stable_graph_argmax(cpp_twohop_q, cpp_twohop),
        },
    }
    report["q"]["argmax_equal"] = (
        report["q"]["python_full_argmax"]
        == report["q"]["python_twohop_argmax"]
        == report["q"]["cpp_full_argmax"]
        == report["q"]["cpp_twohop_argmax"]
    )
    feature_ok = all(
        report[side][key] <= FEATURE_TOL
        if isinstance(report[side][key], float)
        else report[side][key]
        for side in ("full", "twohop")
        for key in report[side]
    )
    q_ok = (
        report["q"]["python_vs_cpp_full"] <= Q_TOL
        and report["q"]["python_vs_cpp_twohop"] <= Q_TOL
        and report["q"]["argmax_equal"]
        and report["effective_dump_equal"]
        and report["local_lower_bounds_max_abs"] <= FEATURE_TOL
    )
    report["gate_passed"] = bool(feature_ok and q_ok)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["gate_passed"]:
        raise SystemExit("syn_medium_s101 parity gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
