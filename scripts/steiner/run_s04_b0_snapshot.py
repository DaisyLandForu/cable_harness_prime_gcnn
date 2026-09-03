#!/usr/bin/env python3
"""Build deterministic real-SCIP B0 snapshots and evaluate the S04 Gate."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import ecole  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from steiner_branching.contracts import content_sha256  # noqa: E402
from steiner_branching.data.generate import GeneratorConfig, generate_graph  # noqa: E402
from steiner_branching.data.split import split_for_synthetic_seed  # noqa: E402
from steiner_branching.milp.mcf import build_mcf  # noqa: E402
from steiner_branching.models.milp_gcnn import (  # noqa: E402
    MilpBipartiteGCNN,
    config_sha256,
    load_b0_config,
    model_state_sha256,
    parameter_count,
    score_state,
    state_tensors,
)
from steiner_branching.solver.bipartite_observation import (  # noqa: E402
    MILP_BIPARTITE_V1,
    SteinerNodeBipartite,
    with_legal_edge_actions,
)
from steiner_branching.solver.branchability import (  # noqa: E402
    configure_p1,
    load_s03_config,
)
from steiner_branching.solver.graph_state import candidate_exact_closure  # noqa: E402


DEFAULT_CONFIG = REPO / "configs/steiner/models/b0_milp_gcnn_v1.yml"
DEFAULT_SNAPSHOT = REPO / "docs/steiner/phases/S04/S04_FORWARD_SNAPSHOT.json"
DEFAULT_SUMMARY = REPO / "docs/steiner/phases/S04/S04_GATE_SUMMARY.json"
P1_CONFIG = REPO / "configs/steiner/experiments/s03_branchability_pilot_v1.yml"


def _resolve(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO / value


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    rank = max(1, math.ceil(float(percentile) * len(ordered)))
    return ordered[rank - 1]


def _profile(
    model: MilpBipartiteGCNN,
    tensors: tuple[torch.Tensor, ...],
    *,
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(*tensors)
        timings: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter_ns()
            model(*tensors)
            timings.append((time.perf_counter_ns() - started) / 1000.0)
    return {
        "iterations": repeats,
        "p50_microseconds": _nearest_rank(timings, 0.50),
        "p95_microseconds": _nearest_rank(timings, 0.95),
        "max_microseconds": max(timings),
    }


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def run_snapshot(config_path: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    if os.environ.get("STEINER_SOLVER_STACK_ID") != "scip804-ecole081-pyscipopt430":
        raise RuntimeError("S04 snapshots must enter through run_with_scip804.sh")
    config = load_b0_config(_resolve(config_path))
    snapshot_config = config["snapshot"]
    seed = int(snapshot_config["generator_seed"])
    if split_for_synthetic_seed(seed, policy_path=_resolve(config["split_policy"])) != "train":
        raise RuntimeError("S04 snapshot seed is outside the frozen train split")

    torch.set_num_threads(int(config["cpu_threads"]))
    torch.manual_seed(int(config["model_seed"]))
    torch.use_deterministic_algorithms(True)
    architecture = config["architecture"]
    model = MilpBipartiteGCNN(
        embedding_dim=int(architecture["embedding_dim"]),
        hidden_dim=int(architecture["hidden_dim"]),
    ).cpu()
    model.eval()

    graph = generate_graph(
        GeneratorConfig(
            family=str(snapshot_config["family"]),
            n_nodes=int(snapshot_config["n_nodes"]),
            n_terminals=int(snapshot_config["n_terminals"]),
            seed=seed,
        )
    )
    build = build_mcf(graph, configure_correctness_profile=False, hide_output=True)
    p1_config = load_s03_config(P1_CONFIG)
    effective_parameters = configure_p1(build.model, p1_config, "relpscost")
    environment = ecole.environment.Branching(
        observation_function=SteinerNodeBipartite(), pseudo_candidates=False
    )
    environment.seed(int(config["solver_seed"]))
    ecole_model = ecole.scip.Model.from_pyscipopt(build.model)
    observation, action_set, _reward, done, _info = environment.reset(ecole_model)

    snapshots: list[dict[str, Any]] = []
    full_timings: list[dict[str, float | int]] = []
    closure_timings: list[dict[str, float | int]] = []
    candidates_seen = 0
    candidates_mapped = 0
    finite_state_count = 0
    max_error = 0.0
    argmax_matches = 0
    max_states = int(snapshot_config["max_states"])
    while observation is not None and action_set is not None and not done and len(snapshots) < max_states:
        full = with_legal_edge_actions(observation, action_set, build.metadata)
        if full.candidate_count == 0:
            break
        closure = candidate_exact_closure(full)
        full.validate()
        closure.validate()
        finite_state_count += 1
        with torch.inference_mode():
            full_logits_tensor = score_state(model, full)
            closure_logits_tensor = score_state(model, closure)
        full_logits = full_logits_tensor.detach().cpu().numpy().astype(np.float64)
        closure_logits = closure_logits_tensor.detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(full_logits).all() or not np.isfinite(closure_logits).all():
            raise RuntimeError("B0 produced a non-finite candidate logit")
        error = float(np.max(np.abs(full_logits - closure_logits)))
        full_argmax = int(np.argmax(full_logits))
        closure_argmax = int(np.argmax(closure_logits))
        argmax_match = full_argmax == closure_argmax
        max_error = max(max_error, error)
        argmax_matches += int(argmax_match)
        candidates_seen += full.candidate_count
        candidates_mapped += int(full.candidate_edge_ids.size)

        warmup = int(snapshot_config["warmup_iterations"])
        repeats = int(snapshot_config["timed_iterations"])
        full_timings.append(_profile(model, state_tensors(full), warmup=warmup, repeats=repeats))
        closure_timings.append(
            _profile(model, state_tensors(closure), warmup=warmup, repeats=repeats)
        )
        snapshots.append(
            {
                "state_index": len(snapshots),
                "full_state_sha256": full.sha256,
                "closure_state_sha256": closure.sha256,
                "full_shape": {
                    "constraints": int(full.constraint_features.shape[0]),
                    "variables": int(full.variable_features.shape[0]),
                    "edges": int(full.edge_features.shape[0]),
                },
                "closure_shape": {
                    "constraints": int(closure.constraint_features.shape[0]),
                    "variables": int(closure.variable_features.shape[0]),
                    "edges": int(closure.edge_features.shape[0]),
                },
                "candidate_global_variable_ids": [
                    int(value) for value in full.variable_global_ids[full.candidate_indices]
                ],
                "candidate_edge_ids": [int(value) for value in full.candidate_edge_ids],
                "candidate_names": list(full.candidate_names),
                "full_logits": [float(value) for value in full_logits],
                "closure_logits": [float(value) for value in closure_logits],
                "max_absolute_logit_error": error,
                "argmax_match": argmax_match,
                "argmax_edge_id": int(full.candidate_edge_ids[full_argmax]),
            }
        )
        selected_position = int(np.argmin(full.candidate_edge_ids))
        selected_action = int(full.candidate_indices[selected_position])
        observation, action_set, _reward, done, _info = environment.step(selected_action)

    deterministic_snapshot = {
        "schema_version": 1,
        "stage": "S04",
        "model_id": config["model_id"],
        "model_seed": int(config["model_seed"]),
        "model_state_sha256": model_state_sha256(model),
        "bipartite_schema_id": MILP_BIPARTITE_V1.schema_id,
        "bipartite_schema_sha256": MILP_BIPARTITE_V1.sha256,
        "config_sha256": config_sha256(config),
        "graph_sha256": graph.graph_sha256,
        "metadata_sha256": build.metadata.sha256,
        "solver_seed": int(config["solver_seed"]),
        "action_policy": snapshot_config["action_policy"],
        "snapshots": snapshots,
    }
    snapshot_digest = content_sha256(deterministic_snapshot)
    mapping_rate = candidates_mapped / candidates_seen if candidates_seen else 0.0
    argmax_rate = argmax_matches / len(snapshots) if snapshots else 0.0
    gates = config["gate"]
    checks = {
        "minimum_snapshot_count": len(snapshots) >= int(gates["min_snapshots"]),
        "all_features_and_logits_finite": finite_state_count == len(snapshots)
        and all(
            np.isfinite(value).all()
            for state in snapshots
            for value in (state["full_logits"], state["closure_logits"])
        ),
        "action_to_edge_mapping_rate": mapping_rate
        >= float(gates["required_action_mapping_rate"]),
        "full_closure_logit_error": max_error
        <= float(gates["max_full_closure_logit_error"]),
        "full_closure_argmax_agreement": argmax_rate
        >= float(gates["required_argmax_agreement"]),
        "parameter_count_recorded": parameter_count(model) > 0,
        "cpu_inference_timing_recorded": bool(full_timings and closure_timings),
    }
    summary = {
        "schema_version": 1,
        "stage": "S04",
        "model_id": config["model_id"],
        "config_sha256": config_sha256(config),
        "snapshot_sha256": snapshot_digest,
        "environment": {
            "solver_stack_id": config["solver_stack_id"],
            "device": "cpu",
            "cpu_threads": int(config["cpu_threads"]),
            "cpu_model": _cpu_model(),
            "torch_version": torch.__version__,
            "ecole_version": ecole.__version__,
        },
        "model": {
            "parameter_count": parameter_count(model),
            "state_sha256": model_state_sha256(model),
            "embedding_dim": int(architecture["embedding_dim"]),
            "hidden_dim": int(architecture["hidden_dim"]),
            "message_passing_rounds": 1,
        },
        "observation": {
            "schema_id": MILP_BIPARTITE_V1.schema_id,
            "schema_sha256": MILP_BIPARTITE_V1.sha256,
            "variable_features": 19,
            "constraint_features": 5,
            "edge_features": 1,
            "snapshot_count": len(snapshots),
            "candidates_seen": candidates_seen,
            "candidates_mapped": candidates_mapped,
            "mapping_rate": mapping_rate,
        },
        "parity": {
            "max_absolute_logit_error": max_error,
            "argmax_matches": argmax_matches,
            "argmax_total": len(snapshots),
            "argmax_agreement": argmax_rate,
        },
        "inference": {
            "unit": "microseconds",
            "warmup_iterations_per_state": int(snapshot_config["warmup_iterations"]),
            "timed_iterations_per_state": int(snapshot_config["timed_iterations"]),
            "full": full_timings,
            "closure": closure_timings,
        },
        "solver": {
            "protocol_id": config["protocol_id"],
            "effective_parameters": effective_parameters,
            "finished_during_snapshot": bool(done),
        },
        "gate": {"checks": checks, "overall_pass": all(checks.values())},
    }
    return deterministic_snapshot, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--snapshot-output", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args()
    snapshot, summary = run_snapshot(args.config)
    _atomic_write_json(_resolve(args.snapshot_output), snapshot)
    _atomic_write_json(_resolve(args.summary_output), summary)
    print(json.dumps(summary["gate"], sort_keys=True))
    return 0 if summary["gate"]["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
