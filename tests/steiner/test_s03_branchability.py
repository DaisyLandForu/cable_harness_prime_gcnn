from __future__ import annotations

import copy
from pathlib import Path
import subprocess

import pytest

from steiner_branching.config import StrictConfigError
from steiner_branching.solver.branchability import (
    CandidateObserver,
    aggregate_results,
    atomic_write_json,
    config_sha256,
    expand_tasks,
    load_s03_config,
    parse_native_probe_output,
    task_sha256,
)


CONFIG_PATH = Path("configs/steiner/experiments/s03_branchability_pilot_v1.yml")


def test_s03_preregistration_expands_frozen_train_matrix():
    config = load_s03_config(CONFIG_PATH)
    formal, ramp = expand_tasks(config)
    assert len(formal) == 5 * 3 * 2 * 3 == 90
    assert len(ramp) == 1 + 3 + 6 == 10
    assert len({task.instance_id for task in formal}) == 30
    assert {task.baseline for task in formal} == {"scip_default", "relpscost", "mostinf"}
    strong = [task for task in formal if task.strong_branch]
    assert len(strong) == 5 * 2
    assert all(task.baseline == "relpscost" and task.bucket_id == "small-low" for task in strong)
    assert all(100000 <= task.generator_seed <= 199999 for task in formal + ramp)
    assert "build/steiner_s03_sb_probe" in Path(".gitignore").read_text(encoding="utf-8")


def test_s03_config_rejects_unknown_or_changed_gate():
    config = load_s03_config(CONFIG_PATH)
    bad = copy.deepcopy(config)
    bad["surprise"] = True
    with pytest.raises(StrictConfigError, match="unknown"):
        from steiner_branching.solver import branchability

        branchability._require_keys(  # type: ignore[attr-defined]
            bad,
            set(config),
            "S03 config",
        )


def test_candidate_observer_maps_transformed_edge_names_once_per_node():
    class Variable:
        def __init__(self, name: str):
            self.name = name

    class Node:
        def __init__(self, number: int, depth: int):
            self.number = number
            self.depth = depth

        def getNumber(self):
            return self.number

        def getDepth(self):
            return self.depth

    class Model:
        node = Node(1, 0)

        def getLPBranchCands(self):
            variables = [Variable("t_stp_x_e00000000"), Variable("bad")]
            return variables, [0.5, 0.5], [0.5, 0.5], 2, 2, 0

        def getCurrentNode(self):
            return self.node

        def getLPObjVal(self):
            return 12.0

    observer = CandidateObserver()
    observer.initialise(["stp_x_e00000000"])
    observer.model = Model()
    observer.observe()
    observer.observe()
    assert observer.branch_states == 1
    assert observer.candidates_observed == 2
    assert observer.candidates_mapped == 1
    assert observer.mapping_failures == ["bad"]
    assert observer.root_fractional_edges == 2


def test_native_probe_parser_preserves_valid_and_tie_flags():
    text = """
noise
S03_STATE node=1 depth=0 legal=3 evaluated=3 mapped=3 fully_valid=3 finite_scores=3 lp_errors=0 score_min=1 score_max=2 lp_iterations_delta=4 sb_lp_iterations_delta=4 sb_calls_delta=3 valid=1 all_tie=0
S03_FINAL status=userinterrupt nodes=1 states=1 lp_iterations=10 peak_rss_mb=42.5
"""
    states, final = parse_native_probe_output(text)
    assert states == [
        {
            "node": 1,
            "depth": 0,
            "legal": 3,
            "evaluated": 3,
            "mapped": 3,
            "fully_valid": 3,
            "finite_scores": 3,
            "lp_errors": 0,
            "score_min": 1.0,
            "score_max": 2.0,
            "lp_iterations_delta": 4,
            "sb_lp_iterations_delta": 4,
            "sb_calls_delta": 3,
            "valid": True,
            "all_tie": False,
        }
    ]
    assert final["status"] == "userinterrupt"
    assert final["peak_rss_mb"] == 42.5


def test_native_scip804_probe_executes_real_strong_branch_lps(tmp_path: Path):
    pytest.importorskip("pyscipopt")
    from steiner_branching.data.generate import GeneratorConfig, generate_graph
    from steiner_branching.milp.mcf import build_mcf

    native = Path("build/steiner_s03_sb_probe")
    assert native.is_file(), "run `make steiner-s03-probe` before the S03 Gate suite"
    graph = generate_graph(
        GeneratorConfig(
            family="sparse_erdos_renyi", n_nodes=48, n_terminals=5, seed=100300
        )
    )
    build = build_mcf(graph, configure_correctness_profile=False)
    instance = tmp_path / "native-probe.cip"
    build.model.writeProblem(str(instance))
    process = subprocess.run(
        [
            str(native), "--instance", str(instance), "--seed", "0",
            "--max-states", "1", "--iteration-limit", "10000",
            "--candidate-limit", "0", "--idempotent", "0",
            "--tie-tolerance", "1e-9", "--time-limit", "30",
            "--node-limit", "100", "--memory-limit", "1024",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    states, final = parse_native_probe_output(process.stdout)
    assert len(states) == 1
    state = states[0]
    assert state["valid"]
    assert state["mapped"] == state["evaluated"] == state["legal"]
    assert state["sb_calls_delta"] >= state["evaluated"]
    assert state["sb_lp_iterations_delta"] > 0
    assert state["score_max"] >= state["score_min"]
    assert final["states"] == 1


def test_aggregate_gate_counts_missing_strong_states_as_invalid(tmp_path: Path):
    config = load_s03_config(CONFIG_PATH)
    formal, ramp = expand_tasks(config)
    digest = config_sha256(config)
    for task in formal + ramp:
        result = {
            "schema_version": 1,
            "config_sha256": digest,
            "task_sha256": task_sha256(task, digest),
            "task": task.to_dict(),
            "status": "optimal",
            "branchability": {
                "legal_decisions": 10,
                "candidates_observed": 2,
                "candidates_mapped": 2,
                "mapping_failures": [],
                "callback_errors": [],
            },
            "resources": {"peak_rss_mb": 100.0},
            "timing": {"build_seconds": 1.0},
            "model": {"continuous_flow_variables": 1000},
            "strong_branch": None,
        }
        if task.strong_branch:
            result["strong_branch"] = {
                "status": "completed",
                "states": [
                    {"valid": True, "all_tie": False},
                    {"valid": True, "all_tie": False},
                ],
            }
        atomic_write_json(tmp_path / f"{task.task_id}.json", result)
    summary = aggregate_results(config, formal, ramp, tmp_path)
    assert summary["gate"]["overall_pass"]
    assert summary["measurements"]["strong_valid_state_fraction"] == 1.0

    victim = next(task for task in formal if task.strong_branch)
    shard = tmp_path / f"{victim.task_id}.json"
    value = __import__("json").loads(shard.read_text(encoding="utf-8"))
    value["strong_branch"]["states"] = []
    atomic_write_json(shard, value)
    summary = aggregate_results(config, formal, ramp, tmp_path)
    assert summary["measurements"]["strong_states_missing_or_failed"] == 2
    assert summary["measurements"]["strong_valid_state_fraction"] == pytest.approx(0.9)
