from __future__ import annotations

import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from steiner_branching.config import StrictConfigError, load_yaml_mapping
from steiner_branching.data.generate import GeneratorConfig, SYNTHETIC_FAMILIES, generate_graph
from steiner_branching.evaluation.s06_online import (
    MAIN_METHODS,
    S05_AUDITED_HEAD,
    S05_AUDITED_TAG,
    S06_CONFIG_FILE_SHA256,
    S06_CONFIG_PATH,
    S06_INSTANCES_FILE_SHA256,
    S06_INSTANCES_PATH,
    S06_SHARD_COUNT,
    S06Instance,
    S06Task,
    aggregate_s06,
    deterministic_argmax,
    deterministic_random_action,
    expand_s06_tasks,
    lineage_shard_assignments,
    load_s06_activation,
    load_s06_config,
    load_s06_instances,
    parse_scip_statistics,
    run_s06_task,
    task_sha256,
    tasks_for_lineage_shard,
    trace_replay_tasks,
)
from steiner_branching.learning.imitation import atomic_write_json
from steiner_branching.learning.teacher_data import file_sha256


REPO = Path(__file__).resolve().parents[2]


def _runner_module():
    path = REPO / "scripts/steiner/run_s06_online.py"
    spec = importlib.util.spec_from_file_location("s06_online_runner_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_s06_protocol_instances_and_task_matrix_are_frozen(tmp_path):
    assert file_sha256(S06_CONFIG_PATH) == S06_CONFIG_FILE_SHA256
    assert file_sha256(S06_INSTANCES_PATH) == S06_INSTANCES_FILE_SHA256
    config = load_s06_config(require_activation=False)
    instances = load_s06_instances()
    assert len(instances) == 30
    assert {family: sum(item.family == family for item in instances) for family in SYNTHETIC_FAMILIES} == {
        family: 6 for family in SYNTHETIC_FAMILIES
    }
    main, strong = expand_s06_tasks(config, instances)
    assert len(main) == 750 and len(strong) == 5
    assert {task.method for task in main} == set(MAIN_METHODS)
    assert {task.solver_seed for task in main} == {0, 1, 2, 3, 4}
    assert all(not task.gate_relevant for task in strong)
    assert {task.instance.family for task in strong} == set(SYNTHETIC_FAMILIES)

    changed = tmp_path / "changed.yml"
    changed.write_bytes(S06_CONFIG_PATH.read_bytes().replace(b"solver_seeds: [0, 1, 2, 3, 4]", b"solver_seeds: [0, 1]"))
    with pytest.raises(StrictConfigError, match="validation identity"):
        load_s06_config(changed, require_activation=False)


def test_s06_six_way_sharding_keeps_complete_lineages_and_is_disjoint():
    config = load_s06_config(require_activation=False)
    instances = load_s06_instances()
    main, strong = expand_s06_tasks(config, instances)
    assignments = lineage_shard_assignments(instances)
    assert set(assignments.values()) == set(range(S06_SHARD_COUNT))

    observed_task_ids: set[str] = set()
    for shard_index in range(S06_SHARD_COUNT):
        selected_instances = [
            item for item in instances
            if assignments[item.graph_sha256] == shard_index
        ]
        selected_main = tasks_for_lineage_shard(main, instances, shard_index)
        assert len(selected_instances) == 5
        assert {item.family for item in selected_instances} == set(SYNTHETIC_FAMILIES)
        assert len(selected_main) == 125
        for instance in selected_instances:
            lineage_tasks = [
                task for task in selected_main
                if task.instance.graph_sha256 == instance.graph_sha256
            ]
            assert len(lineage_tasks) == 25
            assert {task.solver_seed for task in lineage_tasks} == {0, 1, 2, 3, 4}
            assert {task.method for task in lineage_tasks} == set(MAIN_METHODS)
        assert observed_task_ids.isdisjoint(task.task_id for task in selected_main)
        observed_task_ids.update(task.task_id for task in selected_main)

    assert observed_task_ids == {task.task_id for task in main}
    assert [
        len(tasks_for_lineage_shard(strong, instances, index))
        for index in range(S06_SHARD_COUNT)
    ] == [5, 0, 0, 0, 0, 0]
    with pytest.raises(StrictConfigError, match="shard_count"):
        lineage_shard_assignments(instances, shard_count=5)
    with pytest.raises(StrictConfigError, match="shard_index"):
        tasks_for_lineage_shard(main, instances, S06_SHARD_COUNT)


def test_s06_activation_is_separate_and_fail_closed(tmp_path):
    with pytest.raises(StrictConfigError, match="activation is unavailable"):
        load_s06_activation(tmp_path / "missing.json")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=True
    ).stdout.strip()
    record = {
        "schema_version": 1,
        "stage": "S06",
        "audit_kind": "s06_online_protocol_and_implementation_preexecution_review",
        "verdict": "PASS",
        "blocking_findings": [],
        "execution_authorized": True,
        "verdict_source": "user_supplied_external_gpt_audit",
        "recorded_at_utc": "2026-09-08T00:00:00Z",
        "audited_branch": "research/steiner-migration",
        "audited_content_head": head,
        "protocol_yaml_sha256": S06_CONFIG_FILE_SHA256,
        "instance_manifest_sha256": S06_INSTANCES_FILE_SHA256,
        "s05_audited_head": S05_AUDITED_HEAD,
        "s05_audited_tag": S05_AUDITED_TAG,
        "formal_results_at_activation": "NOT_RUN",
        "s07_authorized": False,
        "test_and_final_access_authorized": False,
    }
    path = tmp_path / "activation.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert load_s06_activation(path)["verdict"] == "PASS"
    record["test_and_final_access_authorized"] = True
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(StrictConfigError, match="test_and_final"):
        load_s06_activation(path)


def test_s06_action_selection_is_permutation_invariant_and_fail_closed():
    candidates = [19, 3, 11, 7]
    first = deterministic_random_action(
        candidates, random_policy_seed=6001, graph_sha256="a" * 64,
        solver_seed=2, decision_index=9,
    )
    second = deterministic_random_action(
        list(reversed(candidates)), random_policy_seed=6001, graph_sha256="a" * 64,
        solver_seed=2, decision_index=9,
    )
    assert first == second and first in candidates
    assert deterministic_argmax(np.array([2.0, 5.0, 5.0]), [9, 7, 3]) == 3
    assert deterministic_argmax(np.array([5.0, 2.0, 5.0]), [3, 9, 7]) == 3
    with pytest.raises(ValueError, match="non-empty"):
        deterministic_random_action([], random_policy_seed=6001, graph_sha256="a" * 64, solver_seed=0, decision_index=0)
    with pytest.raises(FloatingPointError, match="non-finite"):
        deterministic_argmax(np.array([0.0, np.nan]), [1, 2])
    with pytest.raises(FloatingPointError, match="duplicate"):
        deterministic_argmax(np.array([0.0, 1.0]), [1, 1])


def test_scip_statistics_parser_extracts_exact_diagnostics():
    text = """
B&B Tree           :
  nodes            :          9 (4 internal, 5 leaves)
  max depth        :          3
Root Node          :
  First LP value   : +1.04000000000000e+02
Solution           :
  First Solution   : +1.18000000000000e+02   (in run 1, after 2 nodes, 0.07 seconds, depth 1, found by <relaxation>)
"""
    assert parse_scip_statistics(text) == {
        "branch_decisions": 4,
        "maximum_depth": 3,
        "root_lp_bound": 104.0,
        "exact_time_to_first_incumbent": 0.07,
    }
    with pytest.raises(ValueError, match="tree counts"):
        parse_scip_statistics("Solution: none")


def _result(method: str, *, bad_il: bool = False):
    learned = method == "b0_il"
    weak_random = method == "random_candidate"
    weak_mostinf = method == "mostinf"
    par2 = 3.0 if learned and bad_il else 1.0 if learned else 2.0 if weak_random else 1.5 if weak_mostinf else 0.8
    pdi = 3.0 if learned and bad_il else 1.0 if learned else 2.2 if weak_random else 1.7 if weak_mostinf else 0.9
    return {
        "status": "optimal", "solved": True,
        "metrics": {
            "solve_wall_seconds": par2, "par2_seconds": par2,
            "primal_dual_integral": pdi, "primal_bound": 100.0,
        },
        "correctness": {
            "invalid_action_count": 0, "mapping_failure_count": 0,
            "nan_score_count": 0, "unexpected_fallback_count": 0,
            "solution_validation_failure_count": 0,
        },
    }


def _write_matrix(tmp_path: Path, *, bad_il: bool = False):
    config = load_s06_config(require_activation=False)
    main, strong = expand_s06_tasks(config, load_s06_instances())
    for task in main + strong:
        atomic_write_json(tmp_path / f"{task.task_id}.json", {
            "protocol_file_sha256": S06_CONFIG_FILE_SHA256,
            "task_sha256": task_sha256(task),
            "execution_status": "completed",
            "result": _result(task.method, bad_il=bad_il),
        })
    return config, main, strong


def test_s06_aggregation_uses_complete_paired_instance_bootstrap(tmp_path):
    config, main, strong = _write_matrix(tmp_path)
    summary = aggregate_s06(config, main, strong, tmp_path)
    assert summary["completion"]["observed_completed_main_tasks"] == 750
    assert summary["gate"]["overall_pass"] is True
    for baseline in ("random_candidate", "mostinf"):
        value = summary["comparisons"][baseline]
        assert value["pairs"] == 150
        assert value["lexicographic_wins"] == 150
        assert value["positive_family_count"] == 5
        assert value["par2_effect_ci95"][0] > 0

    # A protocol-visible failed correctness outcome must not be averaged away.
    first = next(task for task in main if task.method == "b0_il")
    path = tmp_path / f"{first.task_id}.json"
    shard = json.loads(path.read_text())
    shard["result"]["correctness"]["mapping_failure_count"] = 1
    path.write_text(json.dumps(shard), encoding="utf-8")
    failed = aggregate_s06(config, main, strong, tmp_path)
    assert failed["gate"]["correctness"]["zero_mapping_failures"] is False
    assert failed["gate"]["overall_pass"] is False


def test_s06_trace_trigger_is_diagnostic_and_deterministic(tmp_path):
    config, main, strong = _write_matrix(tmp_path, bad_il=True)
    del strong
    shards = {
        task.task_id: json.loads((tmp_path / f"{task.task_id}.json").read_text())
        for task in main
    }
    traces = trace_replay_tasks(main, shards, config)
    assert len(traces) == 30 * 5 * len(MAIN_METHODS)
    assert all(task.trace and not task.gate_relevant for task in traces)
    assert len({task.task_id for task in traces}) == len(traces)


def test_s06_distributed_barrier_rejects_missing_and_wrong_shards(tmp_path):
    runner = _runner_module()
    config = load_s06_config(require_activation=False)
    instances = load_s06_instances()
    main, strong = expand_s06_tasks(config, instances)
    all_tasks = main + strong
    shard_dir = tmp_path / "shards"
    run_dir = tmp_path / "run"
    assignments = lineage_shard_assignments(instances)
    for task in all_tasks:
        shard_index = assignments[task.instance.graph_sha256]
        atomic_write_json(shard_dir / f"{task.task_id}.json", {
            "protocol_file_sha256": S06_CONFIG_FILE_SHA256,
            "task_sha256": task_sha256(task),
            "execution_status": "completed",
            "distributed_execution": {
                "phase": "main",
                "shard_index": shard_index,
                "shard_count": S06_SHARD_COUNT,
            },
            "result": _result(task.method),
        })
    for shard_index in range(S06_SHARD_COUNT):
        assigned = tasks_for_lineage_shard(all_tasks, instances, shard_index)
        runner._write_phase_manifest(
            runner._phase_manifest_path(run_dir, "main", shard_index),
            phase="main",
            shard_index=shard_index,
            status="completed",
            instances=instances,
            assigned_tasks=assigned,
            started_at="2026-09-09T00:00:00Z",
        )
    assert len(runner._load_phase_barrier(
        run_dir,
        phase="main",
        all_tasks=all_tasks,
        instances=instances,
        shard_dir=shard_dir,
    )) == S06_SHARD_COUNT

    first = main[0]
    first_path = shard_dir / f"{first.task_id}.json"
    envelope = json.loads(first_path.read_text(encoding="utf-8"))
    envelope["distributed_execution"]["shard_index"] = 1
    atomic_write_json(first_path, envelope)
    with pytest.raises(RuntimeError, match="wrong shard"):
        runner._load_phase_barrier(
            run_dir,
            phase="main",
            all_tasks=all_tasks,
            instances=instances,
            shard_dir=shard_dir,
        )

    envelope["distributed_execution"]["shard_index"] = 0
    atomic_write_json(first_path, envelope)
    runner._phase_manifest_path(run_dir, "main", 5).unlink()
    with pytest.raises(RuntimeError, match="missing shard 5"):
        runner._load_phase_barrier(
            run_dir,
            phase="main",
            all_tasks=all_tasks,
            instances=instances,
            shard_dir=shard_dir,
        )


def test_s06_foreground_launcher_rejects_duplicate_shard_job():
    lock_path = (
        REPO / "results/steiner/raw/s06/s06-il-online-v1/locks/main-shard-0.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        process = subprocess.run(
            ["bash", "scripts/steiner/run_s06_online_shard.sh", "0", "main"],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )
    assert process.returncode == 75
    assert "already running" in process.stderr


def test_s06_real_frozen_stack_toy_solve_all_methods_agree():
    config = load_s06_config(require_activation=False)
    graph = generate_graph(GeneratorConfig("sparse_erdos_renyi", 24, 4, 100300))
    instance = S06Instance(
        "engineering", 0, "sparse_erdos_renyi", 100300, graph.graph_sha256,
        "engineering-s100300", 24, 4,
    )
    results = {
        method: run_s06_task(
            S06Task(f"engineering--{method}", instance, 0, method, False), config
        )
        for method in (*MAIN_METHODS, "fullstrong")
    }
    assert {result["status"] for result in results.values()} == {"optimal"}
    assert {result["metrics"]["primal_bound"] for result in results.values()} == {107.0}
    assert all(result["metrics"]["primal_dual_integral"] >= 0.0 for result in results.values())
    assert all(sum(result["correctness"].values()) == 0 for result in results.values())
    assert results["b0_il"]["checkpoint"]["training_seed"] == 202
    assert results["b0_il"]["overhead"]["feature_extraction_seconds"] > 0.0
    assert results["mostinf"]["overhead"]["policy_callback_seconds"] == 0.0
