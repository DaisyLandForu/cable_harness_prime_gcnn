from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from steiner_branching.config import StrictConfigError
from steiner_branching.learning.formal_protocol import (
    FORMAL_CONFIG_FILE_SHA256,
    FORMAL_CONFIG_PATH,
    FORMAL_TRAINING_SEEDS,
    expand_formal_tasks,
    load_formal_audit_record,
    load_s05_formal_config,
)
from steiner_branching.learning.teacher_data import file_sha256


REPO = Path(__file__).resolve().parents[2]


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal_protocol_activation_hash_and_exact_task_matrix(tmp_path):
    assert file_sha256(FORMAL_CONFIG_PATH) == FORMAL_CONFIG_FILE_SHA256
    record = load_formal_audit_record()
    assert record["verdict"] == "PASS" and record["s06_authorized"] is False
    config = load_s05_formal_config(require_activation=True)
    plans = expand_formal_tasks(config)
    assert len(plans) == 315
    assert {plan.task.teacher_seed for plan in plans} == {1001, 1002, 1003}
    assert {plan.role for plan in plans} == {
        "train", "validation_select", "validation_gate"
    }
    role_counts = {
        role: sum(plan.role == role for plan in plans)
        for role in ("train", "validation_select", "validation_gate")
    }
    assert role_counts == {"train": 180, "validation_select": 45, "validation_gate": 90}
    graph_seeds = {plan.task.generator_seed for plan in plans}
    assert len(graph_seeds) == 105
    assert min(seed for seed in graph_seeds if seed < 200000) == 101000
    assert max(seed for seed in graph_seeds if seed < 200000) == 101059
    assert min(seed for seed in graph_seeds if seed >= 200000) == 201000
    assert max(seed for seed in graph_seeds if seed >= 200000) == 201044

    changed = tmp_path / "changed.yml"
    changed.write_bytes(FORMAL_CONFIG_PATH.read_bytes().replace(b"epochs: 40", b"epochs: 41"))
    with pytest.raises(StrictConfigError, match="byte-exact"):
        load_s05_formal_config(changed, require_activation=False)


def _records_for_frozen_quotas(config):
    records = []
    counter = 0
    for family, buckets in config["state_selection"]["train_quotas"].items():
        for bucket, quota in buckets.items():
            for index in range(quota):
                counter += 1
                records.append(_record(counter, "train", family, bucket, index))
    for role, quota in (("validation_select", 32), ("validation_gate", 64)):
        for family in config["state_selection"]["train_quotas"]:
            for index in range(quota):
                counter += 1
                records.append(_record(counter, role, family, "medium-mid", index))
    return records


def _record(counter: int, role: str, family: str, bucket: str, index: int):
    return {
        "path": f"states/{counter}.npz",
        "file_sha256": f"{counter:064x}",
        "semantic_sha256": f"{counter + 10000:064x}",
        "task_id": f"task-{counter}",
        "role": role,
        "split": "train" if role == "train" else "validation_iid",
        "family": family,
        "bucket_id": bucket,
        "graph_sha256": f"{counter + 20000:064x}",
        "teacher_seed": 1001 + counter % 3,
        "state_index": index,
        "state_valid": True,
        "all_tie": False,
        "candidate_count": 2,
    }


def test_formal_selection_meets_exact_quotas_and_fails_role_leakage():
    collector = _load_script(
        "s05_formal_collector_test", "scripts/steiner/collect_s05_formal_teacher.py"
    )
    config = load_s05_formal_config(require_activation=False)
    records = _records_for_frozen_quotas(config)
    selected, issues = collector.select_formal_records(config, records)
    assert not issues
    assert {role: len(values) for role, values in selected.items()} == {
        "train": 640, "validation_select": 160, "validation_gate": 320
    }
    leaked = copy.deepcopy(records)
    leaked[-1]["graph_sha256"] = leaked[0]["graph_sha256"]
    with pytest.raises(ValueError, match="crosses formal roles"):
        collector.select_formal_records(config, leaked)
    short, issues = collector.select_formal_records(config, records[:-1])
    assert len(short["validation_gate"]) == 319
    assert any("shortage" in issue for issue in issues)


def _formal_reports(effect: float = 0.25):
    reports = []
    families = [
        "sparse_erdos_renyi", "random_geometric", "grid_with_holes",
        "community_block", "bridge_bottleneck",
    ]
    base_records = []
    for index in range(320):
        graph_index = index % 30
        base_records.append({
            "task_id": f"task-{index}",
            "semantic_sha256": f"{index:064x}",
            "graph_sha256": f"{graph_index:064x}",
            "family": families[graph_index // 6],
            "teacher_seed": 1001 + index % 3,
            "state_index": index,
            "model_regret": 0.5 - effect,
            "random_regret": 0.5,
        })
    for seed in FORMAL_TRAINING_SEEDS:
        reports.append({
            "training_seed": seed,
            "status": "completed",
            "formal_gate_evaluated": False,
            "reload_max_absolute_error": 0.0,
            "reload_state_dict_bit_exact": True,
            "gate_state_records": copy.deepcopy(base_records),
        })
    return reports


def test_formal_gate_bootstraps_graph_lineages_and_requires_positive_effect():
    aggregator = _load_script(
        "s05_formal_aggregate_test", "scripts/steiner/aggregate_s05_formal_training.py"
    )
    config = load_s05_formal_config(require_activation=False)
    teacher = {
        "teacher_gate": {
            "status": "PASS", "checks": {"zero_split_or_role_leakage": True}
        }
    }
    gate = aggregator.evaluate_formal_gate(config, teacher, _formal_reports())
    assert gate["status"] == "PASS"
    assert gate["bootstrap_lineages"] == 30
    assert gate["primary_bootstrap_95_ci"] == pytest.approx([0.25, 0.25])
    assert gate["seed_regret_coefficient_of_variation"] == 0.0

    failed = aggregator.evaluate_formal_gate(config, teacher, _formal_reports(effect=-0.01))
    assert failed["status"] == "FAIL"
    assert not failed["checks"]["primary_bootstrap_ci_lower_bound_above_zero"]
