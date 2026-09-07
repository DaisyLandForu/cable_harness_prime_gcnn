from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from steiner_branching.config import StrictConfigError, load_yaml_mapping
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
CONCURRENCY_AMENDMENT_PATH = (
    REPO
    / "configs/steiner/experiments/s05_teacher_il_formal_v1_concurrency_a1.yml"
)
CONCURRENCY_AMENDMENT_SHA256 = (
    "8ae53206bb55c30e052f489687200a0ca5cea787da7bcf8c0d49f9f9ce49451a"
)
SELECTION_REMEDIATION_PATH = (
    REPO
    / "configs/steiner/experiments/s05_teacher_il_formal_v2_selection_remediation.yml"
)
SELECTION_REMEDIATION_SHA256 = (
    "91e157a8d7853133e3c4fe5865e863763874e41622febc87b94a39e54f6b7913"
)


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


def test_formal_concurrency_amendment_changes_only_the_global_job_limit():
    assert file_sha256(CONCURRENCY_AMENDMENT_PATH) == CONCURRENCY_AMENDMENT_SHA256
    amendment = load_yaml_mapping(CONCURRENCY_AMENDMENT_PATH)
    base = load_s05_formal_config(require_activation=False)

    assert amendment["status"] == "preregistered_amendment_pending_gpt_audit"
    assert amendment["execution_authorized"] is False
    assert amendment["base_protocol"] == {
        "experiment_id": "s05-teacher-il-formal-v1",
        "content_head": "d1717a7ecb6043efd71678175a92325ac9ff4208",
        "yaml_path": "configs/steiner/experiments/s05_teacher_il_formal_v1.yml",
        "yaml_sha256": FORMAL_CONFIG_FILE_SHA256,
        "explanation_path": "docs/steiner/phases/S05/S05_FORMAL_PROTOCOL.md",
        "explanation_sha256": (
            "6b121bb215ad9ad5444ce2c8328dc7049b3abdfd0a7bf191fec888441ec572c9"
        ),
        "audit_record": "docs/steiner/audits/S05_FORMAL_PROTOCOL_AUDIT_RECORD.json",
        "audit_verdict": "PASS",
    }
    assert amendment["scope"] == "execution_scheduling_only"
    assert amendment["allowed_overrides"] == [{
        "path": "training.max_concurrent_training_jobs",
        "before": 2,
        "after": 5,
    }]
    assert base["training"]["max_concurrent_training_jobs"] == 2

    unchanged = amendment["unchanged_contract"]
    assert unchanged["training_seeds"] == base["training"]["formal_seeds"]
    for key in (
        "train_states", "epochs", "learning_rate", "weight_decay",
        "gradient_clip_norm", "target_temperature", "deterministic_algorithms",
        "cublas_workspace_config", "representative_checkpoint_seed",
    ):
        assert unchanged[key] == base["training"][key]
    requirements = amendment["execution_requirements"]
    assert requirements["gpu_model_per_job"] == base["training"]["gpu_model"]
    assert requirements["cpu_cores_per_job"] == base["training"]["cpu_cores_per_job"]
    assert requirements["ram_gib_per_job"] == base["training"]["ram_gib_per_job"]
    assert requirements["gpu_count_per_job"] == 1
    assert requirements["processes_per_gpu"] == 1
    assert requirements["distributed_training"] is False
    assert requirements["shared_optimizer_or_model_state"] is False
    assert amendment["failure_policy"]["on_missing_amendment_pass"] == (
        "keep_base_limit_of_two"
    )


def test_formal_v2_selection_remediation_preserves_family_and_total_budgets():
    assert file_sha256(SELECTION_REMEDIATION_PATH) == SELECTION_REMEDIATION_SHA256
    revision = load_yaml_mapping(SELECTION_REMEDIATION_PATH)
    base = load_s05_formal_config(require_activation=False)

    assert revision["status"] == "preregistered_revision_pending_gpt_audit"
    assert revision["execution_authorized"] is False
    assert revision["failed_v1_evidence"]["manifest_sha256"] == (
        "2bb3b4875d173571df5e4fb7e9c1e1d3f4615708308e891f4e4ccdbaac4c149c"
    )
    selection = revision["state_selection"]
    assert selection["initial_bucket_targets"] == base["state_selection"]["train_quotas"]
    assert selection["train_quota_per_family"] == 128
    expected = selection["expected_train_bucket_counts_from_sealed_manifest"]
    assert set(expected) == set(base["state_selection"]["train_quotas"])
    assert all(sum(buckets.values()) == 128 for buckets in expected.values())
    assert sum(sum(buckets.values()) for buckets in expected.values()) == 640
    assert selection["selected_state_counts"] == base["state_selection"][
        "selected_state_counts"
    ]

    unchanged = revision["unchanged_contract"]
    assert unchanged["train_states"] == base["training"]["train_states"]
    for key in (
        "epochs", "learning_rate", "weight_decay", "gradient_clip_norm",
        "target_temperature", "deterministic_algorithms", "cublas_workspace_config",
        "representative_checkpoint_seed",
    ):
        assert unchanged[key] == base["training"][key]
    assert revision["training_execution"]["formal_seeds"] == base["training"][
        "formal_seeds"
    ]
    assert revision["training_execution"]["max_concurrent_training_jobs"] == 5
    assert revision["teacher_evidence_reuse"]["rerun_teacher_tasks"] is False
    assert revision["teacher_evidence_reuse"]["allow_replacement_instances"] is False
    assert revision["failure_policy"]["v1_result_remains_fail"] is True
    assert revision["failure_policy"]["on_missing_revision_pass"] == (
        "stop_without_reselection_or_training"
    )


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
