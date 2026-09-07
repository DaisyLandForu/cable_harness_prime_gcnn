from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from steiner_branching.config import StrictConfigError, load_yaml_mapping
from steiner_branching.data.generate import GeneratorConfig, SYNTHETIC_FAMILIES, generate_graph
from steiner_branching.data.split import split_for_synthetic_seed
from steiner_branching.learning.formal_protocol import (
    FORMAL_CONFIG_FILE_SHA256,
    FORMAL_CONFIG_PATH,
    FORMAL_TRAINING_SEEDS,
    FORMAL_V2_CONFIG_FILE_SHA256,
    FORMAL_V2_CONFIG_PATH,
    expand_formal_tasks,
    load_formal_audit_record,
    load_formal_v2_activation,
    load_s05_formal_config,
    load_s05_formal_v2_config,
)
from steiner_branching.learning.formal_v3 import (
    V3_MODEL_SEEDS,
    expand_v3_tasks,
    load_v3_activation,
    load_v3_candidates,
    load_v3_protocol,
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
    "f101c038610d391b76c589fc81d08298160bac4883ed169f71953c7159137cbf"
)
FORMAL_V3_CONFIG_PATH = (
    REPO
    / "configs/steiner/experiments/s05_teacher_il_formal_v3_confirmatory_gate.yml"
)
FORMAL_V3_CONFIG_SHA256 = (
    "99e75a4d4fa69f805232c637d4fc0ae750fcfccb57e9979b561f422591006242"
)
FORMAL_V3_CANDIDATES_PATH = (
    REPO / "configs/steiner/experiments/s05_formal_v3_candidate_graphs.json"
)
FORMAL_V3_CANDIDATES_SHA256 = (
    "e65fc9a03fd683277570befe13b11f4b15ea981ee427568d56aeeccdd3847b56"
)
FORMAL_V2_RESULT_AUDIT_PATH = (
    REPO / "docs/steiner/audits/S05_FORMAL_V2_RESULT_AUDIT_RECORD.json"
)
FORMAL_V2_RESULT_AUDIT_SHA256 = (
    "a7cd6af30e9b1e98f46efb1b40f5876bb3f08cf75a45da59b09e319de33ec48a"
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


def test_formal_v2_pass_activation_and_effective_config_are_fail_closed(tmp_path):
    assert FORMAL_V2_CONFIG_PATH == SELECTION_REMEDIATION_PATH
    assert FORMAL_V2_CONFIG_FILE_SHA256 == SELECTION_REMEDIATION_SHA256
    activation = load_formal_v2_activation()
    assert activation["verdict"] == "PASS"
    assert activation["b1_status"] == "CLOSED"
    assert activation["teacher_tasks_rerun_authorized"] is False
    config = load_s05_formal_v2_config(require_activation=True)
    assert config["experiment_id"] == "s05-teacher-il-formal-v2"
    assert config["training"]["formal_seeds"] == [101, 202, 303, 404, 505]
    assert config["training"]["max_concurrent_training_jobs"] == 5
    assert config["training"]["epochs"] == 40
    assert config["training"]["checkpoint_root"].endswith("formal-v2")
    assert config["artifacts"]["report_root"].endswith("formal-v2")

    changed = tmp_path / "activation.json"
    bad = copy.deepcopy(activation)
    bad["b1_status"] = "OPEN"
    import json
    changed.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(StrictConfigError, match="b1_status"):
        load_formal_v2_activation(changed)


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


def _reference_v2_train_selection(records, revision):
    policy = revision["state_selection"]
    bucket_rank = {name: index for index, name in enumerate(policy["bucket_order"])}

    def key(record, fields):
        values = []
        for field in fields:
            if field == "bucket_order":
                values.append(bucket_rank[record["bucket_id"]])
            else:
                values.append(record[field])
        return tuple(values)

    seen = set()
    grouped = {}
    for record in records:
        if record["role"] != "train" or not record["state_valid"]:
            continue
        semantic = record["semantic_sha256"]
        if semantic in seen:
            raise ValueError("duplicate semantic_sha256")
        seen.add(semantic)
        grouped.setdefault((record["family"], record["bucket_id"]), []).append(record)

    selected = []
    for family, targets in policy["initial_bucket_targets"].items():
        family_selected = []
        remaining = []
        for bucket, target in targets.items():
            eligible = sorted(
                grouped.get((family, bucket), []),
                key=lambda record: key(record, policy["primary_order"]),
            )
            family_selected.extend(eligible[:target])
            remaining.extend(eligible[target:])
        shortage = policy["train_quota_per_family"] - len(family_selected)
        remaining.sort(key=lambda record: key(record, policy["fallback_order"]))
        family_selected.extend(remaining[:shortage])
        if len(family_selected) != policy["train_quota_per_family"]:
            raise ValueError("family has fewer than 128 eligible states")
        selected.extend(family_selected)
    return selected


def test_formal_v2_order_keys_match_manifest_schema_and_ignore_input_permutation():
    revision = load_yaml_mapping(SELECTION_REMEDIATION_PATH)
    policy = revision["state_selection"]
    manifest_record_schema = set(_record(1, "train", "community_block", "medium-mid", 0))
    for field in policy["primary_order"]:
        assert field in manifest_record_schema
    for field in policy["fallback_order"]:
        assert field == "bucket_order" or field in manifest_record_schema
    assert "bucket_id" in manifest_record_schema

    available = {
        "sparse_erdos_renyi": {"small-low": 126, "medium-mid": 192, "large-high": 0},
        "random_geometric": {"small-low": 288, "medium-mid": 3},
        "grid_with_holes": {"medium-mid": 288, "large-high": 92},
        "community_block": {"medium-mid": 506},
        "bridge_bottleneck": {"medium-mid": 60, "large-high": 288},
    }
    records = []
    counter = 0
    for family, buckets in available.items():
        for bucket, count in buckets.items():
            for index in range(count):
                counter += 1
                records.append(_record(counter, "train", family, bucket, index % 16))
    expected = _reference_v2_train_selection(records, revision)
    permutation = np.random.default_rng(20260907).permutation(len(records))
    permuted = _reference_v2_train_selection([records[index] for index in permutation], revision)
    assert [row["semantic_sha256"] for row in expected] == [
        row["semantic_sha256"] for row in permuted
    ]
    actual_counts = {}
    for family in available:
        actual_counts[family] = {
            bucket: sum(
                row["family"] == family and row["bucket_id"] == bucket for row in expected
            )
            for bucket in available[family]
        }
    assert actual_counts == policy["expected_train_bucket_counts_from_sealed_manifest"]

    broken = copy.deepcopy(revision)
    broken["state_selection"]["primary_order"][1] = "missing_manifest_field"
    with pytest.raises(KeyError, match="missing_manifest_field"):
        _reference_v2_train_selection(records, broken)

    reselector = _load_script(
        "s05_formal_v2_reselector_test", "scripts/steiner/reselect_s05_formal_v2.py"
    )
    actual = reselector.select_v2_train_records(records, revision)
    assert [row["semantic_sha256"] for row in actual] == [
        row["semantic_sha256"] for row in expected
    ]
    duplicate = copy.deepcopy(records)
    duplicate.append(copy.deepcopy(records[0]))
    with pytest.raises(ValueError, match="duplicate semantic_sha256"):
        reselector.select_v2_train_records(duplicate, revision)


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


def test_formal_v3_is_nonexecuting_and_freezes_the_five_v2_models():
    assert file_sha256(FORMAL_V3_CONFIG_PATH) == FORMAL_V3_CONFIG_SHA256
    assert file_sha256(FORMAL_V2_RESULT_AUDIT_PATH) == FORMAL_V2_RESULT_AUDIT_SHA256
    config = load_yaml_mapping(FORMAL_V3_CONFIG_PATH)
    audit = json.loads(FORMAL_V2_RESULT_AUDIT_PATH.read_text(encoding="utf-8"))

    assert config["revision_id"] == "s05-formal-v3-confirmatory-gate"
    assert config["status"] == "preregistered_protocol_pending_gpt_preexecution_audit"
    assert config["execution_authorized"] is False
    assert config["purpose"] == "fresh_confirmatory_validation_only"
    assert audit["verdict"] == "PASS"
    assert audit["verdict_meaning"] == "formal_v2_fail_and_stop_before_s06_accepted"
    assert audit["formal_v2_result"] == "FAIL_RETAINED"
    assert audit["s05_gate"] == "FAIL"
    assert audit["s06_authorized"] is False
    assert audit["formal_v3_preregistration_authorized"] is True
    assert audit["formal_v3_execution_authorized"] is False

    models = config["frozen_models"]
    assert models["retrain_models"] is False
    assert models["training_data_changed"] is False
    assert models["loss_or_hyperparameters_changed"] is False
    assert models["seeds"] == [101, 202, 303, 404, 505]
    assert models["source_selection_manifest_sha256"] == (
        "35221abeeaae507623d0175d895b5ff807d0b7e5400fce494e615fbefe8cd10e"
    )
    expected_models = {
        101: (
            "66585c122d8ccd2279e559490d7abd2ed9456a7478ec657c507170911c33e7c3",
            "e5dd22a24615599c19d9e956131ed0cc7c5206440f6f908e2d6d57c2a06d1a31",
        ),
        202: (
            "b7f271c36243499e1510a5db19156afb3f607c13abf059c4f8349d49d0efb48c",
            "1b78d2b35a86829eb6afca801b86955a9e006ff678f224cb79fc400d1b42884f",
        ),
        303: (
            "5895e92f283e3cadf119177077dd872d236fd91e0f13efe57a93c3b3865f2db1",
            "3e93fa2654dd39baa6454a5acfb057c31c8a889acdf1567de8b9461d0acdb0ce",
        ),
        404: (
            "fa29b03c2e12b8f6d1960405224e26d34f0658f2d718b93a355d8b32f894b6fc",
            "55c38818b37e519ef43cec598737d82db8334857fc0b35073801fb423eeb4624",
        ),
        505: (
            "f33d145a8651e04e502039409d460d67dda4aca1415fcbc83554f6572546d76c",
            "559406c7c7b4656f8d0c3a80d8eb1d7aea6bdd065087187bf483e7bc2a015900",
        ),
    }
    for seed, (manifest_sha256, model_sha256) in expected_models.items():
        frozen = models["checkpoints"][seed]
        assert frozen == {
            "manifest_sha256": manifest_sha256,
            "model_sha256": model_sha256,
        }
        checkpoint = REPO / f"checkpoints/steiner/s05/s05-teacher-il-formal-v2/seed-{seed}"
        if checkpoint.exists():
            assert file_sha256(checkpoint / "manifest.json") == manifest_sha256
            assert file_sha256(checkpoint / "model.pt") == model_sha256


def test_formal_v3_candidate_pool_is_fresh_deterministic_and_scale_limited():
    assert file_sha256(FORMAL_V3_CANDIDATES_PATH) == FORMAL_V3_CANDIDATES_SHA256
    config = load_yaml_mapping(FORMAL_V3_CONFIG_PATH)
    manifest = json.loads(FORMAL_V3_CANDIDATES_PATH.read_text(encoding="utf-8"))
    assert config["fresh_candidate_pool"]["manifest_sha256"] == (
        FORMAL_V3_CANDIDATES_SHA256
    )
    assert manifest["candidate_pool_id"] == (
        "s05-formal-v3-confirmatory-gate-candidates-v1"
    )
    assert manifest["selection_may_use_model_outputs"] is False
    assert manifest["test_and_final_accessed"] is False

    records = manifest["candidate_graphs"]
    assert len(records) == 80
    assert len({row["generator_seed"] for row in records}) == 80
    assert len({row["graph_sha256"] for row in records}) == 80
    assert len({row["instance_id"] for row in records}) == 80
    assert {row["family"] for row in records} == set(SYNTHETIC_FAMILIES)

    prior = load_s05_formal_config(require_activation=False)
    prior_seeds = {
        int(seed)
        for group in prior["instance_groups"]
        for seed in group["generator_seeds"]
    }
    prior_graph_hashes = {
        generate_graph(
            GeneratorConfig(
                family=group["family"],
                n_nodes=int(group["n_nodes"]),
                n_terminals=int(group["n_terminals"]),
                seed=int(seed),
            )
        ).graph_sha256
        for group in prior["instance_groups"]
        for seed in group["generator_seeds"]
    }
    for pilot_name in (
        "s05_teacher_il_pilot_v1.yml",
        "s05_teacher_il_pilot_v2.yml",
        "s05_teacher_il_pilot_v3.yml",
    ):
        pilot = load_yaml_mapping(REPO / "configs/steiner/experiments" / pilot_name)
        prior_seeds.update(
            int(item["generator_seed"]) for item in pilot["pilot_instances"]
        )
        prior_graph_hashes.update(
            generate_graph(
                GeneratorConfig(
                    family=item["family"],
                    n_nodes=int(item["n_nodes"]),
                    n_terminals=int(item["n_terminals"]),
                    seed=int(item["generator_seed"]),
                )
            ).graph_sha256
            for item in pilot["pilot_instances"]
        )
    assert not ({row["generator_seed"] for row in records} & prior_seeds)
    assert not ({row["graph_sha256"] for row in records} & prior_graph_hashes)

    expected_buckets = {
        "sparse_erdos_renyi": ["small-low", "medium-mid"] * 8,
        "random_geometric": ["small-low"] * 16,
        "grid_with_holes": ["medium-mid"] * 16,
        "community_block": ["medium-mid"] * 16,
        "bridge_bottleneck": ["medium-mid"] * 16,
    }
    for family in SYNTHETIC_FAMILIES:
        family_rows = [row for row in records if row["family"] == family]
        assert [row["candidate_rank"] for row in family_rows] == list(range(16))
        assert [row["bucket_id"] for row in family_rows] == expected_buckets[family]
        for row in family_rows:
            assert split_for_synthetic_seed(row["generator_seed"]) == "validation_iid"
            regenerated = generate_graph(
                GeneratorConfig(
                    family=row["family"],
                    n_nodes=row["n_nodes"],
                    n_terminals=row["n_terminals"],
                    seed=row["generator_seed"],
                )
            )
            assert regenerated.graph_sha256 == row["graph_sha256"]


def test_formal_v3_selection_and_pre_model_barrier_fail_closed():
    config = load_yaml_mapping(FORMAL_V3_CONFIG_PATH)
    pool = config["fresh_candidate_pool"]
    lineage = config["lineage_selection"]
    states = config["state_selection"]
    barrier = config["pre_model_access_gate"]
    statistics = config["statistics"]

    assert pool["total_candidate_graphs"] == 80
    assert pool["candidates_per_family"] == 16
    assert pool["total_teacher_tasks"] == 240
    assert pool["expected_max_states"] == 3840
    assert lineage["eligibility_uses_model_outputs"] is False
    assert lineage["eligibility_rule"][
        "minimum_semantic_unique_valid_states_across_three_teacher_tasks"
    ] == 12
    assert lineage["selected_lineages_per_family"] == 6
    assert lineage["selected_lineages_total"] == 30
    assert states["selected_states_per_family"] == 64
    assert states["selected_states_total"] == 320
    assert barrier["unique_gate_lineages"] == 30
    assert set(barrier["family_lineage_counts"].values()) == {6}
    assert set(barrier["selected_state_counts"].values()) == {64}
    assert barrier["every_lineage_minimum_selected_states"] == 1
    assert barrier["require_zero_cross_role_lineage"] is True
    assert barrier["require_zero_prior_s05_lineage"] is True
    assert barrier["require_zero_old_v2_gate_lineage"] is True
    assert barrier["on_any_failed_check"] == "stop_without_loading_any_checkpoint"
    assert statistics["exact_bootstrap_lineages"] == 30
    assert statistics["exact_lineages_per_family"] == 6
    assert config["failure_policy"]["allow_model_or_regret_based_lineage_selection"] is False
    assert config["failure_policy"]["allow_gate_threshold_relaxation"] is False
    assert config["failure_policy"]["allow_model_retraining_or_checkpoint_reselection"] is False
    assert config["failure_policy"]["test_and_final_access_prohibited"] is True
    assert config["failure_policy"]["s06_authorized"] is False

    def select_ranks(valid_unique_by_rank):
        eligible = sorted(
            rank for rank, count in valid_unique_by_rank.items() if count >= 12
        )
        if len(eligible) < 6:
            raise ValueError("fewer than six eligible lineages")
        return eligible[:6]

    counts = {rank: (11 if rank in {0, 3} else 12 + rank) for rank in range(16)}
    assert select_ranks(counts) == [1, 2, 4, 5, 6, 7]
    permuted = dict(reversed(list(counts.items())))
    assert select_ranks(permuted) == [1, 2, 4, 5, 6, 7]
    assert sum(counts[rank] for rank in select_ranks(counts)) >= 72
    with pytest.raises(ValueError, match="fewer than six"):
        select_ranks({rank: (12 if rank < 5 else 11) for rank in range(16)})


def _v3_selector_records(candidates):
    records = []
    counter = 0
    for candidate in candidates:
        valid = int(candidate["candidate_rank"]) < 8
        for state_index in range(12):
            counter += 1
            records.append({
                "path": f"states/{counter}.npz",
                "file_sha256": f"{counter:064x}",
                "semantic_sha256": f"{counter + 100000:064x}",
                "task_id": f"{candidate['instance_id']}--teacher1001",
                "role": "validation_gate_candidate",
                "split": "validation_iid",
                "family": candidate["family"],
                "bucket_id": candidate["bucket_id"],
                "candidate_rank": candidate["candidate_rank"],
                "generator_seed": candidate["generator_seed"],
                "graph_sha256": candidate["graph_sha256"],
                "teacher_seed": 1001 + state_index % 3,
                "state_index": state_index,
                "state_valid": valid,
                "all_tie": False,
                "candidate_count": 2,
            })
    return records


def test_formal_v3_production_loader_activation_and_exact_task_matrix():
    activation = load_v3_activation()
    assert activation["verdict"] == "PASS"
    assert activation["execution_authorized"] is True
    assert activation["model_retraining_authorized"] is False
    assert activation["s06_authorized"] is False
    config = load_v3_protocol(require_activation=True)
    candidates = load_v3_candidates()
    plans = expand_v3_tasks(config)
    assert len(candidates) == 80
    assert len(plans) == 240
    assert {plan.task.teacher_seed for plan in plans} == {1001, 1002, 1003}
    assert len({plan.task.task_id for plan in plans}) == 240
    assert all(plan.task.split == "validation_iid" for plan in plans)
    assert tuple(config["frozen_models"]["seeds"]) == V3_MODEL_SEEDS


def test_formal_v3_production_selector_is_permutation_stable_and_fail_closed():
    selector = _load_script(
        "s05_formal_v3_selector_test", "scripts/steiner/select_s05_formal_v3_gate.py"
    )
    config = load_v3_protocol(require_activation=False)
    candidates = load_v3_candidates()
    records = _v3_selector_records(candidates)
    lineages, states = selector.select_v3_gate_records(config, candidates, records)
    permutation = np.random.default_rng(20260907).permutation(len(records))
    permuted_lineages, permuted_states = selector.select_v3_gate_records(
        config, tuple(reversed(candidates)), [records[index] for index in permutation]
    )
    assert [row["graph_sha256"] for row in lineages] == [
        row["graph_sha256"] for row in permuted_lineages
    ]
    assert [row["semantic_sha256"] for row in states] == [
        row["semantic_sha256"] for row in permuted_states
    ]
    assert len(lineages) == 30 and len(states) == 320
    assert {
        family: sum(row["family"] == family for row in lineages)
        for family in SYNTHETIC_FAMILIES
    } == {family: 6 for family in SYNTHETIC_FAMILIES}
    assert {
        family: sum(row["family"] == family for row in states)
        for family in SYNTHETIC_FAMILIES
    } == {family: 64 for family in SYNTHETIC_FAMILIES}

    broken = copy.deepcopy(records)
    target = SYNTHETIC_FAMILIES[0]
    disabled = {
        candidate["graph_sha256"] for candidate in candidates
        if candidate["family"] == target and candidate["candidate_rank"] in {5, 6, 7}
    }
    for row in broken:
        if row["graph_sha256"] in disabled:
            row["state_valid"] = False
    with pytest.raises(ValueError, match="fewer than six eligible"):
        selector.select_v3_gate_records(config, candidates, broken)


def test_formal_v3_evaluator_cannot_load_checkpoint_when_barrier_fails(monkeypatch):
    evaluator = _load_script(
        "s05_formal_v3_evaluator_test", "scripts/steiner/evaluate_s05_formal_v3.py"
    )
    checkpoint_calls = []

    def fail_barrier():
        raise StrictConfigError("selection seal missing")

    monkeypatch.setattr(evaluator, "parse_args", lambda: SimpleNamespace(model_seed=101))
    monkeypatch.setattr(evaluator, "load_v3_selection_seal", fail_barrier)
    monkeypatch.setattr(
        evaluator,
        "load_checkpoint_bundle",
        lambda *args, **kwargs: checkpoint_calls.append((args, kwargs)),
    )
    with pytest.raises(StrictConfigError, match="selection seal missing"):
        evaluator.main()
    assert checkpoint_calls == []
