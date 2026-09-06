"""Frozen S05 formal protocol loading, activation, and task expansion."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from ..config import StrictConfigError, load_yaml_mapping
from ..data.generate import SYNTHETIC_FAMILIES
from ..data.split import split_for_synthetic_seed
from .teacher_data import TeacherTask, file_sha256, s05_config_sha256


REPO = Path(__file__).resolve().parents[3]
FORMAL_CONFIG_PATH = REPO / "configs/steiner/experiments/s05_teacher_il_formal_v1.yml"
FORMAL_PROTOCOL_PATH = REPO / "docs/steiner/phases/S05/S05_FORMAL_PROTOCOL.md"
FORMAL_AUDIT_RECORD_PATH = REPO / "docs/steiner/audits/S05_FORMAL_PROTOCOL_AUDIT_RECORD.json"
FORMAL_CONFIG_FILE_SHA256 = "c843f76d69c07b1c8a8093ab6f1426656084b2de2f4e9e0d777f1af32cd4c811"
FORMAL_PROTOCOL_FILE_SHA256 = "6b121bb215ad9ad5444ce2c8328dc7049b3abdfd0a7bf191fec888441ec572c9"
FORMAL_PROTOCOL_CONTENT_HEAD = "d1717a7ecb6043efd71678175a92325ac9ff4208"
FORMAL_EXPERIMENT_ID = "s05-teacher-il-formal-v1"
FORMAL_TRAINING_SEEDS = (101, 202, 303, 404, 505)
FORMAL_TEACHER_SEEDS = (1001, 1002, 1003)
FORMAL_ROLES = ("train", "validation_select", "validation_gate")


def _require_keys(raw: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise StrictConfigError(
            f"{label} fields mismatch: missing={missing}, unknown={unknown}"
        )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise StrictConfigError(f"{label} must be a string-keyed mapping")
    return dict(value)


def load_formal_audit_record(path: Path | str = FORMAL_AUDIT_RECORD_PATH) -> dict[str, Any]:
    record_path = Path(path)
    try:
        raw = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StrictConfigError(f"S05 formal audit record is unreadable: {error}") from error
    if not isinstance(raw, dict):
        raise StrictConfigError("S05 formal audit record must be a mapping")
    _require_keys(
        raw,
        {
            "schema_version", "stage", "audit_kind", "verdict", "authorization",
            "verdict_source", "recorded_at_utc", "protocol_content_head",
            "protocol_yaml_sha256", "protocol_explanation_sha256",
            "blocking_findings", "full_s05_gate_evaluated", "s06_authorized",
        },
        "S05 formal audit record",
    )
    expected = {
        "schema_version": 1,
        "stage": "S05",
        "audit_kind": "formal_protocol_pre_execution",
        "verdict": "PASS",
        "authorization": "formal_implementation_and_execution",
        "verdict_source": "user_supplied_external_gpt_audit",
        "protocol_content_head": FORMAL_PROTOCOL_CONTENT_HEAD,
        "protocol_yaml_sha256": FORMAL_CONFIG_FILE_SHA256,
        "protocol_explanation_sha256": FORMAL_PROTOCOL_FILE_SHA256,
        "blocking_findings": [],
        "full_s05_gate_evaluated": False,
        "s06_authorized": False,
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise StrictConfigError(f"S05 formal audit record {key} is not authorized")
    if file_sha256(FORMAL_CONFIG_PATH) != FORMAL_CONFIG_FILE_SHA256:
        raise StrictConfigError("frozen S05 formal YAML checksum changed")
    if file_sha256(FORMAL_PROTOCOL_PATH) != FORMAL_PROTOCOL_FILE_SHA256:
        raise StrictConfigError("frozen S05 formal protocol checksum changed")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FORMAL_PROTOCOL_CONTENT_HEAD, "HEAD"],
        cwd=REPO,
        check=False,
    )
    if ancestor.returncode != 0:
        raise StrictConfigError("S05 formal protocol content head is outside current history")
    return raw


@dataclass(frozen=True)
class FormalTaskPlan:
    task: TeacherTask
    role: str


def load_s05_formal_config(
    path: Path | str = FORMAL_CONFIG_PATH, *, require_activation: bool = True
) -> dict[str, Any]:
    config_path = Path(path)
    if file_sha256(config_path) != FORMAL_CONFIG_FILE_SHA256:
        raise StrictConfigError("S05 formal config is not the audited byte-exact YAML")
    raw = load_yaml_mapping(config_path)
    _require_keys(
        raw,
        {
            "schema_version", "stage", "experiment_id", "status",
            "execution_authorized", "required_protocol_audit_outcome",
            "activation_method", "protocol_document", "solver_stack_id",
            "protocol_id", "formulation_id", "bipartite_schema_id", "model_id",
            "model_config", "split_policy", "required_s04_audited_tag", "limits",
            "controls", "teacher", "instance_groups", "state_selection", "training",
            "statistics", "artifacts", "gate", "failure_policy",
        },
        "S05 formal config",
    )
    identity = {
        "schema_version": 1,
        "stage": "S05",
        "experiment_id": FORMAL_EXPERIMENT_ID,
        "status": "preregistered_protocol_pending_gpt_audit",
        "execution_authorized": False,
        "required_protocol_audit_outcome": "PASS",
        "activation_method": "separate_committed_gpt_pass_record_referencing_protocol_sha256",
        "protocol_document": "docs/steiner/phases/S05/S05_FORMAL_PROTOCOL.md",
        "solver_stack_id": "scip804-ecole081-pyscipopt430",
        "protocol_id": "P1",
        "formulation_id": "rooted_mcf_v1",
        "bipartite_schema_id": "milp_bipartite_v1",
        "model_id": "b0_milp_gcnn_v1",
        "model_config": "configs/steiner/models/b0_milp_gcnn_v1.yml",
        "split_policy": "configs/steiner/splits/split_policy_v1.yml",
        "required_s04_audited_tag": "steiner-s04-audited-v2",
    }
    for key, expected in identity.items():
        if raw.get(key) != expected:
            raise StrictConfigError(f"S05 formal {key} differs from the audited protocol")
    if raw["limits"] != {"time_seconds": 600, "nodes": 200000, "memory_mb": 8192, "threads": 1}:
        raise StrictConfigError("S05 formal limits differ from P1")
    teacher = _mapping(raw["teacher"], "formal teacher")
    if tuple(teacher.get("collection_seeds", ())) != FORMAL_TEACHER_SEEDS:
        raise StrictConfigError("S05 formal teacher seeds changed")
    if teacher.get("workers") != 6 or teacher.get("max_states_per_task") != 16:
        raise StrictConfigError("S05 formal worker/state budget changed")
    training = _mapping(raw["training"], "formal training")
    if tuple(training.get("formal_seeds", ())) != FORMAL_TRAINING_SEEDS:
        raise StrictConfigError("S05 formal training seeds changed")
    if any(
        (
            training.get("train_states") != 640,
            training.get("epochs") != 40,
            training.get("gpu_model") != "Tesla V100-SXM2-32GB",
            training.get("max_concurrent_training_jobs") != 2,
            training.get("deterministic_algorithms") is not True,
            training.get("cublas_workspace_config") != ":4096:8",
            training.get("gate_evaluation_role") != "validation_gate",
            training.get("representative_checkpoint_seed") != 202,
        )
    ):
        raise StrictConfigError("S05 formal training contract changed")
    selection = _mapping(raw["state_selection"], "formal state selection")
    if selection.get("selected_state_counts") != {
        "train": 640, "validation_select": 160, "validation_gate": 320
    }:
        raise StrictConfigError("S05 formal selected state counts changed")
    gate = _mapping(raw["gate"], "formal gate")
    required_gate = {
        "expected_base_graphs": 105,
        "expected_teacher_tasks": 315,
        "expected_max_states": 5040,
        "min_teacher_valid_state_fraction": 0.60,
        "max_teacher_all_tie_valid_state_fraction": 0.40,
        "required_action_mapping_rate": 1.0,
        "max_seed_regret_coefficient_of_variation": 0.15,
        "require_checkpoint_reload_max_abs_error": 0.0,
        "prohibit_test_and_final_access": True,
    }
    for key, expected in required_gate.items():
        if gate.get(key) != expected:
            raise StrictConfigError(f"S05 formal Gate {key} changed")
    if require_activation:
        load_formal_audit_record()
    plans = expand_formal_tasks(raw)
    if len(plans) != 315:
        raise StrictConfigError("S05 formal task expansion is not 315")
    return raw


def expand_formal_tasks(config: Mapping[str, Any]) -> tuple[FormalTaskPlan, ...]:
    groups = config.get("instance_groups")
    if not isinstance(groups, list):
        raise StrictConfigError("S05 formal instance_groups must be a list")
    plans: list[FormalTaskPlan] = []
    seen_graph_seeds: set[int] = set()
    role_counts = {role: 0 for role in FORMAL_ROLES}
    role_family_counts = {role: {family: 0 for family in SYNTHETIC_FAMILIES} for role in FORMAL_ROLES}
    for index, value in enumerate(groups):
        item = _mapping(value, f"instance_groups[{index}]")
        _require_keys(
            item,
            {"role", "split", "family", "bucket_id", "n_nodes", "n_terminals", "generator_seeds"},
            f"instance_groups[{index}]",
        )
        role, split, family = str(item["role"]), str(item["split"]), str(item["family"])
        if role not in FORMAL_ROLES or family not in SYNTHETIC_FAMILIES:
            raise StrictConfigError("S05 formal role/family is unsupported")
        if split != ("train" if role == "train" else "validation_iid"):
            raise StrictConfigError("S05 formal role does not match its registered split")
        expected_bucket = {
            "small-low": (48, 5), "medium-mid": (96, 19), "large-high": (160, 48)
        }.get(str(item["bucket_id"]))
        if expected_bucket != (int(item["n_nodes"]), int(item["n_terminals"])):
            raise StrictConfigError("S05 formal bucket dimensions changed")
        seeds = item["generator_seeds"]
        if not isinstance(seeds, list) or not seeds:
            raise StrictConfigError("S05 formal generator seed group is empty")
        for raw_seed in seeds:
            seed = int(raw_seed)
            if seed in seen_graph_seeds:
                raise StrictConfigError("duplicate S05 formal generator seed")
            seen_graph_seeds.add(seed)
            if split_for_synthetic_seed(seed, policy_path=config["split_policy"]) != split:
                raise StrictConfigError("S05 formal generator seed crosses split")
            role_counts[role] += 1
            role_family_counts[role][family] += 1
            for teacher_seed in FORMAL_TEACHER_SEEDS:
                instance_id = f"{split}--{family}--s{seed}"
                plans.append(
                    FormalTaskPlan(
                        role=role,
                        task=TeacherTask(
                            task_id=f"{instance_id}--teacher{teacher_seed}",
                            split=split,
                            family=family,
                            bucket_id=str(item["bucket_id"]),
                            n_nodes=int(item["n_nodes"]),
                            n_terminals=int(item["n_terminals"]),
                            generator_seed=seed,
                            teacher_seed=teacher_seed,
                        ),
                    )
                )
    if role_counts != {"train": 60, "validation_select": 15, "validation_gate": 30}:
        raise StrictConfigError(f"S05 formal role graph counts changed: {role_counts}")
    expected_per_family = {"train": 12, "validation_select": 3, "validation_gate": 6}
    for role, count in expected_per_family.items():
        if set(role_family_counts[role].values()) != {count}:
            raise StrictConfigError(f"S05 formal family balance changed for {role}")
    if len({plan.task.task_id for plan in plans}) != len(plans):
        raise StrictConfigError("S05 formal task IDs are not unique")
    return tuple(plans)


def formal_config_sha256(config: Mapping[str, Any]) -> str:
    return s05_config_sha256(config)
