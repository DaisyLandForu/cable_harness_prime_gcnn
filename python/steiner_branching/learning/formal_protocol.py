"""Frozen S05 formal protocol loading, activation, and task expansion."""

from __future__ import annotations

from dataclasses import dataclass
import copy
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

FORMAL_V2_CONFIG_PATH = (
    REPO / "configs/steiner/experiments/s05_teacher_il_formal_v2_selection_remediation.yml"
)
FORMAL_V2_PROTOCOL_PATH = REPO / "docs/steiner/phases/S05/S05_FORMAL_V2_SELECTION_REMEDIATION.md"
FORMAL_V2_B1_PATH = REPO / "docs/steiner/phases/S05/S05_FORMAL_V2_B1_ORDERING_REMEDIATION.md"
FORMAL_V2_ACTIVATION_PATH = REPO / "docs/steiner/audits/S05_FORMAL_V2_ACTIVATION_RECORD.json"
FORMAL_V2_CONFIG_FILE_SHA256 = "f101c038610d391b76c589fc81d08298160bac4883ed169f71953c7159137cbf"
FORMAL_V2_PROTOCOL_FILE_SHA256 = "626568f03e0c622d05616af009b8507cdb832135953c52f06181de908633fff8"
FORMAL_V2_B1_FILE_SHA256 = "0746b0d08cfe10e98980d6764636db9e0259fec1a5d4b5d0a1d7f695ca32289e"
FORMAL_V2_CONTENT_HEAD = "05ffd02a7d491dcbc75dc4700c6fc0e5bd4a925b"
FORMAL_V2_EXPERIMENT_ID = "s05-teacher-il-formal-v2"
FORMAL_V2_SOURCE_MANIFEST = REPO / "results/steiner/raw/s05/s05-teacher-il-formal-v1/manifest.json"
FORMAL_V2_SOURCE_MANIFEST_SHA256 = "2bb3b4875d173571df5e4fb7e9c1e1d3f4615708308e891f4e4ccdbaac4c149c"
FORMAL_V2_MANIFEST = REPO / "results/steiner/raw/s05/s05-teacher-il-formal-v2/manifest.json"
FORMAL_V2_SELECTION_SEAL = REPO / "docs/steiner/phases/S05/S05_FORMAL_V2_SELECTION_SEAL.json"


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


def load_formal_v2_activation(
    path: Path | str = FORMAL_V2_ACTIVATION_PATH,
) -> dict[str, Any]:
    record_path = Path(path)
    try:
        raw = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StrictConfigError(f"S05 formal-v2 activation is unreadable: {error}") from error
    if not isinstance(raw, dict):
        raise StrictConfigError("S05 formal-v2 activation must be a mapping")
    _require_keys(
        raw,
        {
            "schema_version", "stage", "audit_kind", "verdict", "b1_status",
            "execution_authorized", "authorization", "verdict_source",
            "recorded_at_utc", "audited_remediation_content_head",
            "focused_remediation_base", "revision_yaml_sha256",
            "revision_explanation_sha256", "b1_remediation_explanation_sha256",
            "blocking_findings", "formal_v1_result", "teacher_tasks_rerun_authorized",
            "formal_training", "full_s05_gate_evaluated", "s06_authorized",
            "test_and_final_access_authorized",
        },
        "S05 formal-v2 activation",
    )
    expected = {
        "schema_version": 1,
        "stage": "S05",
        "audit_kind": "formal_v2_b1_focused_reaudit_activation",
        "verdict": "PASS",
        "b1_status": "CLOSED",
        "execution_authorized": True,
        "authorization": "v2_reselection_and_five_independent_one_v100_seed_jobs",
        "verdict_source": "user_supplied_external_gpt_reaudit",
        "audited_remediation_content_head": FORMAL_V2_CONTENT_HEAD,
        "focused_remediation_base": "0a6b5ffc06ab49f97e38680ac493bcd7f1577f19",
        "revision_yaml_sha256": FORMAL_V2_CONFIG_FILE_SHA256,
        "revision_explanation_sha256": FORMAL_V2_PROTOCOL_FILE_SHA256,
        "b1_remediation_explanation_sha256": FORMAL_V2_B1_FILE_SHA256,
        "blocking_findings": [],
        "formal_v1_result": "FAIL_RETAINED",
        "teacher_tasks_rerun_authorized": False,
        "formal_training": "NOT_RUN_AT_ACTIVATION",
        "full_s05_gate_evaluated": False,
        "s06_authorized": False,
        "test_and_final_access_authorized": False,
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise StrictConfigError(f"S05 formal-v2 activation {key} is not authorized")
    for candidate, digest, label in (
        (FORMAL_V2_CONFIG_PATH, FORMAL_V2_CONFIG_FILE_SHA256, "YAML"),
        (FORMAL_V2_PROTOCOL_PATH, FORMAL_V2_PROTOCOL_FILE_SHA256, "explanation"),
        (FORMAL_V2_B1_PATH, FORMAL_V2_B1_FILE_SHA256, "B1 explanation"),
    ):
        if file_sha256(candidate) != digest:
            raise StrictConfigError(f"frozen S05 formal-v2 {label} checksum changed")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", FORMAL_V2_CONTENT_HEAD, "HEAD"],
        cwd=REPO, check=False,
    ).returncode != 0:
        raise StrictConfigError("S05 formal-v2 audited head is outside current history")
    return raw


def load_s05_formal_v2_config(*, require_activation: bool = True) -> dict[str, Any]:
    """Return the audited base protocol with only the accepted v2 overrides applied."""
    if file_sha256(FORMAL_V2_CONFIG_PATH) != FORMAL_V2_CONFIG_FILE_SHA256:
        raise StrictConfigError("S05 formal-v2 config is not the audited byte-exact YAML")
    revision = load_yaml_mapping(FORMAL_V2_CONFIG_PATH)
    if revision.get("revision_id") != "s05-formal-v2-selection-remediation":
        raise StrictConfigError("S05 formal-v2 revision identity changed")
    if revision.get("execution_authorized") is not False:
        raise StrictConfigError("audited S05 formal-v2 YAML must remain immutable/unactivated")
    if revision.get("state_selection", {}).get("selected_state_counts") != {
        "train": 640, "validation_select": 160, "validation_gate": 320
    }:
        raise StrictConfigError("S05 formal-v2 selected-state counts changed")
    execution = _mapping(revision.get("training_execution"), "formal-v2 training execution")
    if tuple(execution.get("formal_seeds", ())) != FORMAL_TRAINING_SEEDS:
        raise StrictConfigError("S05 formal-v2 training seeds changed")
    if any((
        execution.get("max_concurrent_training_jobs") != 5,
        execution.get("gpu_count_per_job") != 1,
        execution.get("distributed_training") is not False,
        execution.get("shared_optimizer_or_model_state") is not False,
    )):
        raise StrictConfigError("S05 formal-v2 independent-job semantics changed")
    unchanged = _mapping(revision.get("unchanged_contract"), "formal-v2 unchanged contract")
    if unchanged.get("test_and_final_access_prohibited") is not True:
        raise StrictConfigError("S05 formal-v2 test/final prohibition changed")
    if require_activation:
        load_formal_v2_activation()
    base = copy.deepcopy(load_s05_formal_config(require_activation=True))
    base["experiment_id"] = FORMAL_V2_EXPERIMENT_ID
    base["training"]["max_concurrent_training_jobs"] = 5
    base["training"]["checkpoint_root"] = execution["checkpoint_root"]
    base["artifacts"]["raw_root"] = str(FORMAL_V2_MANIFEST.parent.relative_to(REPO))
    base["artifacts"]["report_root"] = execution["report_root"]
    base["artifacts"]["checkpoint_root"] = execution["checkpoint_root"]
    return base


def load_formal_v2_selection_seal(
    path: Path | str = FORMAL_V2_SELECTION_SEAL,
) -> dict[str, Any]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StrictConfigError(f"S05 formal-v2 selection seal is unreadable: {error}") from error
    expected = {
        "schema_version": 1,
        "stage": "S05",
        "experiment_id": FORMAL_V2_EXPERIMENT_ID,
        "source_manifest_sha256": FORMAL_V2_SOURCE_MANIFEST_SHA256,
        "selected_states": 640,
        "family_counts": {
            "sparse_erdos_renyi": 128, "random_geometric": 128,
            "grid_with_holes": 128, "community_block": 128,
            "bridge_bottleneck": 128,
        },
        "full_s05_gate_evaluated": False,
        "s06_authorized": False,
    }
    if not isinstance(raw, dict):
        raise StrictConfigError("S05 formal-v2 selection seal must be a mapping")
    for key, value in expected.items():
        if raw.get(key) != value:
            raise StrictConfigError(f"S05 formal-v2 selection seal {key} changed")
    if file_sha256(FORMAL_V2_MANIFEST) != raw.get("selection_manifest_sha256"):
        raise StrictConfigError("S05 formal-v2 selection manifest checksum changed")
    return raw


def require_formal_v2_implementation_identity(expected_head: str) -> str:
    """Allow metadata-only descendants while rejecting changes to executable inputs."""
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected_head, current],
        cwd=REPO, check=False,
    ).returncode != 0:
        raise StrictConfigError("formal-v2 implementation head is outside current history")
    protected = [
        "python/steiner_branching", "scripts/steiner",
        "configs/steiner", "tests/steiner",
    ]
    if subprocess.run(
        ["git", "diff", "--quiet", expected_head, current, "--", *protected],
        cwd=REPO, check=False,
    ).returncode != 0:
        raise StrictConfigError("formal-v2 executable inputs changed after selection sealing")
    if subprocess.run(
        ["git", "diff", "--quiet", "--", *protected], cwd=REPO, check=False,
    ).returncode != 0:
        raise StrictConfigError("formal-v2 executable inputs have uncommitted changes")
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *protected],
        cwd=REPO, text=True, capture_output=True, check=True,
    ).stdout.strip()
    if untracked:
        raise StrictConfigError("formal-v2 executable inputs include untracked files")
    return current


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
