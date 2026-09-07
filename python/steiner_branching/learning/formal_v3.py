"""Audited S05 formal-v3 protocol, candidate, and pre-model barrier loading."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from ..config import StrictConfigError, load_yaml_mapping
from ..data.generate import GeneratorConfig, SYNTHETIC_FAMILIES, generate_graph
from ..data.split import split_for_synthetic_seed
from .teacher_data import TeacherTask, file_sha256


REPO = Path(__file__).resolve().parents[3]
V3_CONFIG_PATH = (
    REPO / "configs/steiner/experiments/s05_teacher_il_formal_v3_confirmatory_gate.yml"
)
V3_CONFIG_FILE_SHA256 = "99e75a4d4fa69f805232c637d4fc0ae750fcfccb57e9979b561f422591006242"
V3_EXPLANATION_PATH = REPO / "docs/steiner/phases/S05/S05_FORMAL_V3_CONFIRMATORY_GATE.md"
V3_EXPLANATION_FILE_SHA256 = "c61b58d23f4de6bb2709aa9c40ccab11d6db4507a6bd1c9d2e544eb68595d873"
V3_CANDIDATES_PATH = REPO / "configs/steiner/experiments/s05_formal_v3_candidate_graphs.json"
V3_CANDIDATES_FILE_SHA256 = "e65fc9a03fd683277570befe13b11f4b15ea981ee427568d56aeeccdd3847b56"
V3_ACTIVATION_PATH = REPO / "docs/steiner/audits/S05_FORMAL_V3_ACTIVATION_RECORD.json"
V3_PROTOCOL_CONTENT_HEAD = "8d7accd2a948935174d3113f8787e58dae7936b3"
V3_EXPERIMENT_ID = "s05-teacher-il-formal-v3-confirmatory"
V3_TEACHER_SEEDS = (1001, 1002, 1003)
V3_MODEL_SEEDS = (101, 202, 303, 404, 505)
V3_RAW_ROOT = REPO / "results/steiner/raw/s05/s05-teacher-il-formal-v3-confirmatory"
V3_TEACHER_MANIFEST = V3_RAW_ROOT / "teacher_manifest.json"
V3_SELECTED_MANIFEST = V3_RAW_ROOT / "selected_manifest.json"
V3_SELECTION_SEAL = REPO / "docs/steiner/phases/S05/S05_FORMAL_V3_SELECTION_SEAL.json"


@dataclass(frozen=True)
class V3TaskPlan:
    task: TeacherTask
    candidate_rank: int
    graph_sha256: str


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StrictConfigError(f"{label} is unreadable: {error}") from error
    if not isinstance(raw, dict):
        raise StrictConfigError(f"{label} must be a mapping")
    return raw


def _require_ancestor(commit: str, label: str) -> None:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPO,
        check=False,
    ).returncode != 0:
        raise StrictConfigError(f"{label} is outside current Git history")


def load_v3_activation(path: Path | str = V3_ACTIVATION_PATH) -> dict[str, Any]:
    raw = _read_json(Path(path), "S05 formal-v3 activation")
    expected = {
        "schema_version": 1,
        "stage": "S05",
        "audit_kind": "formal_v3_confirmatory_gate_pre_execution_activation",
        "verdict": "PASS",
        "blocking_findings": [],
        "execution_authorized": True,
        "authorization": "implementation_tests_and_fresh_240_task_teacher_collection",
        "verdict_source": "user_supplied_external_gpt_audit",
        "source_text_sha256": "f2d066a8b8dda7e6344f3c7340953586abccc4a07110e9629d576d31ba1885f8",
        "audited_base": "55833c54d02f3e5bcbc4bd2e19d95f6f3f019293",
        "audited_protocol_content_head": V3_PROTOCOL_CONTENT_HEAD,
        "protocol_yaml_sha256": V3_CONFIG_FILE_SHA256,
        "protocol_explanation_sha256": V3_EXPLANATION_FILE_SHA256,
        "candidate_manifest_sha256": V3_CANDIDATES_FILE_SHA256,
        "formal_v2_result": "FAIL_RETAINED",
        "s05_status_at_activation": "FAIL",
        "fresh_teacher_collection_authorized": True,
        "lineage_selection_authorized_after_teacher_pass": True,
        "checkpoint_loading_authorized_before_pre_model_barrier": False,
        "frozen_checkpoint_evaluation_authorized_after_pre_model_barrier": True,
        "model_retraining_authorized": False,
        "full_s05_gate_evaluated": False,
        "s05_audited_tag_authorized": False,
        "s06_authorized": False,
        "test_and_final_access_authorized": False,
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise StrictConfigError(f"S05 formal-v3 activation {key} changed")
    for path_value, digest, label in (
        (V3_CONFIG_PATH, V3_CONFIG_FILE_SHA256, "YAML"),
        (V3_EXPLANATION_PATH, V3_EXPLANATION_FILE_SHA256, "explanation"),
        (V3_CANDIDATES_PATH, V3_CANDIDATES_FILE_SHA256, "candidate manifest"),
    ):
        if file_sha256(path_value) != digest:
            raise StrictConfigError(f"frozen formal-v3 {label} checksum changed")
    _require_ancestor(V3_PROTOCOL_CONTENT_HEAD, "S05 formal-v3 audited content head")
    return raw


def load_v3_protocol(*, require_activation: bool = True) -> dict[str, Any]:
    if file_sha256(V3_CONFIG_PATH) != V3_CONFIG_FILE_SHA256:
        raise StrictConfigError("S05 formal-v3 YAML is not byte-exact")
    raw = load_yaml_mapping(V3_CONFIG_PATH)
    expected = {
        "schema_version": 1,
        "stage": "S05",
        "revision_id": "s05-formal-v3-confirmatory-gate",
        "status": "preregistered_protocol_pending_gpt_preexecution_audit",
        "execution_authorized": False,
        "required_audit_outcome": "PASS",
        "purpose": "fresh_confirmatory_validation_only",
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise StrictConfigError(f"S05 formal-v3 protocol {key} changed")
    if raw["frozen_models"]["retrain_models"] is not False:
        raise StrictConfigError("formal-v3 may not retrain models")
    if tuple(raw["frozen_models"]["seeds"]) != V3_MODEL_SEEDS:
        raise StrictConfigError("formal-v3 frozen model seeds changed")
    if raw["fresh_candidate_pool"]["manifest_sha256"] != V3_CANDIDATES_FILE_SHA256:
        raise StrictConfigError("formal-v3 candidate manifest identity changed")
    if raw["pre_model_access_gate"]["on_any_failed_check"] != (
        "stop_without_loading_any_checkpoint"
    ):
        raise StrictConfigError("formal-v3 pre-model failure policy changed")
    if raw["failure_policy"]["test_and_final_access_prohibited"] is not True:
        raise StrictConfigError("formal-v3 test/final prohibition changed")
    if require_activation:
        load_v3_activation()
    return raw


def load_v3_candidates() -> tuple[dict[str, Any], ...]:
    if file_sha256(V3_CANDIDATES_PATH) != V3_CANDIDATES_FILE_SHA256:
        raise StrictConfigError("S05 formal-v3 candidate manifest checksum changed")
    raw = _read_json(V3_CANDIDATES_PATH, "S05 formal-v3 candidate manifest")
    if any((
        raw.get("schema_version") != 1,
        raw.get("stage") != "S05",
        raw.get("candidate_pool_id") != "s05-formal-v3-confirmatory-gate-candidates-v1",
        raw.get("split") != "validation_iid",
        raw.get("role") != "validation_gate_candidate",
        raw.get("selection_may_use_model_outputs") is not False,
        raw.get("test_and_final_accessed") is not False,
    )):
        raise StrictConfigError("S05 formal-v3 candidate manifest identity changed")
    records = raw.get("candidate_graphs")
    if not isinstance(records, list) or len(records) != 80:
        raise StrictConfigError("formal-v3 candidate manifest must contain 80 graphs")
    required = {
        "candidate_rank", "family", "bucket_id", "n_nodes", "n_terminals",
        "generator_seed", "instance_id", "graph_sha256",
    }
    seen_seeds: set[int] = set()
    seen_graphs: set[str] = set()
    by_family: dict[str, list[dict[str, Any]]] = {
        family: [] for family in SYNTHETIC_FAMILIES
    }
    for index, value in enumerate(records):
        if not isinstance(value, dict) or set(value) != required:
            raise StrictConfigError(f"formal-v3 candidate schema changed at {index}")
        family = str(value["family"])
        seed = int(value["generator_seed"])
        if family not in by_family or seed in seen_seeds:
            raise StrictConfigError("formal-v3 candidate family/seed is invalid")
        if split_for_synthetic_seed(seed) != "validation_iid":
            raise StrictConfigError("formal-v3 candidate crosses the validation split")
        graph = generate_graph(GeneratorConfig(
            family=family,
            n_nodes=int(value["n_nodes"]),
            n_terminals=int(value["n_terminals"]),
            seed=seed,
        ))
        if graph.graph_sha256 != value["graph_sha256"]:
            raise StrictConfigError("formal-v3 generated graph hash changed")
        if graph.graph_sha256 in seen_graphs:
            raise StrictConfigError("formal-v3 candidate graph hash is duplicated")
        if value["instance_id"] != f"validation_iid--{family}--s{seed}":
            raise StrictConfigError("formal-v3 candidate instance ID changed")
        seen_seeds.add(seed)
        seen_graphs.add(graph.graph_sha256)
        by_family[family].append(dict(value))
    for family, values in by_family.items():
        if [int(item["candidate_rank"]) for item in values] != list(range(16)):
            raise StrictConfigError(f"formal-v3 candidate ranks changed for {family}")
    return tuple(dict(item) for item in records)


def expand_v3_tasks(config: Mapping[str, Any]) -> tuple[V3TaskPlan, ...]:
    candidates = load_v3_candidates()
    teacher_seeds = tuple(config["solver_and_teacher"]["collection_seeds"])
    if teacher_seeds != V3_TEACHER_SEEDS:
        raise StrictConfigError("formal-v3 teacher seeds changed")
    plans = tuple(
        V3TaskPlan(
            task=TeacherTask(
                task_id=f"{item['instance_id']}--teacher{teacher_seed}",
                split="validation_iid",
                family=str(item["family"]),
                bucket_id=str(item["bucket_id"]),
                n_nodes=int(item["n_nodes"]),
                n_terminals=int(item["n_terminals"]),
                generator_seed=int(item["generator_seed"]),
                teacher_seed=int(teacher_seed),
            ),
            candidate_rank=int(item["candidate_rank"]),
            graph_sha256=str(item["graph_sha256"]),
        )
        for item in candidates
        for teacher_seed in teacher_seeds
    )
    if len(plans) != 240 or len({plan.task.task_id for plan in plans}) != 240:
        raise StrictConfigError("formal-v3 task expansion is not 240 unique tasks")
    return plans


def prior_s05_graph_hashes() -> tuple[set[str], set[str]]:
    """Return all prior registered S05 graph hashes and the old v2 Gate subset."""
    from .formal_protocol import load_s05_formal_config

    formal = load_s05_formal_config(require_activation=False)
    all_prior: set[str] = set()
    old_gate: set[str] = set()
    for group in formal["instance_groups"]:
        for seed in group["generator_seeds"]:
            digest = generate_graph(GeneratorConfig(
                family=str(group["family"]),
                n_nodes=int(group["n_nodes"]),
                n_terminals=int(group["n_terminals"]),
                seed=int(seed),
            )).graph_sha256
            all_prior.add(digest)
            if group["role"] == "validation_gate":
                old_gate.add(digest)
    for name in (
        "s05_teacher_il_pilot_v1.yml",
        "s05_teacher_il_pilot_v2.yml",
        "s05_teacher_il_pilot_v3.yml",
    ):
        pilot = load_yaml_mapping(REPO / "configs/steiner/experiments" / name)
        for item in pilot["pilot_instances"]:
            all_prior.add(generate_graph(GeneratorConfig(
                family=str(item["family"]),
                n_nodes=int(item["n_nodes"]),
                n_terminals=int(item["n_terminals"]),
                seed=int(item["generator_seed"]),
            )).graph_sha256)
    return all_prior, old_gate


def v3_teacher_runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(config["solver_and_teacher"])
    teacher = {
        key: source[key]
        for key in (
            "max_states_per_task", "iteration_limit_per_candidate", "candidate_limit",
            "infeasible_gain_cap", "tie_relative_tolerance", "trajectory",
        )
    }
    teacher["idempotent"] = False
    return {
        "limits": dict(source["limits"]),
        "controls": dict(source["controls"]),
        "teacher": teacher,
    }


def require_v3_implementation_identity(expected_head: str) -> str:
    _require_ancestor(expected_head, "formal-v3 implementation head")
    protected = [
        "python/steiner_branching", "scripts/steiner", "configs/steiner", "tests/steiner"
    ]
    if subprocess.run(
        ["git", "diff", "--quiet", expected_head, "HEAD", "--", *protected],
        cwd=REPO,
        check=False,
    ).returncode != 0:
        raise StrictConfigError("formal-v3 executable inputs changed after sealing")
    if subprocess.run(
        ["git", "diff", "--quiet", "--", *protected], cwd=REPO, check=False
    ).returncode != 0:
        raise StrictConfigError("formal-v3 executable inputs have uncommitted changes")
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *protected],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if untracked:
        raise StrictConfigError("formal-v3 executable inputs include untracked files")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        capture_output=True, check=True,
    ).stdout.strip()


def load_v3_selection_seal(path: Path | str = V3_SELECTION_SEAL) -> dict[str, Any]:
    seal_path = Path(path)
    raw = _read_json(seal_path, "S05 formal-v3 selection seal")
    expected = {
        "schema_version": 1,
        "stage": "S05",
        "experiment_id": V3_EXPERIMENT_ID,
        "protocol_yaml_sha256": V3_CONFIG_FILE_SHA256,
        "candidate_manifest_sha256": V3_CANDIDATES_FILE_SHA256,
        "teacher_manifest_path": str(V3_TEACHER_MANIFEST.relative_to(REPO)),
        "selected_manifest_path": str(V3_SELECTED_MANIFEST.relative_to(REPO)),
        "selected_lineages": 30,
        "selected_states": 320,
        "family_lineage_counts": {family: 6 for family in SYNTHETIC_FAMILIES},
        "family_state_counts": {family: 64 for family in SYNTHETIC_FAMILIES},
        "pre_model_access_gate": "PASS",
        "checkpoint_loaded": False,
        "formal_gate_evaluated": False,
        "s06_authorized": False,
        "test_and_final_accessed": False,
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise StrictConfigError(f"formal-v3 selection seal {key} changed")
    if file_sha256(V3_TEACHER_MANIFEST) != raw.get("teacher_manifest_sha256"):
        raise StrictConfigError("formal-v3 teacher manifest checksum changed")
    if file_sha256(V3_SELECTED_MANIFEST) != raw.get("selected_manifest_sha256"):
        raise StrictConfigError("formal-v3 selected manifest checksum changed")
    try:
        relative = seal_path.resolve().relative_to(REPO.resolve())
    except ValueError as error:
        raise StrictConfigError("formal-v3 selection seal must be inside the repository") from error
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative.as_posix()}"], cwd=REPO,
        capture_output=True, check=False,
    )
    if committed.returncode != 0 or committed.stdout != seal_path.read_bytes():
        raise StrictConfigError(
            "formal-v3 selection seal must be committed byte-exact before model access"
        )
    require_v3_implementation_identity(str(raw.get("implementation_run_head")))
    return raw
