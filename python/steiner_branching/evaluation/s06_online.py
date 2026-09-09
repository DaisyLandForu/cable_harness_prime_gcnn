"""Fail-closed S06 online branching protocol, runner, and paired aggregation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import re
import subprocess
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from ..config import StrictConfigError, load_yaml_mapping
from ..contracts import canonical_json
from ..data.generate import GeneratorConfig, SYNTHETIC_FAMILIES, generate_graph
from ..data.split import split_for_synthetic_seed
from ..learning.imitation import load_checkpoint_bundle, normalize_state
from ..learning.teacher_data import file_sha256
from ..milp.mcf import build_mcf
from ..milp.naming import edge_id_from_scip_variable_name, original_variable_name
from ..milp.validate import check_selected_edges
from ..models.milp_gcnn import score_state
from ..solver.bipartite_observation import SteinerNodeBipartite, with_legal_edge_actions


REPO = Path(__file__).resolve().parents[3]
S06_CONFIG_PATH = REPO / "configs/steiner/experiments/s06_il_online_v1.yml"
S06_INSTANCES_PATH = REPO / "configs/steiner/experiments/s06_online_instances_v1.json"
S06_ACTIVATION_PATH = REPO / "docs/steiner/audits/S06_PREEXECUTION_ACTIVATION_RECORD.json"
S06_CONFIG_FILE_SHA256 = "e7f7e9060c25fa038a2749ca43afe6a2769c3652e93697612b8804269750f827"
S06_INSTANCES_FILE_SHA256 = "b50f8048d8ab27a1fa2168e69ef179cbfc490116399180f2bb4a963bf04b292b"
S05_AUDITED_TAG = "steiner-s05-audited-v3"
S05_AUDITED_HEAD = "6cf7acab57525a744233ed3fdfd463f00fcd470c"
EXPECTED_STACK_ID = "scip804-ecole081-pyscipopt430"
MAIN_METHODS = ("b0_il", "random_candidate", "mostinf", "relpscost", "scip_default")
WEAK_BASELINES = ("random_candidate", "mostinf")
SOLVER_SEEDS = (0, 1, 2, 3, 4)
FULLSTRONG = "fullstrong"
S06_SHARD_COUNT = 6
S06_WORKERS_PER_SHARD = 6
S06_PAR2_PENALTY_SECONDS = 1200.0


class S06PolicyFailure(RuntimeError):
    """A fail-closed learned/random policy error with a stable evidence class."""

    def __init__(self, failure_class: str, message: str) -> None:
        super().__init__(message)
        self.failure_class = failure_class


def _require_keys(raw: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise StrictConfigError(f"{label} fields mismatch: missing={missing}, unknown={unknown}")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise StrictConfigError(f"{label} must be a string-keyed mapping")
    return dict(value)


@dataclass(frozen=True)
class S06Instance:
    bucket_id: str
    candidate_rank: int
    family: str
    generator_seed: int
    graph_sha256: str
    instance_id: str
    n_nodes: int
    n_terminals: int

    def generator_config(self) -> GeneratorConfig:
        return GeneratorConfig(self.family, self.n_nodes, self.n_terminals, self.generator_seed)


@dataclass(frozen=True)
class S06Task:
    task_id: str
    instance: S06Instance
    solver_seed: int
    method: str
    gate_relevant: bool
    trace: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["instance"] = asdict(self.instance)
        return value


def load_s06_instances(path: Path | str = S06_INSTANCES_PATH) -> tuple[S06Instance, ...]:
    candidate = Path(path)
    if candidate == S06_INSTANCES_PATH and file_sha256(candidate) != S06_INSTANCES_FILE_SHA256:
        raise StrictConfigError("S06 instance manifest is not byte-exact")
    try:
        raw = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StrictConfigError(f"S06 instance manifest is unreadable: {error}") from error
    if not isinstance(raw, dict):
        raise StrictConfigError("S06 instance manifest must be a mapping")
    _require_keys(
        raw,
        {
            "schema_version", "stage", "manifest_id", "role", "selection_policy",
            "source_selected_manifest", "source_selected_manifest_sha256",
            "selection_used_s06_online_outcomes", "test_and_final_accessed", "instances",
        },
        "S06 instance manifest",
    )
    expected_header = {
        "schema_version": 1,
        "stage": "S06",
        "manifest_id": "s06-online-validation-iid-v1",
        "role": "validation_iid",
        "selection_policy": "reuse_s05_v3_pre_model_teacher_validity_selection",
        "source_selected_manifest_sha256": (
            "0a95f47337338ab58891056f773a99467da2b327b714d1a2cd4ee5e96c3b6981"
        ),
        "selection_used_s06_online_outcomes": False,
        "test_and_final_accessed": False,
    }
    for key, value in expected_header.items():
        if raw.get(key) != value:
            raise StrictConfigError(f"S06 instance manifest {key} changed")
    entries = raw["instances"]
    if not isinstance(entries, list) or len(entries) != 30:
        raise StrictConfigError("S06 instance manifest must contain exactly 30 lineages")
    expected_fields = {
        "bucket_id", "candidate_rank", "family", "generator_seed", "graph_sha256",
        "instance_id", "n_nodes", "n_terminals",
    }
    instances: list[S06Instance] = []
    for index, value in enumerate(entries):
        item = _mapping(value, f"S06 instance {index}")
        _require_keys(item, expected_fields, f"S06 instance {index}")
        instance = S06Instance(
            bucket_id=str(item["bucket_id"]),
            candidate_rank=int(item["candidate_rank"]),
            family=str(item["family"]),
            generator_seed=int(item["generator_seed"]),
            graph_sha256=str(item["graph_sha256"]),
            instance_id=str(item["instance_id"]),
            n_nodes=int(item["n_nodes"]),
            n_terminals=int(item["n_terminals"]),
        )
        if split_for_synthetic_seed(instance.generator_seed) != "validation_iid":
            raise StrictConfigError(f"S06 instance {instance.instance_id} is outside validation_iid")
        graph = generate_graph(instance.generator_config())
        if graph.graph_sha256 != instance.graph_sha256:
            raise StrictConfigError(f"S06 instance {instance.instance_id} graph checksum changed")
        instances.append(instance)
    if len({item.instance_id for item in instances}) != 30:
        raise StrictConfigError("S06 instance IDs are not unique")
    if len({item.graph_sha256 for item in instances}) != 30:
        raise StrictConfigError("S06 graph lineages are not unique")
    if Counter(item.family for item in instances) != Counter({family: 6 for family in SYNTHETIC_FAMILIES}):
        raise StrictConfigError("S06 must contain exactly six lineages per family")
    if tuple(dict.fromkeys(item.family for item in instances)) != SYNTHETIC_FAMILIES:
        raise StrictConfigError("S06 family ordering changed")
    source_path = REPO / str(raw["source_selected_manifest"])
    if not source_path.is_file() or file_sha256(source_path) != raw["source_selected_manifest_sha256"]:
        raise StrictConfigError("S06 source S05 pre-model selected manifest is missing or changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("checkpoint_loaded") is not False or source.get("pre_model_access_gate", {}).get("status") != "PASS":
        raise StrictConfigError("S06 source lineage selection did not precede model access")
    if source.get("lineage_selection") != entries:
        raise StrictConfigError("S06 lineages differ from the S05 pre-model selection")
    return tuple(instances)


def _verify_s05_anchor(config: Mapping[str, Any]) -> None:
    upstream = config["upstream"]
    audit_path = REPO / upstream["s05_result_audit_record"]
    if file_sha256(audit_path) != upstream["s05_result_audit_record_sha256"]:
        raise StrictConfigError("S05 result-audit record checksum changed")
    record = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        record.get("verdict") != "PASS"
        or record.get("s05_scientific_status") != "PASS_VIA_FORMAL_V3_CONFIRMATORY_GATE"
        or record.get("s06_implementation_and_handoff_authorized") is not True
        or record.get("test_and_final_access_authorized") is not False
    ):
        raise StrictConfigError("S05 result audit does not authorize S06")
    peeled = subprocess.run(
        ["git", "rev-parse", f"{S05_AUDITED_TAG}^{{}}"], cwd=REPO, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    if peeled != S05_AUDITED_HEAD:
        raise StrictConfigError("S05 audited tag target changed")


def load_s06_activation(path: Path | str = S06_ACTIVATION_PATH) -> dict[str, Any]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StrictConfigError(f"S06 pre-execution activation is unavailable: {error}") from error
    if not isinstance(raw, dict):
        raise StrictConfigError("S06 pre-execution activation must be a mapping")
    expected_fields = {
        "schema_version", "stage", "audit_kind", "verdict", "blocking_findings",
        "execution_authorized", "verdict_source", "recorded_at_utc",
        "audited_branch", "audited_content_head", "protocol_yaml_sha256",
        "instance_manifest_sha256", "s05_audited_head", "s05_audited_tag",
        "container_runtime_fingerprint",
        "formal_results_at_activation", "s07_authorized", "test_and_final_access_authorized",
    }
    _require_keys(raw, expected_fields, "S06 pre-execution activation")
    expected = {
        "schema_version": 1,
        "stage": "S06",
        "audit_kind": "s06_online_protocol_and_implementation_preexecution_review",
        "verdict": "PASS",
        "blocking_findings": [],
        "execution_authorized": True,
        "audited_branch": "research/steiner-migration",
        "protocol_yaml_sha256": S06_CONFIG_FILE_SHA256,
        "instance_manifest_sha256": S06_INSTANCES_FILE_SHA256,
        "s05_audited_head": S05_AUDITED_HEAD,
        "s05_audited_tag": S05_AUDITED_TAG,
        "formal_results_at_activation": "NOT_RUN",
        "s07_authorized": False,
        "test_and_final_access_authorized": False,
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise StrictConfigError(f"S06 activation {key} is not authorized")
    head = str(raw["audited_content_head"])
    if raw.get("verdict_source") != "user_supplied_external_gpt_audit":
        raise StrictConfigError("S06 activation verdict source is not registered")
    if not isinstance(raw.get("recorded_at_utc"), str) or not raw["recorded_at_utc"].endswith("Z"):
        raise StrictConfigError("S06 activation timestamp is invalid")
    if len(head) != 40 or any(char not in "0123456789abcdef" for char in head):
        raise StrictConfigError("S06 activation audited_content_head is invalid")
    runtime_fingerprint = raw.get("container_runtime_fingerprint")
    if (
        not isinstance(runtime_fingerprint, str) or len(runtime_fingerprint) != 64
        or any(char not in "0123456789abcdef" for char in runtime_fingerprint)
    ):
        raise StrictConfigError("S06 activation container runtime fingerprint is invalid")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", head, "HEAD"], cwd=REPO, check=False,
    ).returncode != 0:
        raise StrictConfigError("S06 audited content head is outside current history")
    protected = ["python/steiner_branching", "scripts/steiner", "configs/steiner", "tests/steiner"]
    if subprocess.run(
        ["git", "diff", "--quiet", head, "HEAD", "--", *protected], cwd=REPO, check=False,
    ).returncode != 0:
        raise StrictConfigError("S06 executable inputs changed after the audited content head")
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *protected], cwd=REPO, check=False,
    ).returncode != 0:
        raise StrictConfigError("S06 executable inputs have uncommitted changes")
    return raw


def load_s06_config(
    path: Path | str = S06_CONFIG_PATH, *, require_activation: bool = True
) -> dict[str, Any]:
    candidate = Path(path)
    if candidate == S06_CONFIG_PATH and file_sha256(candidate) != S06_CONFIG_FILE_SHA256:
        raise StrictConfigError("S06 protocol YAML is not byte-exact")
    raw = load_yaml_mapping(candidate)
    _require_keys(
        raw,
        {
            "schema_version", "stage", "experiment_id", "status", "execution_authorized",
            "upstream", "problem", "validation", "methods", "profile", "checkpoint",
            "inference", "pdi", "metrics", "statistics", "trace", "parallelism", "gate",
            "failure_policy", "prohibitions",
        },
        "S06 config",
    )
    if (
        raw["schema_version"] != 1 or raw["stage"] != "S06"
        or raw["experiment_id"] != "s06-il-online-v1"
        or raw["execution_authorized"] is not False
    ):
        raise StrictConfigError("S06 config identity or immutable authorization flag changed")
    problem = _mapping(raw["problem"], "S06 problem")
    if problem != {
        "problem": "SPG", "formulation_id": "rooted_mcf_v1",
        "bipartite_schema_id": "milp_bipartite_v1", "model_id": "b0_milp_gcnn_v1",
        "solver_stack_id": EXPECTED_STACK_ID,
    }:
        raise StrictConfigError("S06 problem/stack identity changed")
    validation = _mapping(raw["validation"], "S06 validation")
    if (
        validation.get("split") != "validation_iid"
        or tuple(validation.get("solver_seeds", ())) != SOLVER_SEEDS
        or validation.get("lineages") != 30
        or validation.get("lineages_per_family") != 6
        or validation.get("test_or_final_instances_allowed") is not False
        or validation.get("source_selection_must_precede_model_access") is not True
    ):
        raise StrictConfigError("S06 validation identity/split/seeds changed")
    if (REPO / validation["instance_manifest"]).resolve() != S06_INSTANCES_PATH.resolve():
        raise StrictConfigError("S06 instance manifest path changed")
    methods = _mapping(raw["methods"], "S06 methods")
    if (
        tuple(methods.get("main", ())) != MAIN_METHODS
        or tuple(methods.get("weak_gate_baselines", ())) != WEAK_BASELINES
        or methods.get("main_tasks") != 750
        or methods.get("strong_diagnostic_tasks") != 5
        or methods.get("learned_method") != "b0_il"
    ):
        raise StrictConfigError("S06 method matrix changed")
    strong = _mapping(methods.get("strong_branching"), "S06 strong branching")
    if strong != {
        "method": FULLSTRONG, "gate_relevant": False, "solver_seeds": [0],
        "selection": "first_registered_lineage_per_family", "required_lineages": 5,
    }:
        raise StrictConfigError("S06 strong-branch diagnostic subset changed")
    profile = _mapping(raw["profile"], "S06 profile")
    expected_profile = {
        "protocol_id": "P1", "protocol_name": "controlled-branching-v1",
        "time_limit_seconds": 600, "node_limit": 200000, "memory_limit_mb": 8192,
        "threads": 1, "presolving_maxrounds": 0, "separating_maxrounds": 0,
        "separating_maxroundsroot": 0, "heuristics": "off", "restart_limit": 0,
        "node_selector": "estimate", "propagation": "default",
    }
    if profile != expected_profile:
        raise StrictConfigError("S06 P1 profile changed")
    checkpoint = _mapping(raw["checkpoint"], "S06 checkpoint")
    for key, value in {
        "representative_training_seed": 202,
        "manifest_sha256": "b7f271c36243499e1510a5db19156afb3f607c13abf059c4f8349d49d0efb48c",
        "model_payload_sha256": "1b78d2b35a86829eb6afca801b86955a9e006ff678f224cb79fc400d1b42884f",
        "normalization_sha256": "b0dcdca751525d9144cdaa4009bb99f542820addfe47f1bd90c64b4bee96a5b6",
        "model_config_sha256": "9360f5893103adcf3b12baa3f8b2d1d5e0549791c7da68060e43542257ed1fc2",
        "retraining_allowed": False,
    }.items():
        if checkpoint.get(key) != value:
            raise StrictConfigError(f"S06 checkpoint {key} changed")
    manifest_path = REPO / checkpoint["manifest"]
    model_config_path = REPO / checkpoint["model_config"]
    if file_sha256(manifest_path) != checkpoint["manifest_sha256"]:
        raise StrictConfigError("S06 checkpoint manifest checksum changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("training_seed") != 202
        or manifest.get("checkpoint_sha256") != checkpoint["model_payload_sha256"]
        or manifest.get("normalization_sha256") != checkpoint["normalization_sha256"]
        or file_sha256(model_config_path) != checkpoint["model_config_sha256"]
    ):
        raise StrictConfigError("S06 checkpoint payload identity changed")
    if file_sha256(manifest_path.parent / manifest["checkpoint"]) != checkpoint["model_payload_sha256"]:
        raise StrictConfigError("S06 checkpoint model payload bytes changed")
    if file_sha256(manifest_path.parent / manifest["normalization"]) != checkpoint["normalization_sha256"]:
        raise StrictConfigError("S06 normalization payload bytes changed")
    inference = _mapping(raw["inference"], "S06 inference")
    if (
        inference.get("device") != "cpu" or inference.get("torch_threads") != 1
        or inference.get("deterministic_algorithms") is not True
        or inference.get("invalid_action_fallback") != "forbidden"
        or inference.get("random_policy_seed") != 6001
    ):
        raise StrictConfigError("S06 inference contract changed")
    statistics = _mapping(raw["statistics"], "S06 statistics")
    if (
        statistics.get("bootstrap_unit") != "instance"
        or statistics.get("bootstrap_replicates") != 10000
        or statistics.get("bootstrap_seed") != 20260902
        or statistics.get("confidence_level") != 0.95
        or statistics.get("percentile_method") != "numpy_linear"
        or statistics.get("paired_effect_direction") != "baseline_minus_b0_il"
    ):
        raise StrictConfigError("S06 paired statistics changed")
    parallelism = _mapping(raw["parallelism"], "S06 parallelism")
    if parallelism != {
        "execution_mode": "six_independent_lineage_shards",
        "shard_count": S06_SHARD_COUNT,
        "shard_ids": list(range(S06_SHARD_COUNT)),
        "assignment_unit": "base_graph_lineage",
        "assignment_rule": "instance_manifest_index_modulo_shard_count",
        "lineages_per_shard": 5,
        "families_per_shard": 5,
        "main_tasks_per_shard": 125,
        "workers_per_shard": S06_WORKERS_PER_SHARD,
        "maximum_total_workers": 36,
        "requested_resources_per_shard": {
            "cpu_cores": 8, "memory_gib": 96, "gpu_cards": 0,
        },
        "registered_runtime": {
            "minimum_effective_cpu_cores": 8,
            "minimum_memory_bytes": 103079215104,
            "required_gpu_visible_count": 0,
            "environment_lock": "configs/steiner/environment.lock.yml",
            "environment_lock_sha256": "f70afe548f2b640a3c1375686ad8c8ef4dced63d0229c9fa4eb36e62f6d7628e",
            "container_runtime_fingerprint_source": "activation_record",
            "audited_executable_content_head_source": "activation_record",
        },
        "one_scip_thread_per_worker": True,
        "shared_artifact_root_required": True,
        "identical_container_and_resource_request_required": True,
        "gpu_required": False,
        "estimated_single_shard_worst_case_hours": 4,
        "estimated_matrix_worst_case_hours": 4,
    }:
        raise StrictConfigError("S06 six-shard execution contract changed")
    gate = _mapping(raw["gate"], "S06 gate")
    correctness = _mapping(gate.get("correctness"), "S06 correctness Gate")
    if any(correctness.get(key) != 0 for key in (
        "max_invalid_actions", "max_mapping_failures", "max_nan_scores",
        "max_unexpected_fallbacks", "max_solver_errors", "max_solution_validation_failures",
    )):
        raise StrictConfigError("S06 correctness Gate was lowered")
    if gate.get("required_main_tasks") != 750 or gate.get("require_complete_main_matrix") is not True:
        raise StrictConfigError("S06 completeness Gate changed")
    if any(raw["prohibitions"].get(key) is not True for key in (
        "model_retraining", "checkpoint_reselection_from_s06", "b1_or_rl_implementation",
        "test_and_final_access", "s07_before_s06_audit_pass",
    )):
        raise StrictConfigError("S06 prohibition changed")
    if raw != load_yaml_mapping(S06_CONFIG_PATH):
        raise StrictConfigError("S06 config differs from the complete frozen protocol")
    load_s06_instances()
    _verify_s05_anchor(raw)
    if require_activation:
        load_s06_activation()
    return raw


def config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(config)).encode("utf-8")).hexdigest()


def expand_s06_tasks(
    config: Mapping[str, Any], instances: Sequence[S06Instance]
) -> tuple[tuple[S06Task, ...], tuple[S06Task, ...]]:
    main = tuple(
        S06Task(
            task_id=f"main--{instance.instance_id}--solver{seed}--{method}",
            instance=instance,
            solver_seed=seed,
            method=method,
            gate_relevant=True,
        )
        for instance in instances
        for seed in SOLVER_SEEDS
        for method in MAIN_METHODS
    )
    first_by_family: dict[str, S06Instance] = {}
    for instance in instances:
        first_by_family.setdefault(instance.family, instance)
    strong = tuple(
        S06Task(
            task_id=f"strong--{first_by_family[family].instance_id}--solver0--{FULLSTRONG}",
            instance=first_by_family[family],
            solver_seed=0,
            method=FULLSTRONG,
            gate_relevant=False,
        )
        for family in SYNTHETIC_FAMILIES
    )
    if len(main) != config["methods"]["main_tasks"] or len(strong) != config["methods"]["strong_diagnostic_tasks"]:
        raise StrictConfigError("expanded S06 task counts do not match the frozen matrix")
    if len({task.task_id for task in main + strong}) != len(main) + len(strong):
        raise StrictConfigError("expanded S06 task IDs are not unique")
    return main, strong


def lineage_shard_assignments(
    instances: Sequence[S06Instance], *, shard_count: int = S06_SHARD_COUNT
) -> dict[str, int]:
    """Assign complete graph lineages by frozen manifest position.

    The committed manifest contains six consecutive lineages per family.  Taking
    its index modulo six therefore gives every shard one lineage from each of
    the five families while keeping every solver seed and method together.
    """
    if shard_count != S06_SHARD_COUNT:
        raise StrictConfigError(f"S06 shard_count must remain {S06_SHARD_COUNT}")
    if len(instances) != 30 or len({item.graph_sha256 for item in instances}) != 30:
        raise StrictConfigError("S06 sharding requires the 30 unique frozen lineages")
    assignments = {
        instance.graph_sha256: index % shard_count
        for index, instance in enumerate(instances)
    }
    for shard_index in range(shard_count):
        selected = [
            instance for instance in instances
            if assignments[instance.graph_sha256] == shard_index
        ]
        if len(selected) != 5:
            raise StrictConfigError(f"S06 shard {shard_index} must contain five lineages")
        if Counter(item.family for item in selected) != Counter({family: 1 for family in SYNTHETIC_FAMILIES}):
            raise StrictConfigError(
                f"S06 shard {shard_index} must contain one lineage from every family"
            )
    return assignments


def tasks_for_lineage_shard(
    tasks: Sequence[S06Task], instances: Sequence[S06Instance], shard_index: int,
    *, shard_count: int = S06_SHARD_COUNT,
) -> tuple[S06Task, ...]:
    """Return the canonical, disjoint task subset for one formal custom job."""
    if isinstance(shard_index, bool) or not isinstance(shard_index, int):
        raise StrictConfigError("S06 shard_index must be an integer")
    if not 0 <= shard_index < shard_count:
        raise StrictConfigError(f"S06 shard_index must be in [0, {shard_count})")
    assignments = lineage_shard_assignments(instances, shard_count=shard_count)
    unknown = sorted({task.instance.graph_sha256 for task in tasks} - set(assignments))
    if unknown:
        raise StrictConfigError(f"S06 task set contains unregistered lineages: {unknown}")
    selected = tuple(
        task for task in tasks
        if assignments[task.instance.graph_sha256] == shard_index
    )
    if len({task.task_id for task in selected}) != len(selected):
        raise StrictConfigError(f"S06 shard {shard_index} contains duplicate task IDs")
    return selected


def task_sha256(task: S06Task, protocol_file_sha256: str = S06_CONFIG_FILE_SHA256) -> str:
    return hashlib.sha256(
        canonical_json({"protocol_file_sha256": protocol_file_sha256, "task": task.to_dict()}).encode("utf-8")
    ).hexdigest()


def deterministic_random_action(
    candidates: Iterable[int], *, random_policy_seed: int, graph_sha256: str,
    solver_seed: int, decision_index: int,
) -> int:
    values = tuple(int(value) for value in candidates)
    if not values or len(values) != len(set(values)) or min(values) < 0:
        raise ValueError("random action requires non-empty unique non-negative candidates")
    ranked = []
    for candidate in values:
        payload = f"{random_policy_seed}|{graph_sha256}|{solver_seed}|{decision_index}|{candidate}"
        ranked.append((hashlib.sha256(payload.encode("ascii")).digest(), -candidate, candidate))
    return max(ranked)[2]


def deterministic_argmax(logits: np.ndarray, candidates: Iterable[int]) -> int:
    scores = np.asarray(logits, dtype=np.float64)
    values = np.asarray(tuple(int(value) for value in candidates), dtype=np.int64)
    if scores.ndim != 1 or values.ndim != 1 or scores.size != values.size or scores.size == 0:
        raise ValueError("logits and candidates must be aligned non-empty vectors")
    if not np.isfinite(scores).all() or len(set(map(int, values))) != values.size:
        raise FloatingPointError("non-finite logits or duplicate candidates")
    maximum = float(scores.max())
    return int(values[scores == maximum].min())


def _finite_or_none(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) and abs(number) < 1.0e19 else None


class TimedSteinerNodeBipartite(SteinerNodeBipartite):
    """Return the immutable observation together with extraction wall time."""

    def extract(self, model: Any, done: bool) -> tuple[Any, float]:
        started = time.perf_counter()
        state = super().extract(model, done)
        return state, time.perf_counter() - started


class SolveEventRecorder:
    """Low-overhead exact solve events; optional detailed tree trace on replays."""

    def __init__(self, *, trace: bool, maximum_branch_records: int) -> None:
        self.trace = bool(trace)
        self.maximum_branch_records = int(maximum_branch_records)
        self.first_incumbent_seconds: float | None = None
        self.root_lp_bound: float | None = None
        self.branch_decisions = 0
        self.branch_trace: list[dict[str, Any]] = []
        self._trace_by_node: dict[int, dict[str, Any]] = {}
        self.instrumentation_seconds = 0.0

    def install(self, model: Any) -> Any:
        from pyscipopt import Eventhdlr, SCIP_EVENTTYPE

        recorder = self

        class Handler(Eventhdlr):
            def eventinit(self) -> None:
                mask = (
                    SCIP_EVENTTYPE.BESTSOLFOUND
                    | SCIP_EVENTTYPE.FIRSTLPSOLVED
                    | SCIP_EVENTTYPE.NODEBRANCHED
                )
                if recorder.trace:
                    mask |= SCIP_EVENTTYPE.NODEFOCUSED
                self.model.catchEvent(mask, self)
                self._mask = mask
                self._caught = True

            def eventexit(self) -> None:
                # Ecole owns the copied SCIP instance.  Calling back through
                # PySCIPOpt while that instance is being destroyed can access
                # an expired weak model wrapper; SCIP frees the caught events
                # with the handler, so no explicit drop is required here.
                pass

            def detach(self, retained_model: Any) -> None:
                if getattr(self, "_caught", False):
                    retained_model.dropEvent(self._mask, self)
                    self._caught = False

            def eventexec(self, event: Any) -> None:
                started = time.perf_counter()
                try:
                    event_type = int(event.getType())
                    if event_type & int(SCIP_EVENTTYPE.BESTSOLFOUND):
                        if recorder.first_incumbent_seconds is None:
                            recorder.first_incumbent_seconds = float(self.model.getSolvingTime())
                    if event_type & int(SCIP_EVENTTYPE.FIRSTLPSOLVED):
                        node = self.model.getCurrentNode()
                        if node is not None and int(node.getDepth()) == 0 and recorder.root_lp_bound is None:
                            recorder.root_lp_bound = _finite_or_none(self.model.getLPObjVal())
                    if event_type & int(SCIP_EVENTTYPE.NODEBRANCHED):
                        recorder.branch_decisions += 1
                    if recorder.trace and event_type & int(SCIP_EVENTTYPE.NODEFOCUSED):
                        recorder._node_focused(event.getNode(), self.model)
                finally:
                    recorder.instrumentation_seconds += time.perf_counter() - started

        handler = Handler()
        model.includeEventhdlr(handler, "steiner_s06_metrics", "exact S06 metric events")
        return handler

    def _node_focused(self, node: Any, model: Any) -> None:
        if node is None:
            return
        parent = node.getParent()
        if parent is not None:
            parent_number = int(parent.getNumber())
            if parent_number not in self._trace_by_node and len(self.branch_trace) < self.maximum_branch_records:
                branchings = node.getParentBranchings()
                if branchings is not None and branchings[0]:
                    variable = branchings[0][0]
                    name = original_variable_name(variable.name)
                    try:
                        edge_id = edge_id_from_scip_variable_name(name)
                    except ValueError:
                        edge_id = None
                    row = {
                        "branch_index": len(self.branch_trace),
                        "node_number": parent_number,
                        "depth": int(parent.getDepth()),
                        "node_lower_bound": _finite_or_none(parent.getLowerbound()),
                        "global_dual_bound": _finite_or_none(model.getDualbound()),
                        "selected_variable": name,
                        "selected_edge_id": edge_id,
                        "processed_descendant_nodes": 0,
                    }
                    self.branch_trace.append(row)
                    self._trace_by_node[parent_number] = row
        cursor = node
        while cursor is not None:
            row = self._trace_by_node.get(int(cursor.getNumber()))
            if row is not None and cursor is not node:
                row["processed_descendant_nodes"] += 1
            cursor = cursor.getParent()


class SolveInformation:
    """Ecole information function retaining the PySCIPOpt wrapper for plugin lifetime."""

    def __init__(self, *, trace: bool, maximum_branch_records: int) -> None:
        self.recorder = SolveEventRecorder(trace=trace, maximum_branch_records=maximum_branch_records)
        self._pyscip_model: Any | None = None
        self._handler: Any | None = None

    def before_reset(self, model: Any) -> None:
        self._pyscip_model = model.as_pyscipopt()
        if self.recorder.trace:
            self._handler = self.recorder.install(self._pyscip_model)

    def extract(self, model: Any, done: bool) -> dict[str, Any]:
        del model
        if self._pyscip_model is None:
            raise RuntimeError("S06 metric model was not retained")
        # Configuring.reset() extracts information in SCIP_STAGE_PROBLEM (1),
        # where getCurrentNode/getDualbound are not legal C API calls.
        stage = int(self._pyscip_model.getStage())
        node = self._pyscip_model.getCurrentNode() if stage >= 9 else None
        return {
            "done": bool(done),
            "current_node_number": None if node is None else int(node.getNumber()),
            "current_depth": None if node is None else int(node.getDepth()),
            "current_dual_bound": (
                _finite_or_none(self._pyscip_model.getDualbound()) if stage >= 9 else None
            ),
        }



_TREE_RE = re.compile(r"^\s*nodes\s*:\s*\d+\s+\((\d+) internal,", re.MULTILINE)
_DEPTH_RE = re.compile(r"^\s*max depth\s*:\s*(\d+)", re.MULTILINE)
_ROOT_LP_RE = re.compile(r"^\s*First LP value\s*:\s*([+\-0-9.eE]+)", re.MULTILINE)
_FIRST_SOL_RE = re.compile(
    r"^\s*First Solution\s*:.*?after\s+\d+\s+nodes,\s*([0-9.eE+\-]+)\s+seconds,",
    re.MULTILINE,
)


def parse_scip_statistics(text: str) -> dict[str, Any]:
    """Parse exact SCIP 8.0.4 statistics fields absent from PySCIPOpt 4.3.0."""
    tree = _TREE_RE.search(text)
    depth = _DEPTH_RE.search(text)
    root = _ROOT_LP_RE.search(text)
    first = _FIRST_SOL_RE.search(text)
    if tree is None or depth is None:
        raise ValueError("SCIP statistics omit B&B tree counts")
    return {
        "branch_decisions": int(tree.group(1)),
        "maximum_depth": int(depth.group(1)),
        "root_lp_bound": None if root is None else _finite_or_none(root.group(1)),
        "exact_time_to_first_incumbent": None if first is None else float(first.group(1)),
    }


def read_scip_statistics(model: Any) -> dict[str, Any]:
    descriptor, name = tempfile.mkstemp(prefix="steiner-s06-statistics-", suffix=".txt")
    os.close(descriptor)
    path = Path(name)
    try:
        model.writeStatistics(str(path))
        return parse_scip_statistics(path.read_text(encoding="utf-8"))
    finally:
        path.unlink(missing_ok=True)


def configure_s06_p1(model: Any, config: Mapping[str, Any], task: S06Task) -> dict[str, Any]:
    from pyscipopt import SCIP_PARAMSETTING

    profile = config["profile"]
    model.setParam("limits/time", float(profile["time_limit_seconds"]))
    model.setParam("limits/nodes", int(profile["node_limit"]))
    model.setParam("limits/memory", float(profile["memory_limit_mb"]))
    model.setParam("parallel/minnthreads", 1)
    model.setParam("parallel/maxnthreads", 1)
    model.setParam("lp/threads", 1)
    model.setParam("randomization/randomseedshift", task.solver_seed)
    model.setParam("randomization/permutationseed", task.solver_seed)
    model.setParam("randomization/lpseed", task.solver_seed)
    model.setParam("presolving/maxrounds", 0)
    model.setParam("separating/maxrounds", 0)
    model.setParam("separating/maxroundsroot", 0)
    model.setParam("limits/restarts", 0)
    model.setHeuristics(SCIP_PARAMSETTING.OFF)
    model.setParam("nodeselection/estimate/stdpriority", 1_000_000)
    if task.method == "relpscost":
        model.setParam("branching/relpscost/priority", 900_000)
    elif task.method == "mostinf":
        model.setParam("branching/mostinf/priority", 900_000)
    elif task.method == FULLSTRONG:
        model.setParam("branching/fullstrong/priority", 900_000)
    elif task.method not in {"scip_default", "b0_il", "random_candidate"}:
        raise ValueError(f"unsupported S06 method: {task.method}")
    keys = (
        "limits/time", "limits/nodes", "limits/memory", "parallel/minnthreads",
        "parallel/maxnthreads", "lp/threads", "randomization/randomseedshift",
        "randomization/permutationseed", "randomization/lpseed", "presolving/maxrounds",
        "separating/maxrounds", "separating/maxroundsroot", "limits/restarts",
        "nodeselection/estimate/stdpriority", "branching/relpscost/priority",
        "branching/mostinf/priority", "branching/fullstrong/priority",
    )
    return {key: model.getParam(key) for key in keys}


def _reward_value(value: Any) -> float:
    if value is None:
        return 0.0
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise FloatingPointError("S06 PDI contribution is invalid")
    return number


def failed_task_result(task: S06Task, error: BaseException) -> dict[str, Any]:
    """Encode a terminal exception without inventing unavailable solve metrics."""
    failure_class = (
        error.failure_class if isinstance(error, S06PolicyFailure) else "solver_or_runtime_error"
    )
    correctness = {
        "invalid_action_count": int(failure_class == "invalid_action"),
        "mapping_failure_count": int(failure_class == "mapping_failure"),
        "nan_score_count": int(failure_class == "nan_score"),
        "unexpected_fallback_count": 0,
        "solution_validation_failure_count": 0,
    }
    return {
        "schema_version": 1,
        "task": task.to_dict(),
        "status": "solver_error",
        "solved": False,
        "classification": failure_class,
        "failure": {
            "failure_class": failure_class,
            "exception_type": type(error).__name__,
            "message": str(error),
            "par2_penalty_applied": True,
            "pdi_available": False,
        },
        "metrics": {
            "solve_wall_seconds": None,
            "scip_solve_seconds": None,
            "par2_seconds": S06_PAR2_PENALTY_SECONDS,
            "primal_dual_integral": None,
            "primal_bound": None,
            "dual_bound": None,
            "final_gap": None,
            "nodes": None,
            "lp_iterations": None,
            "exact_time_to_first_incumbent": None,
            "root_lp_bound": None,
            "root_gap_to_final_primal": None,
            "branch_decisions": None,
            "maximum_depth": None,
        },
        "overhead": None,
        "correctness": correctness,
        "solution_validation": None,
        "trace": {"enabled": task.trace, "unavailable_due_to_failure": True},
        "resources": {"peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0},
        "effective_parameters": None,
    }


def run_s06_task(task: S06Task, config: Mapping[str, Any]) -> dict[str, Any]:
    """Run one isolated task. Caller owns exception-to-failure shard conversion."""
    if os.environ.get("STEINER_SOLVER_STACK_ID") != EXPECTED_STACK_ID:
        raise RuntimeError("S06 tasks must enter through run_with_scip804.sh")
    import ecole

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    graph = generate_graph(task.instance.generator_config())
    if graph.graph_sha256 != task.instance.graph_sha256:
        raise RuntimeError("S06 runtime graph identity changed")
    setup_started = time.perf_counter()
    build = build_mcf(graph, configure_correctness_profile=False, hide_output=True)
    effective = configure_s06_p1(build.model, config, task)
    model = stats = checkpoint_manifest = None
    if task.method == "b0_il":
        checkpoint = config["checkpoint"]
        model, stats, checkpoint_manifest = load_checkpoint_bundle(
            REPO / checkpoint["manifest"], model_config_path=REPO / checkpoint["model_config"],
            device=torch.device("cpu"),
        )
    setup_seconds = time.perf_counter() - setup_started
    information = SolveInformation(
        trace=task.trace,
        maximum_branch_records=int(config["trace"]["maximum_branch_records"]),
    )
    upper_bound = float(sum(edge.cost for edge in graph.edges))
    pdi_reward = ecole.reward.PrimalDualIntegral(
        wall=False, bound_function=lambda unused_model: (upper_bound, 0.0)
    )
    ecole_model = ecole.scip.Model.from_pyscipopt(build.model)
    solve_started = time.perf_counter()
    extraction_seconds = 0.0
    normalization_seconds = 0.0
    inference_seconds = 0.0
    selection_seconds = 0.0
    policy_trace: list[dict[str, Any]] = []
    invalid_action_count = mapping_failure_count = nan_score_count = 0
    unexpected_fallback_count = 0
    pdi = 0.0

    if task.method in {"b0_il", "random_candidate"}:
        observation = TimedSteinerNodeBipartite()
        environment = ecole.environment.Branching(
            observation_function=observation, reward_function=pdi_reward,
            information_function=information, scip_params={},
        )
        timed_state, action_set, reward, done, info = environment.reset(ecole_model)
        pdi += _reward_value(reward)
        decision_index = 0
        while not done:
            state, elapsed = timed_state
            extraction_seconds += float(elapsed)
            try:
                state = with_legal_edge_actions(state, action_set, build.metadata)
            except Exception as error:
                mapping_failure_count += 1
                raise S06PolicyFailure("mapping_failure", str(error)) from error
            if state.candidate_count == 0:
                invalid_action_count += 1
                raise S06PolicyFailure(
                    "invalid_action",
                    "Ecole requested an S06 decision without legal edge candidates",
                )
            if task.method == "b0_il":
                if model is None or stats is None:
                    raise RuntimeError("S06 learned checkpoint was not loaded")
                started = time.perf_counter()
                normalized = normalize_state(state, stats)
                normalization_seconds += time.perf_counter() - started
                started = time.perf_counter()
                with torch.inference_mode():
                    logits = score_state(model, normalized, device="cpu").detach().cpu().numpy()
                inference_seconds += time.perf_counter() - started
                if not np.isfinite(logits).all():
                    nan_score_count += 1
                    raise S06PolicyFailure(
                        "nan_score", "S06 learned policy emitted NaN or Inf"
                    )
                started = time.perf_counter()
                action = deterministic_argmax(logits, state.candidate_indices)
                selection_seconds += time.perf_counter() - started
            else:
                started = time.perf_counter()
                action = deterministic_random_action(
                    state.candidate_indices,
                    random_policy_seed=int(config["inference"]["random_policy_seed"]),
                    graph_sha256=graph.graph_sha256, solver_seed=task.solver_seed,
                    decision_index=decision_index,
                )
                selection_seconds += time.perf_counter() - started
            if action not in set(map(int, state.candidate_indices)):
                invalid_action_count += 1
                raise S06PolicyFailure(
                    "invalid_action",
                    "S06 policy selected an action outside SCIP's action set",
                )
            if task.trace:
                action_position = int(np.flatnonzero(state.candidate_indices == action)[0])
                policy_trace.append({
                    "decision_index": decision_index,
                    "node_number": info["current_node_number"],
                    "depth": info["current_depth"],
                    "dual_bound": info["current_dual_bound"],
                    "candidate_count": state.candidate_count,
                    "selected_probindex": action,
                    "selected_variable": state.variable_names[action],
                    "selected_edge_id": int(state.candidate_edge_ids[action_position]),
                })
            timed_state, action_set, reward, done, info = environment.step(action)
            pdi += _reward_value(reward)
            decision_index += 1
    else:
        environment = ecole.environment.Configuring(
            reward_function=pdi_reward, information_function=information, scip_params={},
        )
        _, _, reward, done, _ = environment.reset(ecole_model)
        pdi += _reward_value(reward)
        if done:
            raise RuntimeError("S06 native environment terminated before configuration action")
        _, _, reward, done, _ = environment.step({})
        pdi += _reward_value(reward)
        if not done:
            raise RuntimeError("S06 native configuration action did not finish the solve")

    solve_wall_seconds = time.perf_counter() - solve_started
    solved_model = information._pyscip_model
    if solved_model is None:
        raise RuntimeError("S06 solve produced no retained SCIP model")
    status = str(solved_model.getStatus())
    scip_statistics = read_scip_statistics(solved_model)
    if task.method in {"b0_il", "random_candidate"} and scip_statistics["branch_decisions"] != decision_index:
        raise RuntimeError("S06 Ecole actions and SCIP internal-node counts disagree")
    solved = status == "optimal"
    par2 = solve_wall_seconds if solved else 2.0 * float(config["profile"]["time_limit_seconds"])
    primal = _finite_or_none(solved_model.getPrimalbound())
    dual = _finite_or_none(solved_model.getDualbound())
    gap = _finite_or_none(solved_model.getGap())
    solution_validation: dict[str, Any] | None = None
    solution_validation_failure_count = 0
    if solved:
        best_solution = solved_model.getBestSol()
        if best_solution is None:
            raise RuntimeError("optimal S06 solve has no best solution")
        selected_edges = []
        for variable in solved_model.getVars(transformed=False):
            name = original_variable_name(variable.name)
            if not name.startswith("stp_x_"):
                continue
            if float(solved_model.getSolVal(best_solution, variable)) > 0.5:
                selected_edges.append(edge_id_from_scip_variable_name(name))
        check = check_selected_edges(graph, selected_edges, claimed_objective=primal)
        solution_validation_failure_count = 0 if check.feasible else 1
        solution_validation = {
            "feasible": check.feasible,
            "selected_edge_count": len(check.selected_edge_ids),
            "reached_terminal_count": len(check.reached_terminals),
            "objective": check.objective,
            "errors": list(check.errors),
        }
    root_lp = scip_statistics["root_lp_bound"]
    root_gap = None
    if root_lp is not None and primal is not None:
        denominator = min(abs(root_lp), abs(primal))
        if denominator > 0.0 and root_lp * primal >= 0.0:
            root_gap = abs(primal - root_lp) / denominator
    policy_callback_seconds = (
        extraction_seconds + normalization_seconds + inference_seconds + selection_seconds
    )
    result = {
        "schema_version": 1,
        "task": task.to_dict(),
        "status": status,
        "solved": solved,
        "classification": "root_solved" if solved and scip_statistics["branch_decisions"] == 0 else status,
        "graph_sha256": graph.graph_sha256,
        "metadata_sha256": build.metadata.sha256,
        "checkpoint": None if checkpoint_manifest is None else {
            "training_seed": checkpoint_manifest["training_seed"],
            "checkpoint_sha256": checkpoint_manifest["checkpoint_sha256"],
            "normalization_sha256": checkpoint_manifest["normalization_sha256"],
        },
        "metrics": {
            "solve_wall_seconds": solve_wall_seconds,
            "scip_solve_seconds": float(solved_model.getSolvingTime()),
            "par2_seconds": par2,
            "primal_dual_integral": pdi,
            "primal_bound": primal,
            "dual_bound": dual,
            "final_gap": gap,
            "nodes": int(solved_model.getNNodes()),
            "lp_iterations": int(solved_model.getNLPIterations()),
            "exact_time_to_first_incumbent": scip_statistics["exact_time_to_first_incumbent"],
            "root_lp_bound": root_lp,
            "root_gap_to_final_primal": root_gap,
            "branch_decisions": scip_statistics["branch_decisions"],
            "maximum_depth": scip_statistics["maximum_depth"],
        },
        "overhead": {
            "setup_seconds": setup_seconds,
            "feature_extraction_seconds": extraction_seconds,
            "normalization_seconds": normalization_seconds,
            "inference_seconds": inference_seconds,
            "action_selection_seconds": selection_seconds,
            "policy_callback_seconds": policy_callback_seconds,
            "metric_instrumentation_seconds": information.recorder.instrumentation_seconds,
        },
        "correctness": {
            "invalid_action_count": invalid_action_count,
            "mapping_failure_count": mapping_failure_count,
            "nan_score_count": nan_score_count,
            "unexpected_fallback_count": unexpected_fallback_count,
            "solution_validation_failure_count": solution_validation_failure_count,
        },
        "solution_validation": solution_validation,
        "trace": {
            "enabled": task.trace,
            "policy_decisions": policy_trace[: int(config["trace"]["maximum_branch_records"])],
            "branch_subtrees": information.recorder.branch_trace,
        },
        "resources": {"peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0},
        "effective_parameters": effective,
    }
    return result


def load_valid_shard(path: Path, task: S06Task) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if raw.get("protocol_file_sha256") != S06_CONFIG_FILE_SHA256 or raw.get("task_sha256") != task_sha256(task):
        raise RuntimeError(f"stale or mismatched S06 shard must not be reused: {path}")
    return raw


def _pair_order(result: Mapping[str, Any]) -> tuple[int, float, float]:
    metrics = result["metrics"]
    pdi = metrics.get("primal_dual_integral")
    return (
        1 if result["solved"] else 0,
        -float(metrics["par2_seconds"]),
        -float(pdi) if pdi is not None and math.isfinite(float(pdi)) else -math.inf,
    )


def _percentile_interval(values: np.ndarray, confidence: float) -> list[float]:
    alpha = (1.0 - confidence) / 2.0
    return [float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))]


def aggregate_s06(
    config: Mapping[str, Any], main_tasks: Sequence[S06Task], strong_tasks: Sequence[S06Task],
    shard_dir: Path,
) -> dict[str, Any]:
    task_by_id = {task.task_id: task for task in main_tasks + strong_tasks}
    shards: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for task_id, task in task_by_id.items():
        shard = load_valid_shard(shard_dir / f"{task_id}.json", task)
        if shard is None:
            missing.append(task_id)
        else:
            shards[task_id] = shard
    main_ids = {task.task_id for task in main_tasks}
    missing_main = [task_id for task_id in missing if task_id in main_ids]
    results: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    completed_results: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    solver_errors = 0
    correctness_totals = Counter()
    for task in main_tasks:
        shard = shards.get(task.task_id)
        if shard is None or "result" not in shard:
            solver_errors += 1
            continue
        result = shard["result"]
        results[(task.instance.graph_sha256, task.solver_seed, task.method)] = result
        correctness_totals.update(result["correctness"])
        if shard.get("execution_status") == "completed":
            completed_results[(task.instance.graph_sha256, task.solver_seed, task.method)] = result
        else:
            solver_errors += 1
    expected_pairs = 30 * len(SOLVER_SEEDS)
    rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]))
    bootstrap_indices = rng.integers(0, 30, size=(int(config["statistics"]["bootstrap_replicates"]), 30))
    comparisons: dict[str, Any] = {}
    comparison_gates: dict[str, bool] = {}
    for baseline in WEAK_BASELINES:
        pair_rows: list[dict[str, Any]] = []
        by_graph: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for task in main_tasks:
            if task.method != "b0_il":
                continue
            key = (task.instance.graph_sha256, task.solver_seed)
            il = results.get((*key, "b0_il"))
            base = results.get((*key, baseline))
            if il is None or base is None:
                continue
            il_pdi = il["metrics"].get("primal_dual_integral")
            base_pdi = base["metrics"].get("primal_dual_integral")
            pdi_effect = (
                float(base_pdi) - float(il_pdi)
                if il_pdi is not None and base_pdi is not None
                and math.isfinite(float(il_pdi)) and math.isfinite(float(base_pdi))
                else None
            )
            row = {
                "graph_sha256": key[0], "family": task.instance.family,
                "solver_seed": key[1],
                "par2_effect": float(base["metrics"]["par2_seconds"]) - float(il["metrics"]["par2_seconds"]),
                "pdi_effect": pdi_effect,
                "il_solved": bool(il["solved"]), "baseline_solved": bool(base["solved"]),
                "order": 1 if _pair_order(il) > _pair_order(base) else -1 if _pair_order(il) < _pair_order(base) else 0,
                "catastrophic": bool(base["solved"] and (
                    not il["solved"] or float(il["metrics"]["solve_wall_seconds"]) >
                    float(config["gate"]["catastrophic_slowdown_threshold"]) * float(base["metrics"]["solve_wall_seconds"])
                )),
            }
            pair_rows.append(row)
            by_graph[key[0]].append(row)
        graph_order = [instance.graph_sha256 for instance in load_s06_instances()]
        graph_par2 = np.array([
            np.mean([row["par2_effect"] for row in by_graph[graph]]) for graph in graph_order
        ]) if len(by_graph) == 30 and all(len(by_graph[g]) == 5 for g in graph_order) else np.array([])
        pdi_complete = bool(
            graph_par2.size
            and all(row["pdi_effect"] is not None for row in pair_rows)
        )
        graph_pdi = np.array([
            np.mean([row["pdi_effect"] for row in by_graph[graph]]) for graph in graph_order
        ]) if pdi_complete else np.array([])
        if graph_par2.size:
            par2_bootstrap = graph_par2[bootstrap_indices].mean(axis=1)
            par2_ci = _percentile_interval(par2_bootstrap, float(config["statistics"]["confidence_level"]))
        else:
            par2_ci = [None, None]
        if graph_pdi.size:
            pdi_bootstrap = graph_pdi[bootstrap_indices].mean(axis=1)
            pdi_ci = _percentile_interval(pdi_bootstrap, float(config["statistics"]["confidence_level"]))
        else:
            pdi_ci = [None, None]
        family_effects = {
            family: float(np.mean([row["par2_effect"] for row in pair_rows if row["family"] == family]))
            for family in SYNTHETIC_FAMILIES if any(row["family"] == family for row in pair_rows)
        }
        wins = sum(row["order"] > 0 for row in pair_rows)
        losses = sum(row["order"] < 0 for row in pair_rows)
        il_solved = sum(row["il_solved"] for row in pair_rows)
        baseline_solved = sum(row["baseline_solved"] for row in pair_rows)
        mean_par2 = float(np.mean([row["par2_effect"] for row in pair_rows])) if pair_rows else None
        pdi_values = [row["pdi_effect"] for row in pair_rows if row["pdi_effect"] is not None]
        mean_pdi = float(np.mean(pdi_values)) if len(pdi_values) == expected_pairs else None
        positive_families = sum(value > 0.0 for value in family_effects.values())
        gate_values = {
            "complete_150_pairs": len(pair_rows) == expected_pairs,
            "b0_solved_count_not_lower": il_solved >= baseline_solved,
            "mean_par2_effect_strictly_positive": mean_par2 is not None and mean_par2 > 0.0,
            "par2_bootstrap_ci_lower_bound": par2_ci[0] is not None and par2_ci[0] >= float(config["gate"]["each_weak_baseline"]["par2_bootstrap_ci_lower_bound_min"]),
            "mean_pdi_effect_strictly_positive": mean_pdi is not None and mean_pdi > 0.0,
            "pdi_bootstrap_ci_lower_bound": pdi_ci[0] is not None and pdi_ci[0] >= float(config["gate"]["each_weak_baseline"]["pdi_bootstrap_ci_lower_bound_min"]),
            "paired_lexicographic_wins_exceed_losses": wins > losses,
            "minimum_positive_family_par2_effects": positive_families >= int(config["gate"]["each_weak_baseline"]["minimum_positive_family_par2_effects"]),
        }
        comparisons[baseline] = {
            "pairs": len(pair_rows), "pdi_pairs": len(pdi_values),
            "b0_solved": il_solved, "baseline_solved": baseline_solved,
            "mean_par2_effect_seconds": mean_par2, "par2_effect_ci95": par2_ci,
            "mean_pdi_effect": mean_pdi, "pdi_effect_ci95": pdi_ci,
            "lexicographic_wins": wins, "ties": len(pair_rows) - wins - losses,
            "lexicographic_losses": losses, "family_mean_par2_effects": family_effects,
            "positive_family_count": positive_families,
            "catastrophic_slowdown_count": sum(row["catastrophic"] for row in pair_rows),
            "catastrophic_slowdown_rate": (
                sum(row["catastrophic"] for row in pair_rows) / len(pair_rows) if pair_rows else None
            ),
            "gate": gate_values,
        }
        comparison_gates[baseline] = all(gate_values.values())
    objective_ok = True
    objective_disagreements: list[dict[str, Any]] = []
    tolerance = float(config["gate"]["correctness"]["objective_tolerance"])
    for graph in {task.instance.graph_sha256 for task in main_tasks}:
        for seed in SOLVER_SEEDS:
            solved_values = [
                (method, results[(graph, seed, method)]["metrics"]["primal_bound"])
                for method in MAIN_METHODS
                if (graph, seed, method) in results and results[(graph, seed, method)]["solved"]
            ]
            finite = [(method, float(value)) for method, value in solved_values if value is not None]
            if finite and max(value for _, value in finite) - min(value for _, value in finite) > tolerance:
                objective_ok = False
                objective_disagreements.append({"graph_sha256": graph, "solver_seed": seed, "values": finite})
    correctness_gate = {
        "complete_main_matrix": not missing_main and len(completed_results) == 750,
        "complete_strong_diagnostic_subset": all(
            task.task_id in shards for task in strong_tasks
        ),
        "zero_solver_errors": solver_errors == 0,
        "zero_invalid_actions": correctness_totals["invalid_action_count"] == 0,
        "zero_mapping_failures": correctness_totals["mapping_failure_count"] == 0,
        "zero_nan_scores": correctness_totals["nan_score_count"] == 0,
        "zero_unexpected_fallbacks": correctness_totals["unexpected_fallback_count"] == 0,
        "zero_solution_validation_failures": correctness_totals["solution_validation_failure_count"] == 0,
        "finite_pdi": len(completed_results) == 750 and all(
            result["metrics"].get("primal_dual_integral") is not None
            and math.isfinite(float(result["metrics"]["primal_dual_integral"]))
            for result in completed_results.values()
        ),
        "objective_agreement": objective_ok,
    }
    overall = all(correctness_gate.values()) and all(comparison_gates.values())
    return {
        "schema_version": 1, "stage": "S06", "experiment_id": config["experiment_id"],
        "protocol_file_sha256": S06_CONFIG_FILE_SHA256,
        "instance_manifest_sha256": S06_INSTANCES_FILE_SHA256,
        "completion": {
            "expected_main_tasks": 750,
            "observed_terminal_main_results": len(results),
            "observed_completed_main_tasks": len(completed_results),
            "expected_strong_diagnostic_tasks": 5,
            "observed_strong_shards": sum(task.task_id in shards for task in strong_tasks),
            "missing_main_tasks": missing_main,
            "missing_strong_diagnostic_tasks": [task_id for task_id in missing if task_id not in main_ids],
            "solver_errors": solver_errors,
        },
        "correctness_totals": dict(correctness_totals),
        "objective_disagreements": objective_disagreements,
        "comparisons": comparisons,
        "gate": {"correctness": correctness_gate, "weak_baselines": comparison_gates, "overall_pass": overall},
        "test_and_final_accessed": False,
        "s07_authorized": False,
    }


def trace_replay_tasks(
    main_tasks: Sequence[S06Task], shards: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[S06Task, ...]:
    by_key = {(task.instance.graph_sha256, task.solver_seed, task.method): task for task in main_tasks}
    result_by_key = {
        (task.instance.graph_sha256, task.solver_seed, task.method): shards[task.task_id]["result"]
        for task in main_tasks
        if task.task_id in shards and "result" in shards[task.task_id]
    }
    triggered: set[tuple[str, int]] = set()
    threshold = float(config["gate"]["catastrophic_slowdown_threshold"])
    for graph, seed, _ in by_key:
        il = result_by_key.get((graph, seed, "b0_il"))
        if il is None:
            continue
        for baseline in WEAK_BASELINES:
            base = result_by_key.get((graph, seed, baseline))
            if base is None:
                continue
            lost = _pair_order(il) < _pair_order(base)
            slow = bool(base["solved"] and (
                not il["solved"] or float(il["metrics"]["solve_wall_seconds"]) >
                threshold * float(base["metrics"]["solve_wall_seconds"])
            ))
            if lost or slow:
                triggered.add((graph, seed))
    tasks = []
    for graph, seed in sorted(triggered):
        for method in MAIN_METHODS:
            original = by_key[(graph, seed, method)]
            tasks.append(replace(original, task_id=f"trace--{original.task_id}", gate_relevant=False, trace=True))
    return tuple(tasks)
