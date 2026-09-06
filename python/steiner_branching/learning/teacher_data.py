"""Strict S05 teacher configuration and checksum-addressed state shards."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import StrictConfigError, load_yaml_mapping
from ..contracts import canonical_json, content_sha256
from ..data.generate import SYNTHETIC_FAMILIES
from ..data.split import split_for_synthetic_seed
from ..solver.bipartite_observation import (
    MILP_BIPARTITE_V1,
    MilpBipartiteState,
    make_bipartite_state,
)


EXPECTED_STACK_ID = "scip804-ecole081-pyscipopt430"
PILOT_TEACHER_SEEDS = (1001,)
FORMAL_TEACHER_SEEDS = (1001, 1002, 1003)
PILOT_TRAINING_SEEDS = (101, 202, 303)
FORMAL_TRAINING_SEEDS = (101, 202, 303, 404, 505)
PILOT_V1_INSTANCES = (
    ("train", "sparse_erdos_renyi", 100302),
    ("train", "random_geometric", 100308),
    ("train", "grid_with_holes", 100314),
    ("train", "community_block", 100320),
    ("train", "bridge_bottleneck", 100326),
    ("validation_iid", "sparse_erdos_renyi", 200000),
    ("validation_iid", "random_geometric", 200001),
    ("validation_iid", "grid_with_holes", 200002),
    ("validation_iid", "community_block", 200003),
    ("validation_iid", "bridge_bottleneck", 200004),
)
PILOT_V2_INSTANCES = PILOT_V1_INSTANCES + (
    ("train", "sparse_erdos_renyi", 100303),
    ("train", "grid_with_holes", 100315),
)
PILOT_V3_INSTANCES = PILOT_V2_INSTANCES
PILOT_EXPERIMENTS = {
    "s05-teacher-il-pilot-v1": PILOT_V1_INSTANCES,
    "s05-teacher-il-pilot-v2": PILOT_V2_INSTANCES,
    "s05-teacher-il-pilot-v3": PILOT_V3_INSTANCES,
}


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


@dataclass(frozen=True)
class TeacherTask:
    task_id: str
    split: str
    family: str
    bucket_id: str
    n_nodes: int
    n_terminals: int
    generator_seed: int
    teacher_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_s05_config(path: Path | str) -> dict[str, Any]:
    raw = load_yaml_mapping(path)
    _require_keys(
        raw,
        {
            "schema_version", "stage", "experiment_id", "solver_stack_id",
            "protocol_id", "formulation_id", "bipartite_schema_id", "model_id",
            "model_config", "split_policy", "required_s04_audited_tag", "limits",
            "controls", "teacher", "pilot_instances", "training", "artifacts", "gate",
        },
        "S05 config",
    )
    if raw["schema_version"] != 1 or raw["stage"] != "S05":
        raise StrictConfigError("S05 config schema_version/stage mismatch")
    experiment_id = str(raw["experiment_id"])
    expected_instances = PILOT_EXPERIMENTS.get(experiment_id)
    if expected_instances is None:
        raise StrictConfigError("S05 experiment_id is not a registered pilot")
    expected_identity = {
        "solver_stack_id": EXPECTED_STACK_ID,
        "protocol_id": "P1",
        "formulation_id": "rooted_mcf_v1",
        "bipartite_schema_id": MILP_BIPARTITE_V1.schema_id,
        "model_id": "b0_milp_gcnn_v1",
    }
    for key, expected in expected_identity.items():
        if raw[key] != expected:
            raise StrictConfigError(f"S05 {key} must remain {expected}")
    if raw["required_s04_audited_tag"] != "steiner-s04-audited-v2":
        raise StrictConfigError("S05 must require the S04 audited-v2 tag")
    if raw["model_config"] != "configs/steiner/models/b0_milp_gcnn_v1.yml":
        raise StrictConfigError("S05 model config path changed")
    if raw["split_policy"] != "configs/steiner/splits/split_policy_v1.yml":
        raise StrictConfigError("S05 split policy path changed")

    limits = _mapping(raw["limits"], "limits")
    _require_keys(limits, {"time_seconds", "nodes", "memory_mb", "threads"}, "limits")
    if limits != {
        "time_seconds": 600,
        "nodes": 200000,
        "memory_mb": 8192,
        "threads": 1,
    }:
        raise StrictConfigError("S05 limits differ from frozen P1")
    controls = _mapping(raw["controls"], "controls")
    _require_keys(
        controls,
        {
            "presolving_maxrounds", "separating_maxrounds",
            "separating_maxroundsroot", "heuristics", "restart_limit",
            "node_selector", "propagation",
        },
        "controls",
    )
    if controls != {
        "presolving_maxrounds": 0,
        "separating_maxrounds": 0,
        "separating_maxroundsroot": 0,
        "heuristics": "off",
        "restart_limit": 0,
        "node_selector": "estimate",
        "propagation": "default",
    }:
        raise StrictConfigError("S05 controls differ from frozen P1")

    teacher = _mapping(raw["teacher"], "teacher")
    _require_keys(
        teacher,
        {
            "collection_seeds_pilot", "collection_seeds_formal",
            "max_states_per_task", "iteration_limit_per_candidate", "candidate_limit",
            "idempotent", "infeasible_gain_cap", "tie_relative_tolerance", "trajectory",
        },
        "teacher",
    )
    if tuple(teacher["collection_seeds_pilot"]) != PILOT_TEACHER_SEEDS:
        raise StrictConfigError("S05 pilot teacher seeds changed")
    if tuple(teacher["collection_seeds_formal"]) != FORMAL_TEACHER_SEEDS:
        raise StrictConfigError("S05 formal teacher seeds changed")
    if int(teacher["max_states_per_task"]) < 1:
        raise StrictConfigError("teacher max_states_per_task must be positive")
    if int(teacher["iteration_limit_per_candidate"]) < 1:
        raise StrictConfigError("teacher iteration limit must be positive")
    if int(teacher["candidate_limit"]) != 0 or teacher["idempotent"] is not False:
        raise StrictConfigError("pilot teacher must evaluate all candidates non-idempotently")
    if float(teacher["infeasible_gain_cap"]) <= 0.0:
        raise StrictConfigError("teacher infeasible gain cap must be positive")
    if float(teacher["tie_relative_tolerance"]) != 1.0e-9:
        raise StrictConfigError("teacher tie tolerance differs from S03")
    if teacher["trajectory"] != "strong_then_most_infeasible":
        raise StrictConfigError("unknown S05 teacher trajectory")

    training = _mapping(raw["training"], "training")
    training_fields = {
        "pilot_seeds", "formal_seeds", "learning_curve_train_states", "epochs",
        "learning_rate", "weight_decay", "gradient_clip_norm", "target_temperature",
        "device", "checkpoint_root",
    }
    if experiment_id == "s05-teacher-il-pilot-v3":
        training_fields.update({"deterministic_algorithms", "cublas_workspace_config"})
    _require_keys(training, training_fields, "training")
    if tuple(training["pilot_seeds"]) != PILOT_TRAINING_SEEDS:
        raise StrictConfigError("S05 pilot training seeds changed")
    if tuple(training["formal_seeds"]) != FORMAL_TRAINING_SEEDS:
        raise StrictConfigError("S05 formal training seeds changed")
    curve = tuple(int(value) for value in training["learning_curve_train_states"])
    if not curve or curve != tuple(sorted(set(curve))) or any(value < 1 for value in curve):
        raise StrictConfigError("learning curve state counts must be positive and increasing")
    if training["device"] != "cuda":
        raise StrictConfigError("S05 long training device must be CUDA")
    if experiment_id == "s05-teacher-il-pilot-v3" and (
        training["deterministic_algorithms"] is not True
        or training["cublas_workspace_config"] != ":4096:8"
    ):
        raise StrictConfigError("S05 pilot-v3 CUDA determinism controls changed")
    for key in (
        "epochs", "learning_rate", "weight_decay", "gradient_clip_norm",
        "target_temperature",
    ):
        if not math.isfinite(float(training[key])) or float(training[key]) <= 0.0:
            raise StrictConfigError(f"training.{key} must be finite and positive")
    if training["checkpoint_root"] != f"checkpoints/steiner/s05/{experiment_id}":
        raise StrictConfigError("S05 checkpoint root changed")

    instances = raw["pilot_instances"]
    if not isinstance(instances, list) or len(instances) != len(expected_instances):
        raise StrictConfigError(
            f"S05 {experiment_id} must contain exactly {len(expected_instances)} "
            "preregistered instances"
        )
    seen: set[tuple[str, int]] = set()
    by_split: dict[str, set[str]] = {"train": set(), "validation_iid": set()}
    observed_instances: list[tuple[str, str, int]] = []
    for index, value in enumerate(instances):
        item = _mapping(value, f"pilot_instances[{index}]")
        _require_keys(
            item,
            {"split", "family", "bucket_id", "n_nodes", "n_terminals", "generator_seed"},
            f"pilot_instances[{index}]",
        )
        split = str(item["split"])
        family = str(item["family"])
        seed = int(item["generator_seed"])
        if split not in by_split or family not in SYNTHETIC_FAMILIES:
            raise StrictConfigError("pilot instance has an unsupported split/family")
        observed_split = split_for_synthetic_seed(seed, policy_path=raw["split_policy"])
        if observed_split != split:
            raise StrictConfigError(
                f"pilot seed {seed} belongs to {observed_split}, not {split}"
            )
        if item["bucket_id"] != "medium-mid" or int(item["n_nodes"]) != 96 or int(item["n_terminals"]) != 19:
            raise StrictConfigError("S05 pilot instances must use medium-mid 96/19")
        if (split, seed) in seen:
            raise StrictConfigError("duplicate S05 pilot instance seed")
        seen.add((split, seed))
        by_split[split].add(family)
        observed_instances.append((split, family, seed))
    if tuple(observed_instances) != expected_instances:
        raise StrictConfigError(f"S05 {experiment_id} instance identities/order changed")
    expected_families = set(SYNTHETIC_FAMILIES)
    if any(families != expected_families for families in by_split.values()):
        raise StrictConfigError("each S05 pilot split must cover all five families")

    gate = _mapping(raw["gate"], "gate")
    _require_keys(
        gate,
        {
            "min_teacher_valid_state_fraction", "max_teacher_all_tie_valid_state_fraction",
            "required_action_mapping_rate", "require_validation_regret_better_than_random",
            "require_all_pilot_training_seeds", "require_zero_split_leakage",
            "require_checkpoint_reload_parity",
        },
        "gate",
    )
    if float(gate["min_teacher_valid_state_fraction"]) != 0.60:
        raise StrictConfigError("S05 teacher-valid Gate differs from S03")
    if float(gate["max_teacher_all_tie_valid_state_fraction"]) != 0.40:
        raise StrictConfigError("S05 tie Gate differs from S03")
    if float(gate["required_action_mapping_rate"]) != 1.0:
        raise StrictConfigError("S05 requires exact action mapping")
    if any(gate[key] is not True for key in (
        "require_validation_regret_better_than_random",
        "require_all_pilot_training_seeds",
        "require_zero_split_leakage",
        "require_checkpoint_reload_parity",
    )):
        raise StrictConfigError("S05 Boolean Gate requirements cannot be disabled")
    artifacts = _mapping(raw["artifacts"], "artifacts")
    _require_keys(artifacts, {"raw_root", "report_root"}, "artifacts")
    if artifacts != {
        "raw_root": f"results/steiner/raw/s05/{experiment_id}",
        "report_root": f"results/steiner/s05/{experiment_id}",
    }:
        raise StrictConfigError("S05 artifact roots changed")
    return raw


def s05_config_sha256(config: Mapping[str, Any]) -> str:
    return content_sha256(dict(config))


def expand_pilot_tasks(config: Mapping[str, Any]) -> tuple[TeacherTask, ...]:
    tasks: list[TeacherTask] = []
    for item in config["pilot_instances"]:
        for teacher_seed in config["teacher"]["collection_seeds_pilot"]:
            instance_id = f"{item['split']}--{item['family']}--s{item['generator_seed']}"
            tasks.append(
                TeacherTask(
                    task_id=f"{instance_id}--teacher{teacher_seed}",
                    split=str(item["split"]),
                    family=str(item["family"]),
                    bucket_id=str(item["bucket_id"]),
                    n_nodes=int(item["n_nodes"]),
                    n_terminals=int(item["n_terminals"]),
                    generator_seed=int(item["generator_seed"]),
                    teacher_seed=int(teacher_seed),
                )
            )
    if len({task.task_id for task in tasks}) != len(tasks):
        raise StrictConfigError("expanded S05 task IDs are not unique")
    return tuple(tasks)


def task_sha256(task: TeacherTask, config_sha256: str) -> str:
    return content_sha256({"task": task.to_dict(), "config_sha256": config_sha256})


def immutable_array(values: Any, dtype: Any) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class StrongBranchLabels:
    candidate_probindices: np.ndarray
    scores: np.ndarray
    down_bounds: np.ndarray
    up_bounds: np.ndarray
    down_valid: np.ndarray
    up_valid: np.ndarray
    down_infeasible: np.ndarray
    up_infeasible: np.ndarray
    lp_errors: np.ndarray
    node_number: int
    depth: int
    elapsed_seconds: float
    lp_iterations_delta: int
    strong_lp_iterations_delta: int
    strong_calls_delta: int

    def __post_init__(self) -> None:
        n = int(self.candidate_probindices.size)
        if self.candidate_probindices.shape != (n,) or len(set(map(int, self.candidate_probindices))) != n:
            raise ValueError("teacher candidate probindices must be unique and one-dimensional")
        for values in (self.scores, self.down_bounds, self.up_bounds):
            if values.shape != (n,):
                raise ValueError("teacher floating arrays must align with candidates")
        for values in (
            self.down_valid, self.up_valid, self.down_infeasible,
            self.up_infeasible, self.lp_errors,
        ):
            if values.shape != (n,) or values.dtype != np.bool_:
                raise ValueError("teacher validity arrays must be Boolean and candidate-aligned")
        if any(values.flags.writeable for values in self.arrays):
            raise ValueError("teacher label arrays must be immutable")
        if n < 1 or int(self.candidate_probindices.min()) < 0:
            raise ValueError("teacher state must contain legal candidate probindices")
        valid = self.fully_valid
        if np.any(~np.isfinite(self.scores[valid])) or np.any(~np.isnan(self.scores[~valid])):
            raise ValueError("teacher scores must be finite exactly for fully valid candidates")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise ValueError("teacher elapsed_seconds must be finite and non-negative")
        if min(self.lp_iterations_delta, self.strong_lp_iterations_delta, self.strong_calls_delta) < 0:
            raise ValueError("teacher cost counters must be non-negative")

    @property
    def arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.candidate_probindices, self.scores, self.down_bounds, self.up_bounds,
            self.down_valid, self.up_valid, self.down_infeasible,
            self.up_infeasible, self.lp_errors,
        )

    @property
    def fully_valid(self) -> np.ndarray:
        result = (
            ~self.lp_errors
            & (self.down_valid | self.down_infeasible)
            & (self.up_valid | self.up_infeasible)
        )
        result.setflags(write=False)
        return result

    @property
    def state_valid(self) -> bool:
        return (
            self.scores.size >= 2
            and bool(np.all(self.fully_valid))
            and self.strong_calls_delta >= self.scores.size
        )

    def all_tie(self, tolerance: float) -> bool:
        if not self.state_valid:
            return False
        scale = max(1.0, float(np.max(np.abs(self.scores))))
        return float(np.max(self.scores) - np.min(self.scores)) <= tolerance * scale


@dataclass(frozen=True)
class TeacherSample:
    task_id: str
    split: str
    family: str
    graph_sha256: str
    teacher_seed: int
    state_index: int
    state: MilpBipartiteState
    labels: StrongBranchLabels
    pseudocost_scores: np.ndarray

    def __post_init__(self) -> None:
        self.state.validate()
        n = self.state.candidate_count
        if n != self.labels.scores.size or self.pseudocost_scores.shape != (n,):
            raise ValueError("sample labels/pseudocosts must align with legal candidates")
        if not np.array_equal(self.state.candidate_indices, self.labels.candidate_probindices):
            raise ValueError("teacher probindices do not equal Ecole legal action rows")
        if self.pseudocost_scores.flags.writeable:
            raise ValueError("pseudocost scores must be immutable")
        if np.any(~np.isfinite(self.pseudocost_scores) & ~np.isnan(self.pseudocost_scores)):
            raise ValueError("pseudocost scores may contain only finite values or NaN")
        if self.split not in {"train", "validation_iid"}:
            raise ValueError("S05 pilot sample has an illegal split")
        if not re.fullmatch(r"[0-9a-f]{64}", self.graph_sha256) or self.state_index < 0:
            raise ValueError("sample graph hash/state index is invalid")

    @property
    def semantic_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            canonical_json(
                {
                    "task_id": self.task_id,
                    "split": self.split,
                    "family": self.family,
                    "graph_sha256": self.graph_sha256,
                    "teacher_seed": self.teacher_seed,
                    "state_index": self.state_index,
                    "state_sha256": self.state.sha256,
                    "node_number": self.labels.node_number,
                    "depth": self.labels.depth,
                    "elapsed_seconds": self.labels.elapsed_seconds,
                    "lp_iterations_delta": self.labels.lp_iterations_delta,
                    "strong_lp_iterations_delta": self.labels.strong_lp_iterations_delta,
                    "strong_calls_delta": self.labels.strong_calls_delta,
                }
            ).encode("utf-8")
        )
        for values in self.labels.arrays + (self.pseudocost_scores,):
            array = np.ascontiguousarray(values)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(tuple(array.shape)).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_teacher_sample(path: Path | str, sample: TeacherSample) -> dict[str, str]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "task_id": sample.task_id,
        "split": sample.split,
        "family": sample.family,
        "graph_sha256": sample.graph_sha256,
        "teacher_seed": sample.teacher_seed,
        "state_index": sample.state_index,
        "node_number": sample.labels.node_number,
        "depth": sample.labels.depth,
        "elapsed_seconds": sample.labels.elapsed_seconds,
        "lp_iterations_delta": sample.labels.lp_iterations_delta,
        "strong_lp_iterations_delta": sample.labels.strong_lp_iterations_delta,
        "strong_calls_delta": sample.labels.strong_calls_delta,
        "semantic_sha256": sample.semantic_sha256,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                metadata=np.asarray(canonical_json(metadata)),
                constraint_features=sample.state.constraint_features,
                variable_features=sample.state.variable_features,
                edge_indices=sample.state.edge_indices,
                edge_features=sample.state.edge_features,
                variable_names=np.asarray(sample.state.variable_names, dtype=np.str_),
                variable_global_ids=sample.state.variable_global_ids,
                constraint_global_ids=sample.state.constraint_global_ids,
                candidate_indices=sample.state.candidate_indices,
                candidate_edge_ids=sample.state.candidate_edge_ids,
                teacher_scores=sample.labels.scores,
                down_bounds=sample.labels.down_bounds,
                up_bounds=sample.labels.up_bounds,
                down_valid=sample.labels.down_valid,
                up_valid=sample.labels.up_valid,
                down_infeasible=sample.labels.down_infeasible,
                up_infeasible=sample.labels.up_infeasible,
                lp_errors=sample.labels.lp_errors,
                pseudocost_scores=sample.pseudocost_scores,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return {"file_sha256": file_sha256(output), "semantic_sha256": sample.semantic_sha256}


def load_teacher_sample(path: Path | str, *, expected_file_sha256: str | None = None) -> TeacherSample:
    input_path = Path(path)
    if expected_file_sha256 is not None and file_sha256(input_path) != expected_file_sha256:
        raise ValueError(f"teacher shard checksum mismatch: {input_path}")
    with np.load(input_path, allow_pickle=False) as shard:
        metadata = json.loads(str(shard["metadata"].item()))
        if metadata.get("schema_version") != 1:
            raise ValueError("unsupported teacher shard schema")
        state = make_bipartite_state(
            constraint_features=shard["constraint_features"],
            variable_features=shard["variable_features"],
            edge_indices=shard["edge_indices"],
            edge_features=shard["edge_features"],
            variable_names=tuple(str(value) for value in shard["variable_names"]),
            variable_global_ids=shard["variable_global_ids"],
            constraint_global_ids=shard["constraint_global_ids"],
            candidate_indices=shard["candidate_indices"],
            candidate_edge_ids=shard["candidate_edge_ids"],
        )
        labels = StrongBranchLabels(
            candidate_probindices=immutable_array(shard["candidate_indices"], np.int64),
            scores=immutable_array(shard["teacher_scores"], np.float64),
            down_bounds=immutable_array(shard["down_bounds"], np.float64),
            up_bounds=immutable_array(shard["up_bounds"], np.float64),
            down_valid=immutable_array(shard["down_valid"], np.bool_),
            up_valid=immutable_array(shard["up_valid"], np.bool_),
            down_infeasible=immutable_array(shard["down_infeasible"], np.bool_),
            up_infeasible=immutable_array(shard["up_infeasible"], np.bool_),
            lp_errors=immutable_array(shard["lp_errors"], np.bool_),
            node_number=int(metadata["node_number"]),
            depth=int(metadata["depth"]),
            elapsed_seconds=float(metadata["elapsed_seconds"]),
            lp_iterations_delta=int(metadata["lp_iterations_delta"]),
            strong_lp_iterations_delta=int(metadata["strong_lp_iterations_delta"]),
            strong_calls_delta=int(metadata["strong_calls_delta"]),
        )
        sample = TeacherSample(
            task_id=str(metadata["task_id"]),
            split=str(metadata["split"]),
            family=str(metadata["family"]),
            graph_sha256=str(metadata["graph_sha256"]),
            teacher_seed=int(metadata["teacher_seed"]),
            state_index=int(metadata["state_index"]),
            state=state,
            labels=labels,
            pseudocost_scores=immutable_array(shard["pseudocost_scores"], np.float64),
        )
    if sample.semantic_sha256 != metadata.get("semantic_sha256"):
        raise ValueError(f"teacher shard semantic checksum mismatch: {input_path}")
    return sample


def assert_no_split_leakage(samples: Sequence[TeacherSample]) -> None:
    graph_splits: dict[str, str] = {}
    for sample in samples:
        previous = graph_splits.setdefault(sample.graph_sha256, sample.split)
        if previous != sample.split:
            raise ValueError(
                f"graph lineage crosses splits: {sample.graph_sha256} ({previous}/{sample.split})"
            )
