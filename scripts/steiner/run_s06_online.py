#!/usr/bin/env python3
"""Run/resume six frozen S06 lineage shards and aggregate them fail closed."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch


REPO = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from steiner_branching.evaluation.s06_online import (  # noqa: E402
    EXPECTED_STACK_ID,
    MAIN_METHODS,
    S06_CONFIG_FILE_SHA256,
    S06_CONFIG_PATH,
    S06_INSTANCES_FILE_SHA256,
    S06_SHARD_COUNT,
    S06_WORKERS_PER_SHARD,
    S06Task,
    aggregate_s06,
    expand_s06_tasks,
    failed_task_result,
    lineage_shard_assignments,
    load_s06_config,
    load_s06_instances,
    load_valid_shard,
    run_s06_task,
    task_sha256,
    tasks_for_lineage_shard,
    trace_replay_tasks,
)
from steiner_branching.config import StrictConfigError, load_yaml_mapping  # noqa: E402
from steiner_branching.contracts import canonical_json  # noqa: E402
from steiner_branching.learning.imitation import atomic_write_json  # noqa: E402
from steiner_branching.learning.teacher_data import file_sha256  # noqa: E402


DEFAULT_ARTIFACT_ROOT = REPO / "results/steiner/raw/s06"
DEFAULT_SUMMARY = REPO / "docs/steiner/phases/S06/S06_GATE_SUMMARY.json"
ENVIRONMENT_LOCK = REPO / "configs/steiner/environment.lock.yml"
ENVIRONMENT_LOCK_SHA256 = "f70afe548f2b640a3c1375686ad8c8ef4dced63d0229c9fa4eb36e62f6d7628e"
BASE_ACTIVATION_PATH = REPO / "docs/steiner/audits/S06_PREEXECUTION_ACTIVATION_RECORD.json"
BASE_ACTIVATION_SHA256 = "2e8d30c47bf746762b8cdf2b9c605c16e45ab57c68d73d7d11a0f8e87b8a21fd"
AMENDMENT_PATH = REPO / "configs/steiner/experiments/s06_execution_amendment_a1.yml"
AMENDMENT_SHA256 = "b862f81893fb5dd46635dfc55366b5961268de41d74dd2f6cf89f11705690abe"
MAIN_WAVE_SEAL_PATH = REPO / "docs/steiner/phases/S06/S06_MAIN_WAVE_V1_SEAL.json"
MAIN_WAVE_SEAL_SHA256 = "11925fc0e5c6169a0fbdb7c1770ebb0709cefe599f6c0018f07d875595a81603"
AMENDMENT_ACTIVATION_PATH = (
    REPO / "docs/steiner/audits/S06_EXECUTION_AMENDMENT_A1_ACTIVATION_RECORD.json"
)
NORMALIZED_COMPATIBILITY_KEYS = (
    "cpu_quota_cores", "effective_cpu_cores", "memory_limit_bytes",
    "gpu_visible_count", "python_version", "solver_stack_id",
    "environment_lock_sha256", "container_runtime_fingerprint",
    "activation_record_sha256", "audited_executable_content_head", "git_head",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _committed_bytes(path: Path, label: str) -> bytes:
    candidate = path.resolve()
    try:
        local = candidate.read_bytes()
        relative = candidate.relative_to(REPO.resolve()).as_posix()
    except (OSError, ValueError) as error:
        raise StrictConfigError(f"{label} must be readable inside the repository") from error
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=REPO,
        capture_output=True, check=False,
    )
    if committed.returncode != 0 or committed.stdout != local:
        raise StrictConfigError(f"{label} must equal its committed bytes at HEAD")
    return local


def _committed_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(_committed_bytes(path, label))
    except json.JSONDecodeError as error:
        raise StrictConfigError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise StrictConfigError(f"{label} must be a mapping")
    return value


def load_execution_amendment() -> dict[str, Any]:
    if file_sha256(AMENDMENT_PATH) != AMENDMENT_SHA256:
        raise StrictConfigError("S06 execution amendment checksum changed")
    value = load_yaml_mapping(AMENDMENT_PATH)
    if (
        value.get("schema_version") != 1
        or value.get("stage") != "S06"
        or value.get("amendment_id") != "s06-main-runtime-compatibility-a1"
        or value.get("execution_authorized") is not False
        or value.get("sealed_main_wave", {}).get("evidence_tree_sha256")
        != "671cb10a78a30e9f227e6b8b86a62d3ba4437a013ba9350850bb2ef1b1fb8964"
        or tuple(value.get("compatibility_identity", {}).get("equal_across_shards", ()))
        != NORMALIZED_COMPATIBILITY_KEYS
        or value.get("post_amendment_execution", {}).get("main_phase_rerun_forbidden")
        is not True
    ):
        raise StrictConfigError("S06 execution amendment contract changed")
    return value


def load_main_wave_seal() -> dict[str, Any]:
    if file_sha256(MAIN_WAVE_SEAL_PATH) != MAIN_WAVE_SEAL_SHA256:
        raise StrictConfigError("S06 main-wave seal checksum changed")
    value = json.loads(MAIN_WAVE_SEAL_PATH.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "stage": "S06",
        "experiment_id": "s06-il-online-v1",
        "task_files": 755,
        "main_shard_manifests": 6,
        "sealed_files": 761,
        "all_task_envelopes_terminal": True,
        "solver_error_envelopes": 0,
        "evidence_tree_sha256": "671cb10a78a30e9f227e6b8b86a62d3ba4437a013ba9350850bb2ef1b1fb8964",
        "base_activation_record_sha256": BASE_ACTIVATION_SHA256,
        "base_activation_audited_content_head": "a29ef9eeb818c1694f78d2de1b29436f1861f8c3",
        "execution_git_head": "a80d7459b9b1d4d5bff9743711984cd5a649ef97",
        "model_or_baseline_effects_aggregated": False,
        "trace_wave_started": False,
        "gate_summary_created": False,
        "test_and_final_accessed": False,
        "rerun_or_mutation_authorized": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise StrictConfigError(f"S06 main-wave seal {key} changed")
    return value


def load_amendment_activation() -> dict[str, Any]:
    load_execution_amendment()
    seal = load_main_wave_seal()
    if file_sha256(BASE_ACTIVATION_PATH) != BASE_ACTIVATION_SHA256:
        raise StrictConfigError("S06 base activation checksum changed")
    _committed_bytes(BASE_ACTIVATION_PATH, "S06 base activation")
    value = _committed_json(AMENDMENT_ACTIVATION_PATH, "S06 amendment activation")
    expected = {
        "schema_version": 1,
        "stage": "S06",
        "audit_kind": "s06_execution_amendment_a1_pretrace_review",
        "verdict": "PASS",
        "blocking_findings": [],
        "execution_authorized": True,
        "verdict_source": "user_supplied_external_gpt_audit",
        "audited_branch": "research/steiner-migration",
        "protocol_yaml_sha256": S06_CONFIG_FILE_SHA256,
        "instance_manifest_sha256": S06_INSTANCES_FILE_SHA256,
        "execution_amendment_sha256": AMENDMENT_SHA256,
        "main_wave_seal_sha256": MAIN_WAVE_SEAL_SHA256,
        "main_wave_evidence_tree_sha256": seal["evidence_tree_sha256"],
        "base_activation_record_sha256": BASE_ACTIVATION_SHA256,
        "container_runtime_fingerprint": seal["registered_runtime_identity"]["container_runtime_fingerprint"],
        "formal_main_wave_at_activation": "SEALED_NO_RERUN",
        "trace_execution_authorized": True,
        "scientific_gate_at_activation": "NOT_EVALUATED",
        "s07_authorized": False,
        "test_and_final_access_authorized": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise StrictConfigError(f"S06 amendment activation {key} is not authorized")
    for key in ("recorded_at_utc", "audited_content_head", "audit_record", "audit_record_sha256"):
        if key not in value:
            raise StrictConfigError(f"S06 amendment activation {key} is missing")
    if not str(value["recorded_at_utc"]).endswith("Z"):
        raise StrictConfigError("S06 amendment activation timestamp is invalid")
    head = str(value["audited_content_head"])
    if len(head) != 40 or any(char not in "0123456789abcdef" for char in head):
        raise StrictConfigError("S06 amendment activation audited head is invalid")
    audit_path = resolve_path(str(value["audit_record"]))
    if file_sha256(audit_path) != value["audit_record_sha256"]:
        raise StrictConfigError("S06 amendment PASS audit record checksum changed")
    audit = _committed_json(audit_path, "S06 amendment PASS audit record")
    if (
        audit.get("audit_kind") != "s06_execution_amendment_a1_pretrace_review"
        or audit.get("verdict") != "PASS"
        or audit.get("blocking_findings") != []
        or audit.get("audited_content_head") != head
        or audit.get("protocol_yaml_sha256") != S06_CONFIG_FILE_SHA256
        or audit.get("instance_manifest_sha256") != S06_INSTANCES_FILE_SHA256
        or audit.get("execution_amendment_sha256") != AMENDMENT_SHA256
        or audit.get("main_wave_seal_sha256") != MAIN_WAVE_SEAL_SHA256
        or audit.get("main_wave_evidence_tree_sha256")
        != seal["evidence_tree_sha256"]
        or audit.get("base_activation_record_sha256") != BASE_ACTIVATION_SHA256
        or audit.get("trace_execution_authorized") is not True
        or audit.get("scientific_gate_at_audit") != "NOT_EVALUATED"
        or audit.get("s07_authorized") is not False
        or audit.get("test_and_final_access_authorized") is not False
    ):
        raise StrictConfigError("S06 amendment audit record does not authorize trace")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", head, "HEAD"], cwd=REPO, check=False,
    ).returncode != 0:
        raise StrictConfigError("S06 amendment audited head is outside current history")
    protected = ["python/steiner_branching", "scripts/steiner", "configs/steiner", "tests/steiner"]
    if subprocess.run(
        ["git", "diff", "--quiet", head, "HEAD", "--", *protected], cwd=REPO, check=False,
    ).returncode != 0:
        raise StrictConfigError("S06 executable inputs changed after amendment audit")
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *protected], cwd=REPO, check=False,
    ).returncode != 0:
        raise StrictConfigError("S06 executable inputs have uncommitted changes")
    value["_record_sha256"] = file_sha256(AMENDMENT_ACTIVATION_PATH)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(S06_CONFIG_PATH))
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--max-workers", type=int, default=S06_WORKERS_PER_SHARD)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=S06_SHARD_COUNT)
    parser.add_argument("--phase", choices=("main", "trace"))
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--run-one", help=argparse.SUPPRESS)
    return parser.parse_args()


def _shard_path(directory: Path, task: S06Task) -> Path:
    return directory / f"{task.task_id}.json"


def _memory_limit_bytes() -> int:
    candidates = (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    )
    for candidate in candidates:
        try:
            value = candidate.read_text(encoding="utf-8").strip()
            if value != "max":
                return int(value)
        except (OSError, ValueError):
            pass
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return -1


def _cpu_quota_cores() -> float | None:
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").split()
        if quota != "max":
            return float(quota) / float(period)
    except (OSError, ValueError):
        pass
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text(encoding="utf-8"))
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text(encoding="utf-8"))
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return None


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, IndexError):
        pass
    return platform.processor() or "unknown"


def container_runtime_fingerprint() -> str:
    if file_sha256(ENVIRONMENT_LOCK) != ENVIRONMENT_LOCK_SHA256:
        raise RuntimeError("frozen environment lock checksum changed")
    import ecole
    import numpy
    import pyscipopt
    import yaml

    payload = {
        "environment_lock_sha256": ENVIRONMENT_LOCK_SHA256,
        "python_executable_sha256": file_sha256(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "numpy_version": numpy.__version__,
        "pyyaml_version": yaml.__version__,
        "torch_version": torch.__version__,
        "ecole_version": ecole.__version__,
        "pyscipopt_version": pyscipopt.__version__,
        "solver_stack_id": os.environ.get("STEINER_SOLVER_STACK_ID"),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def runtime_identity(
    activation: Mapping[str, Any] | None = None,
    *,
    activation_record_path: Path = AMENDMENT_ACTIVATION_PATH,
) -> dict[str, Any]:
    try:
        affinity_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity_cpus = os.cpu_count() or -1
    quota_cores = _cpu_quota_cores()
    effective_cpu_cores = min(affinity_cpus, quota_cores) if quota_cores is not None else float(affinity_cpus)
    return {
        "hostname": platform.node(),
        "cpu_model": _cpu_model(),
        "cpu_affinity_count": affinity_cpus,
        "cpu_quota_cores": quota_cores,
        "effective_cpu_cores": effective_cpu_cores,
        "memory_limit_bytes": _memory_limit_bytes(),
        "gpu_visible_count": torch.cuda.device_count(),
        "python_version": platform.python_version(),
        "solver_stack_id": os.environ.get("STEINER_SOLVER_STACK_ID"),
        "environment_lock_sha256": file_sha256(ENVIRONMENT_LOCK),
        "container_runtime_fingerprint": container_runtime_fingerprint(),
        "activation_record_sha256": (
            file_sha256(activation_record_path)
            if activation is not None else None
        ),
        "audited_executable_content_head": (
            activation.get("audited_content_head") if activation is not None else None
        ),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
            capture_output=True, check=True,
        ).stdout.strip(),
    }


def validate_registered_resources(
    identity: Mapping[str, Any], activation: Mapping[str, Any]
) -> None:
    if float(identity.get("effective_cpu_cores", -1)) < 8.0:
        raise RuntimeError("S06 shard has fewer than the registered 8 effective CPU cores")
    if int(identity.get("memory_limit_bytes", -1)) < 96 * 1024 ** 3:
        raise RuntimeError("S06 shard has less than the registered 96 GiB memory")
    if identity.get("gpu_visible_count") != 0:
        raise RuntimeError("S06 shard must have zero visible GPUs")
    if identity.get("solver_stack_id") != EXPECTED_STACK_ID:
        raise RuntimeError("S06 shard solver stack differs from the frozen stack")
    if identity.get("environment_lock_sha256") != ENVIRONMENT_LOCK_SHA256:
        raise RuntimeError("S06 shard environment lock differs from the frozen lock")
    if identity.get("container_runtime_fingerprint") != activation.get("container_runtime_fingerprint"):
        raise RuntimeError("S06 shard container runtime fingerprint differs from activation")
    if identity.get("audited_executable_content_head") != activation.get("audited_content_head"):
        raise RuntimeError("S06 executable content head differs from activation")
    activation_sha = identity.get("activation_record_sha256")
    if (
        not isinstance(activation_sha, str) or len(activation_sha) != 64
        or any(char not in "0123456789abcdef" for char in activation_sha)
    ):
        raise RuntimeError("S06 activation record checksum is missing")
    expected_activation_sha = activation.get("_record_sha256")
    if expected_activation_sha is not None and activation_sha != expected_activation_sha:
        raise RuntimeError("S06 activation record checksum differs from runtime")


def validate_sealed_main_resources(
    identity: Mapping[str, Any], seal: Mapping[str, Any]
) -> None:
    """Validate legacy main evidence without pretending it used the amendment."""
    registered = seal.get("registered_runtime_identity")
    if not isinstance(registered, dict):
        raise RuntimeError("S06 sealed main runtime identity is missing")
    for key, expected in registered.items():
        if identity.get(key) != expected:
            raise RuntimeError(f"S06 sealed main runtime identity {key} mismatch")
    expected_legacy = {
        "activation_record_sha256": seal.get("base_activation_record_sha256"),
        "audited_executable_content_head": seal.get(
            "base_activation_audited_content_head"
        ),
        "git_head": seal.get("execution_git_head"),
    }
    for key, expected in expected_legacy.items():
        if identity.get(key) != expected:
            raise RuntimeError(f"S06 sealed main runtime identity {key} mismatch")
    if float(identity.get("effective_cpu_cores", -1)) < 8.0:
        raise RuntimeError("S06 sealed main shard had fewer than 8 effective CPU cores")
    if int(identity.get("memory_limit_bytes", -1)) < 96 * 1024 ** 3:
        raise RuntimeError("S06 sealed main shard had less than 96 GiB memory")
    if identity.get("gpu_visible_count") != 0:
        raise RuntimeError("S06 sealed main shard exposed a GPU")


def _main_wave_evidence_records(
    run_dir: Path, all_tasks: Sequence[S06Task]
) -> list[dict[str, str]]:
    expected_task_paths = {
        run_dir / "shards" / f"{task.task_id}.json" for task in all_tasks
    }
    actual_task_paths = set((run_dir / "shards").glob("*.json"))
    if actual_task_paths != expected_task_paths:
        raise RuntimeError("S06 sealed main task-file membership changed")
    expected_manifest_paths = {
        _phase_manifest_path(run_dir, "main", shard_index)
        for shard_index in range(S06_SHARD_COUNT)
    }
    actual_manifest_paths = set(
        (run_dir / "shard_manifests").glob("main-shard-*.json")
    )
    if actual_manifest_paths != expected_manifest_paths:
        raise RuntimeError("S06 sealed main-manifest membership changed")
    paths = list(expected_task_paths | expected_manifest_paths)
    records: list[dict[str, str]] = []
    for path in sorted(paths):
        if not path.is_file():
            raise RuntimeError(f"S06 sealed main evidence is missing: {path}")
        records.append({
            "path": path.relative_to(run_dir).as_posix(),
            "sha256": file_sha256(path),
        })
    return records


def verify_main_wave_seal(
    run_dir: Path, all_tasks: Sequence[S06Task]
) -> dict[str, Any]:
    seal = load_main_wave_seal()
    records = _main_wave_evidence_records(run_dir, all_tasks)
    if len(records) != seal["sealed_files"]:
        raise RuntimeError("S06 sealed main evidence file count changed")
    evidence_tree_sha256 = hashlib.sha256(
        canonical_json(records).encode("utf-8")
    ).hexdigest()
    if evidence_tree_sha256 != seal["evidence_tree_sha256"]:
        raise RuntimeError("S06 sealed main evidence tree checksum changed")
    return seal


def assert_amendment_phase_authorized(phase: str) -> None:
    if phase == "main":
        raise SystemExit(
            "S06 formal main wave is sealed; amendment A1 forbids rerunning main tasks"
        )


def _write_one(task: S06Task, shard_dir: Path) -> int:
    output = _shard_path(shard_dir, task)
    if load_valid_shard(output, task) is not None:
        print(f"S06 SKIP valid-shard {task.task_id}", flush=True)
        return 0
    try:
        task_runtime_identity = json.loads(os.environ["S06_RUNTIME_IDENTITY_JSON"])
    except (KeyError, json.JSONDecodeError) as error:
        raise RuntimeError("S06 task runtime identity is unavailable") from error
    activation = load_amendment_activation()
    validate_registered_resources(task_runtime_identity, activation)
    envelope: dict[str, Any] = {
        "schema_version": 1,
        "stage": "S06",
        "protocol_file_sha256": S06_CONFIG_FILE_SHA256,
        "task_sha256": task_sha256(task),
        "task": task.to_dict(),
        "distributed_execution": {
            "phase": os.environ.get("S06_DISTRIBUTED_PHASE"),
            "shard_index": int(os.environ["S06_DISTRIBUTED_SHARD_INDEX"]),
            "shard_count": int(os.environ["S06_DISTRIBUTED_SHARD_COUNT"]),
        },
        "runtime_identity": task_runtime_identity,
        "started_at_utc": utc_now(),
    }
    try:
        config = load_s06_config(require_activation=False)
        envelope["result"] = run_s06_task(task, config)
        envelope["execution_status"] = "completed"
    except BaseException as error:
        envelope.update({
            "execution_status": "solver_error",
            "error": f"{type(error).__name__}: {error}",
            "result": failed_task_result(task, error),
        })
    envelope["finished_at_utc"] = utc_now()
    atomic_write_json(output, envelope)
    print(f"S06 WRITE status={envelope['execution_status']} task={task.task_id}", flush=True)
    return 0 if envelope["execution_status"] == "completed" else 1


def _launch(
    tasks: Sequence[S06Task], *, workers: int, script_args: argparse.Namespace,
    shard_dir: Path, phase: str, shard_index: int,
    runtime_identity_value: Mapping[str, Any],
) -> None:
    pending = [
        task for task in tasks
        if load_valid_shard(_shard_path(shard_dir, task), task) is None
    ]
    print(
        f"S06 WAVE phase={phase} shard={shard_index}/{S06_SHARD_COUNT} "
        f"workers={workers} tasks={len(tasks)} pending={len(pending)} started={utc_now()}",
        flush=True,
    )
    if not pending:
        return
    prefix = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(resolve_path(script_args.config)),
        "--artifact-root",
        str(resolve_path(script_args.artifact_root)),
        "--summary-output",
        str(resolve_path(script_args.summary_output)),
    ]
    child_environment = dict(os.environ)
    child_environment.update({
        "S06_DISTRIBUTED_PHASE": phase,
        "S06_DISTRIBUTED_SHARD_INDEX": str(shard_index),
        "S06_DISTRIBUTED_SHARD_COUNT": str(S06_SHARD_COUNT),
        "S06_RUNTIME_IDENTITY_JSON": canonical_json(dict(runtime_identity_value)),
    })

    def invoke(task: S06Task) -> tuple[str, int, str]:
        process = subprocess.run(
            prefix + ["--run-one", task.task_id],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
            env=child_environment,
        )
        return task.task_id, process.returncode, (process.stdout + process.stderr)[-4000:]

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(invoke, task): task for task in pending}
        for future in as_completed(futures):
            task_id, returncode, tail = future.result()
            print(tail, end="" if tail.endswith("\n") else "\n", flush=True)
            if returncode:
                failures.append(task_id)
    missing = [task.task_id for task in tasks if not _shard_path(shard_dir, task).is_file()]
    if missing:
        raise RuntimeError(f"S06 wave ended without terminal shards: {missing}")
    print(
        f"S06 WAVE phase={phase} shard={shard_index} finished={utc_now()} "
        f"solver_error_shards={len(failures)}",
        flush=True,
    )


def _read_shards(tasks: Sequence[S06Task], shard_dir: Path) -> dict[str, Mapping[str, Any]]:
    values: dict[str, Mapping[str, Any]] = {}
    for task in tasks:
        shard = load_valid_shard(_shard_path(shard_dir, task), task)
        if shard is not None:
            values[task.task_id] = shard
    return values


def _assigned_lineage_ids(instances: Sequence[Any], shard_index: int) -> list[str]:
    assignments = lineage_shard_assignments(instances)
    return [
        instance.instance_id
        for instance in instances
        if assignments[instance.graph_sha256] == shard_index
    ]


def _phase_manifest_path(run_dir: Path, phase: str, shard_index: int) -> Path:
    return run_dir / "shard_manifests" / f"{phase}-shard-{shard_index}.json"


def assert_new_aggregate_outputs(summary_path: Path, manifest_path: Path) -> None:
    existing = [str(path) for path in (summary_path, manifest_path) if path.exists()]
    if existing:
        raise RuntimeError(
            f"S06 formal aggregate evidence is immutable and already exists: {existing}"
        )


@contextmanager
def aggregate_lock(path: Path):
    """Let the evidence-writing Python process own the singleton OS lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("S06 aggregate lock is already held") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _phase_tasks(
    phase: str,
    shard_index: int,
    main_tasks: Sequence[S06Task],
    strong_tasks: Sequence[S06Task],
    trace_tasks: Sequence[S06Task],
    instances: Sequence[Any],
) -> tuple[S06Task, ...]:
    source = tuple(main_tasks) + tuple(strong_tasks) if phase == "main" else tuple(trace_tasks)
    return tasks_for_lineage_shard(source, instances, shard_index)


def _write_phase_manifest(
    path: Path,
    *,
    phase: str,
    shard_index: int,
    status: str,
    instances: Sequence[Any],
    assigned_tasks: Sequence[S06Task],
    started_at: str,
    runtime_identity_value: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> None:
    value: dict[str, Any] = {
        "schema_version": 1,
        "stage": "S06",
        "experiment_id": "s06-il-online-v1",
        "protocol_file_sha256": S06_CONFIG_FILE_SHA256,
        "instance_manifest_sha256": S06_INSTANCES_FILE_SHA256,
        "phase": phase,
        "shard_index": shard_index,
        "shard_count": S06_SHARD_COUNT,
        "assignment_rule": "instance_manifest_index_modulo_shard_count",
        "assigned_lineage_ids": _assigned_lineage_ids(instances, shard_index),
        "assigned_task_ids": [task.task_id for task in assigned_tasks],
        "status": status,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now() if status != "running" else None,
        "runtime_identity": dict(runtime_identity_value or runtime_identity()),
        "test_and_final_accessed": False,
    }
    if error is not None:
        value["error"] = error
    atomic_write_json(path, value)


def _load_phase_barrier(
    run_dir: Path,
    *,
    phase: str,
    all_tasks: Sequence[S06Task],
    instances: Sequence[Any],
    shard_dir: Path,
    activation: Mapping[str, Any],
    sealed_main: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    compatibility: set[tuple[Any, ...]] = set()
    for shard_index in range(S06_SHARD_COUNT):
        path = _phase_manifest_path(run_dir, phase, shard_index)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"S06 {phase} barrier missing shard {shard_index}: {error}"
            ) from error
        expected_tasks = tasks_for_lineage_shard(all_tasks, instances, shard_index)
        expected = {
            "schema_version": 1,
            "stage": "S06",
            "experiment_id": "s06-il-online-v1",
            "protocol_file_sha256": S06_CONFIG_FILE_SHA256,
            "instance_manifest_sha256": S06_INSTANCES_FILE_SHA256,
            "phase": phase,
            "shard_index": shard_index,
            "shard_count": S06_SHARD_COUNT,
            "assignment_rule": "instance_manifest_index_modulo_shard_count",
            "assigned_lineage_ids": _assigned_lineage_ids(instances, shard_index),
            "assigned_task_ids": [task.task_id for task in expected_tasks],
            "status": "completed",
            "test_and_final_accessed": False,
        }
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                raise RuntimeError(
                    f"S06 {phase} shard {shard_index} manifest {key} mismatch"
                )
        identity = value.get("runtime_identity")
        if not isinstance(identity, dict):
            raise RuntimeError(
                f"S06 {phase} shard {shard_index} runtime identity is missing"
            )
        if sealed_main is None:
            validate_registered_resources(identity, activation)
        else:
            if phase != "main":
                raise RuntimeError("S06 legacy evidence seal is valid only for main phase")
            validate_sealed_main_resources(identity, sealed_main)
        compatibility.add(tuple(
            identity.get(key) for key in NORMALIZED_COMPATIBILITY_KEYS
        ))
        for task in expected_tasks:
            envelope = load_valid_shard(_shard_path(shard_dir, task), task)
            if envelope is None:
                raise RuntimeError(
                    f"S06 {phase} barrier missing terminal task {task.task_id}"
                )
            if envelope.get("distributed_execution") != {
                "phase": phase,
                "shard_index": shard_index,
                "shard_count": S06_SHARD_COUNT,
            }:
                raise RuntimeError(
                    f"S06 {phase} task was produced by the wrong shard: {task.task_id}"
                )
            if envelope.get("runtime_identity") != identity:
                raise RuntimeError(
                    f"S06 {phase} task used a different runtime than its shard manifest: "
                    f"{task.task_id}"
                )
        manifests.append(value)
    if len(compatibility) != 1:
        raise RuntimeError(
            f"S06 {phase} shards used non-identical resource/runtime requests: "
            f"{sorted(compatibility, key=repr)!r}"
        )
    return manifests


def _run_phase(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    instances: Sequence[Any],
    main_tasks: Sequence[S06Task],
    strong_tasks: Sequence[S06Task],
    run_dir: Path,
    activation: Mapping[str, Any],
) -> int:
    if args.shard_index is None or args.phase is None:
        raise SystemExit("formal shard execution requires --shard-index and --phase")
    if args.shard_count != S06_SHARD_COUNT:
        raise SystemExit(f"--shard-count must remain {S06_SHARD_COUNT}")
    if not 0 <= args.shard_index < S06_SHARD_COUNT:
        raise SystemExit(f"--shard-index must be in [0, {S06_SHARD_COUNT})")
    assert_amendment_phase_authorized(args.phase)
    shard_dir = run_dir / "shards"
    trace_dir = run_dir / "trace_shards"
    trace_tasks: tuple[S06Task, ...] = ()
    identity = runtime_identity(activation)
    validate_registered_resources(identity, activation)
    if args.phase == "trace":
        sealed_main = verify_main_wave_seal(
            run_dir, tuple(main_tasks) + tuple(strong_tasks)
        )
        _load_phase_barrier(
            run_dir,
            phase="main",
            all_tasks=tuple(main_tasks) + tuple(strong_tasks),
            instances=instances,
            shard_dir=shard_dir,
            activation=activation,
            sealed_main=sealed_main,
        )
        trace_tasks = trace_replay_tasks(
            main_tasks, _read_shards(main_tasks, shard_dir), config
        )
    assigned = _phase_tasks(
        args.phase,
        args.shard_index,
        main_tasks,
        strong_tasks,
        trace_tasks,
        instances,
    )
    target_dir = shard_dir if args.phase == "main" else trace_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _phase_manifest_path(run_dir, args.phase, args.shard_index)
    started_at = utc_now()
    _write_phase_manifest(
        manifest_path,
        phase=args.phase,
        shard_index=args.shard_index,
        status="running",
        instances=instances,
        assigned_tasks=assigned,
        started_at=started_at,
        runtime_identity_value=identity,
    )
    try:
        _launch(
            assigned,
            workers=args.max_workers,
            script_args=args,
            shard_dir=target_dir,
            phase=args.phase,
            shard_index=args.shard_index,
            runtime_identity_value=identity,
        )
        _write_phase_manifest(
            manifest_path,
            phase=args.phase,
            shard_index=args.shard_index,
            status="completed",
            instances=instances,
            assigned_tasks=assigned,
            started_at=started_at,
            runtime_identity_value=identity,
        )
        return 0
    except BaseException as error:
        _write_phase_manifest(
            manifest_path,
            phase=args.phase,
            shard_index=args.shard_index,
            status="failed",
            instances=instances,
            assigned_tasks=assigned,
            started_at=started_at,
            runtime_identity_value=identity,
            error=f"{type(error).__name__}: {error}",
        )
        raise


def _aggregate_under_lock(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    instances: Sequence[Any],
    main_tasks: Sequence[S06Task],
    strong_tasks: Sequence[S06Task],
    run_dir: Path,
    activation: Mapping[str, Any],
) -> int:
    summary_path = resolve_path(args.summary_output)
    manifest_path = run_dir / "manifest.json"
    assert_new_aggregate_outputs(summary_path, manifest_path)
    shard_dir = run_dir / "shards"
    trace_dir = run_dir / "trace_shards"
    sealed_main = verify_main_wave_seal(
        run_dir, tuple(main_tasks) + tuple(strong_tasks)
    )
    main_manifests = _load_phase_barrier(
        run_dir,
        phase="main",
        all_tasks=tuple(main_tasks) + tuple(strong_tasks),
        instances=instances,
        shard_dir=shard_dir,
        activation=activation,
        sealed_main=sealed_main,
    )
    trace_tasks = trace_replay_tasks(
        main_tasks, _read_shards(main_tasks, shard_dir), config
    )
    trace_manifests = _load_phase_barrier(
        run_dir,
        phase="trace",
        all_tasks=trace_tasks,
        instances=instances,
        shard_dir=trace_dir,
        activation=activation,
    )
    summary = aggregate_s06(config, main_tasks, strong_tasks, shard_dir)
    summary["distributed_execution"] = {
        "shard_count": S06_SHARD_COUNT,
        "assignment_unit": "base_graph_lineage",
        "main_shard_manifest_count": len(main_manifests),
        "trace_shard_manifest_count": len(trace_manifests),
        "resource_runtime_identity_match": True,
        "main_wave_seal_sha256": MAIN_WAVE_SEAL_SHA256,
        "main_wave_evidence_tree_sha256": sealed_main["evidence_tree_sha256"],
        "execution_amendment_sha256": AMENDMENT_SHA256,
        "amendment_activation_record_sha256": file_sha256(
            AMENDMENT_ACTIVATION_PATH
        ),
    }
    summary["trace_replays"] = {
        "triggered_instance_solver_pairs": len(trace_tasks) // len(MAIN_METHODS),
        "expected_tasks": len(trace_tasks),
        "observed_shards": len(_read_shards(trace_tasks, trace_dir)),
        "gate_relevant": False,
    }
    atomic_write_json(summary_path, summary)
    manifest = {
        "schema_version": 1,
        "stage": "S06",
        "experiment_id": config["experiment_id"],
        "protocol_file_sha256": S06_CONFIG_FILE_SHA256,
        "instance_manifest_sha256": S06_INSTANCES_FILE_SHA256,
        "execution_amendment_sha256": AMENDMENT_SHA256,
        "main_wave_seal_sha256": MAIN_WAVE_SEAL_SHA256,
        "main_wave_evidence_tree_sha256": sealed_main["evidence_tree_sha256"],
        "amendment_activation_record_sha256": file_sha256(
            AMENDMENT_ACTIVATION_PATH
        ),
        "status": "completed",
        "finished_at_utc": utc_now(),
        "summary_output": str(summary_path.relative_to(REPO)),
        "scientific_gate_pass": summary["gate"]["overall_pass"],
        "main_task_ids": [task.task_id for task in main_tasks],
        "strong_diagnostic_task_ids": [task.task_id for task in strong_tasks],
        "trace_task_ids": [task.task_id for task in trace_tasks],
        "failed_or_skipped_tasks_must_remain": True,
        "test_and_final_accessed": False,
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(summary["gate"], sort_keys=True), flush=True)
    return 0 if summary["gate"]["overall_pass"] else 2


def _aggregate(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    instances: Sequence[Any],
    main_tasks: Sequence[S06Task],
    strong_tasks: Sequence[S06Task],
    run_dir: Path,
    activation: Mapping[str, Any],
) -> int:
    with aggregate_lock(run_dir / "locks" / "aggregate.lock"):
        return _aggregate_under_lock(
            args, config, instances, main_tasks, strong_tasks, run_dir, activation
        )


def main() -> int:
    args = parse_args()
    if args.max_workers != S06_WORKERS_PER_SHARD:
        raise SystemExit(f"--max-workers must remain {S06_WORKERS_PER_SHARD}")
    if args.validate_only:
        config = load_s06_config(resolve_path(args.config), require_activation=False)
        amendment = load_execution_amendment()
        seal = load_main_wave_seal()
        instances = load_s06_instances()
        main_tasks, strong_tasks = expand_s06_tasks(config, instances)
        print(json.dumps({
            "protocol_file_sha256": S06_CONFIG_FILE_SHA256,
            "execution_amendment_sha256": AMENDMENT_SHA256,
            "main_wave_seal_sha256": MAIN_WAVE_SEAL_SHA256,
            "main_wave_evidence_tree_sha256": seal["evidence_tree_sha256"],
            "instances": len(instances),
            "main_tasks": len(main_tasks),
            "strong_diagnostic_tasks": len(strong_tasks),
            "formal_execution_authorized": False,
            "amendment_execution_authorized": amendment["execution_authorized"],
            "container_runtime_fingerprint": container_runtime_fingerprint(),
            "distributed_execution": {
                "shard_count": S06_SHARD_COUNT,
                "workers_per_shard": S06_WORKERS_PER_SHARD,
                "main_tasks_per_shard": [
                    len(tasks_for_lineage_shard(main_tasks, instances, index))
                    for index in range(S06_SHARD_COUNT)
                ],
                "lineages_per_shard": [
                    _assigned_lineage_ids(instances, index)
                    for index in range(S06_SHARD_COUNT)
                ],
            },
        }, sort_keys=True))
        return 0
    if os.environ.get("STEINER_SOLVER_STACK_ID") != EXPECTED_STACK_ID:
        raise SystemExit("run through scripts/steiner/run_with_scip804.sh --python")
    config = load_s06_config(resolve_path(args.config), require_activation=False)
    activation = load_amendment_activation()
    instances = load_s06_instances()
    main_tasks, strong_tasks = expand_s06_tasks(config, instances)
    all_tasks = {task.task_id: task for task in main_tasks + strong_tasks}
    run_dir = resolve_path(args.artifact_root) / config["experiment_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.run_one:
        required_environment = (
            "S06_DISTRIBUTED_PHASE",
            "S06_DISTRIBUTED_SHARD_INDEX",
            "S06_DISTRIBUTED_SHARD_COUNT",
        )
        if not all(key in os.environ for key in required_environment):
            raise SystemExit("--run-one is internal to an audited distributed shard")
        task = all_tasks.get(args.run_one)
        if task is None and args.run_one.startswith("trace--"):
            original = all_tasks.get(args.run_one[len("trace--"):])
            if original is not None:
                from dataclasses import replace

                task = replace(
                    original,
                    task_id=args.run_one,
                    gate_relevant=False,
                    trace=True,
                )
        if task is None:
            raise SystemExit(f"unknown S06 task: {args.run_one}")
        distributed_phase = os.environ["S06_DISTRIBUTED_PHASE"]
        distributed_index = int(os.environ["S06_DISTRIBUTED_SHARD_INDEX"])
        distributed_count = int(os.environ["S06_DISTRIBUTED_SHARD_COUNT"])
        if distributed_phase not in {"main", "trace"}:
            raise SystemExit("invalid distributed phase")
        assert_amendment_phase_authorized(distributed_phase)
        if distributed_count != S06_SHARD_COUNT:
            raise SystemExit("invalid distributed shard count")
        if task.trace != (distributed_phase == "trace"):
            raise SystemExit("task kind does not match distributed phase")
        assigned = tasks_for_lineage_shard(
            (task,), instances, distributed_index, shard_count=distributed_count
        )
        if assigned != (task,):
            raise SystemExit("task does not belong to the declared distributed shard")
        directory = run_dir / ("trace_shards" if task.trace else "shards")
        directory.mkdir(parents=True, exist_ok=True)
        return _write_one(task, directory)
    if args.aggregate_only:
        if args.shard_index is not None or args.phase is not None:
            raise SystemExit("--aggregate-only cannot be combined with shard execution")
        return _aggregate(
            args, config, instances, main_tasks, strong_tasks, run_dir, activation
        )
    return _run_phase(
        args, config, instances, main_tasks, strong_tasks, run_dir, activation
    )


if __name__ == "__main__":
    raise SystemExit(main())
