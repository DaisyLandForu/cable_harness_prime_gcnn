#!/usr/bin/env python3
"""Run/resume six frozen S06 lineage shards and aggregate them fail closed."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence


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
    lineage_shard_assignments,
    load_s06_config,
    load_s06_instances,
    load_valid_shard,
    run_s06_task,
    task_sha256,
    tasks_for_lineage_shard,
    trace_replay_tasks,
)
from steiner_branching.learning.imitation import atomic_write_json  # noqa: E402


DEFAULT_ARTIFACT_ROOT = REPO / "results/steiner/raw/s06"
DEFAULT_SUMMARY = REPO / "docs/steiner/phases/S06/S06_GATE_SUMMARY.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


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


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, IndexError):
        pass
    return platform.processor() or "unknown"


def runtime_identity() -> dict[str, Any]:
    try:
        affinity_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity_cpus = os.cpu_count() or -1
    return {
        "hostname": platform.node(),
        "cpu_model": _cpu_model(),
        "cpu_affinity_count": affinity_cpus,
        "memory_limit_bytes": _memory_limit_bytes(),
        "python_version": platform.python_version(),
        "solver_stack_id": os.environ.get("STEINER_SOLVER_STACK_ID"),
    }


def _write_one(task: S06Task, shard_dir: Path) -> int:
    output = _shard_path(shard_dir, task)
    if load_valid_shard(output, task) is not None:
        print(f"S06 SKIP valid-shard {task.task_id}", flush=True)
        return 0
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
        "started_at_utc": utc_now(),
    }
    try:
        config = load_s06_config(require_activation=True)
        envelope["result"] = run_s06_task(task, config)
        envelope["execution_status"] = "completed"
    except BaseException as error:
        envelope.update({
            "execution_status": "solver_error",
            "error": f"{type(error).__name__}: {error}",
        })
    envelope["finished_at_utc"] = utc_now()
    atomic_write_json(output, envelope)
    print(f"S06 WRITE status={envelope['execution_status']} task={task.task_id}", flush=True)
    return 0 if envelope["execution_status"] == "completed" else 1


def _launch(
    tasks: Sequence[S06Task], *, workers: int, script_args: argparse.Namespace,
    shard_dir: Path, phase: str, shard_index: int,
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
        "runtime_identity": runtime_identity(),
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
        compatibility.add(tuple(identity.get(key) for key in (
            "cpu_model",
            "cpu_affinity_count",
            "memory_limit_bytes",
            "python_version",
            "solver_stack_id",
        )))
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
) -> int:
    if args.shard_index is None or args.phase is None:
        raise SystemExit("formal shard execution requires --shard-index and --phase")
    if args.shard_count != S06_SHARD_COUNT:
        raise SystemExit(f"--shard-count must remain {S06_SHARD_COUNT}")
    if not 0 <= args.shard_index < S06_SHARD_COUNT:
        raise SystemExit(f"--shard-index must be in [0, {S06_SHARD_COUNT})")
    shard_dir = run_dir / "shards"
    trace_dir = run_dir / "trace_shards"
    trace_tasks: tuple[S06Task, ...] = ()
    if args.phase == "trace":
        _load_phase_barrier(
            run_dir,
            phase="main",
            all_tasks=tuple(main_tasks) + tuple(strong_tasks),
            instances=instances,
            shard_dir=shard_dir,
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
    )
    try:
        _launch(
            assigned,
            workers=args.max_workers,
            script_args=args,
            shard_dir=target_dir,
            phase=args.phase,
            shard_index=args.shard_index,
        )
        _write_phase_manifest(
            manifest_path,
            phase=args.phase,
            shard_index=args.shard_index,
            status="completed",
            instances=instances,
            assigned_tasks=assigned,
            started_at=started_at,
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
            error=f"{type(error).__name__}: {error}",
        )
        raise


def _aggregate(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    instances: Sequence[Any],
    main_tasks: Sequence[S06Task],
    strong_tasks: Sequence[S06Task],
    run_dir: Path,
) -> int:
    shard_dir = run_dir / "shards"
    trace_dir = run_dir / "trace_shards"
    main_manifests = _load_phase_barrier(
        run_dir,
        phase="main",
        all_tasks=tuple(main_tasks) + tuple(strong_tasks),
        instances=instances,
        shard_dir=shard_dir,
    )
    summary = aggregate_s06(config, main_tasks, strong_tasks, shard_dir)
    trace_tasks = trace_replay_tasks(
        main_tasks, _read_shards(main_tasks, shard_dir), config
    )
    trace_manifests = _load_phase_barrier(
        run_dir,
        phase="trace",
        all_tasks=trace_tasks,
        instances=instances,
        shard_dir=trace_dir,
    )
    summary["distributed_execution"] = {
        "shard_count": S06_SHARD_COUNT,
        "assignment_unit": "base_graph_lineage",
        "main_shard_manifest_count": len(main_manifests),
        "trace_shard_manifest_count": len(trace_manifests),
        "resource_runtime_identity_match": True,
    }
    summary["trace_replays"] = {
        "triggered_instance_solver_pairs": len(trace_tasks) // len(MAIN_METHODS),
        "expected_tasks": len(trace_tasks),
        "observed_shards": len(_read_shards(trace_tasks, trace_dir)),
        "gate_relevant": False,
    }
    atomic_write_json(resolve_path(args.summary_output), summary)
    manifest = {
        "schema_version": 1,
        "stage": "S06",
        "experiment_id": config["experiment_id"],
        "protocol_file_sha256": S06_CONFIG_FILE_SHA256,
        "instance_manifest_sha256": S06_INSTANCES_FILE_SHA256,
        "status": "completed",
        "finished_at_utc": utc_now(),
        "summary_output": str(resolve_path(args.summary_output).relative_to(REPO)),
        "scientific_gate_pass": summary["gate"]["overall_pass"],
        "main_task_ids": [task.task_id for task in main_tasks],
        "strong_diagnostic_task_ids": [task.task_id for task in strong_tasks],
        "trace_task_ids": [task.task_id for task in trace_tasks],
        "failed_or_skipped_tasks_must_remain": True,
        "test_and_final_accessed": False,
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    print(json.dumps(summary["gate"], sort_keys=True), flush=True)
    return 0 if summary["gate"]["overall_pass"] else 2


def main() -> int:
    args = parse_args()
    if args.max_workers != S06_WORKERS_PER_SHARD:
        raise SystemExit(f"--max-workers must remain {S06_WORKERS_PER_SHARD}")
    if args.validate_only:
        config = load_s06_config(resolve_path(args.config), require_activation=False)
        instances = load_s06_instances()
        main_tasks, strong_tasks = expand_s06_tasks(config, instances)
        print(json.dumps({
            "protocol_file_sha256": S06_CONFIG_FILE_SHA256,
            "instances": len(instances),
            "main_tasks": len(main_tasks),
            "strong_diagnostic_tasks": len(strong_tasks),
            "formal_execution_authorized": False,
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
    config = load_s06_config(resolve_path(args.config), require_activation=True)
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
        return _aggregate(args, config, instances, main_tasks, strong_tasks, run_dir)
    return _run_phase(args, config, instances, main_tasks, strong_tasks, run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
