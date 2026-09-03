#!/usr/bin/env python3
"""Run/resume the preregistered S03 worker ramp and formal matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from steiner_branching.solver.branchability import (  # noqa: E402
    EXPECTED_STACK_ID,
    ProbeTask,
    aggregate_results,
    atomic_write_json,
    config_sha256,
    expand_tasks,
    load_s03_config,
    load_valid_shard,
    run_probe_task,
    task_sha256,
)


DEFAULT_CONFIG = REPO / "configs/steiner/experiments/s03_branchability_pilot_v1.yml"
DEFAULT_ARTIFACT_ROOT = REPO / "results/steiner/raw"
DEFAULT_SUMMARY = REPO / "docs/steiner/phases/S03/S03_GATE_SUMMARY.json"
NATIVE_PROBE = REPO / "build/steiner_s03_sb_probe"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--skip-ramp", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--run-one", help=argparse.SUPPRESS)
    return parser.parse_args()


def shard_path(shard_dir: Path, task: ProbeTask) -> Path:
    return shard_dir / f"{task.task_id}.json"


def run_one(
    *, task: ProbeTask, config: dict[str, Any], digest: str, shard_dir: Path
) -> int:
    output = shard_path(shard_dir, task)
    cached = load_valid_shard(output, task, digest)
    if cached is not None:
        print(f"S03 SKIP valid-shard {task.task_id}", flush=True)
        return 0
    envelope: dict[str, Any] = {
        "config_sha256": digest,
        "task_sha256": task_sha256(task, digest),
        "started_at_utc": utc_now(),
        "task": task.to_dict(),
    }
    try:
        envelope.update(
            run_probe_task(task, config, native_probe=NATIVE_PROBE, task_dir=output.with_suffix(""))
        )
    except BaseException as error:
        envelope.update(
            {
                "status": "solver_error",
                "classification": "solver_error",
                "error": f"{type(error).__name__}: {error}",
                "branchability": {
                    "legal_decisions": 0,
                    "candidates_observed": 0,
                    "candidates_mapped": 0,
                    "mapping_failures": [],
                    "callback_errors": [f"{type(error).__name__}: {error}"],
                },
            }
        )
    envelope["finished_at_utc"] = utc_now()
    atomic_write_json(output, envelope)
    print(f"S03 WRITE status={envelope['status']} task={task.task_id}", flush=True)
    return 0 if envelope["status"] != "solver_error" else 1


def launch_wave(
    tasks: list[ProbeTask], *, workers: int, script_args: argparse.Namespace,
    shard_dir: Path, digest: str,
) -> None:
    pending = [task for task in tasks if load_valid_shard(shard_path(shard_dir, task), task, digest) is None]
    print(
        f"S03 WAVE workers={workers} tasks={len(tasks)} pending={len(pending)} "
        f"started={utc_now()}",
        flush=True,
    )
    if not pending:
        return
    command_prefix = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config", str(resolve_path(script_args.config)),
        "--artifact-root", str(resolve_path(script_args.artifact_root)),
        "--summary-output", str(resolve_path(script_args.summary_output)),
    ]

    def invoke(task: ProbeTask) -> tuple[str, int, str]:
        process = subprocess.run(
            command_prefix + ["--run-one", task.task_id],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )
        return task.task_id, process.returncode, (process.stdout + process.stderr)[-4000:]

    failures: list[tuple[str, int, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(invoke, task): task for task in pending}
        for future in as_completed(futures):
            task_id, returncode, tail = future.result()
            print(tail, end="" if tail.endswith("\n") else "\n", flush=True)
            if returncode != 0:
                failures.append((task_id, returncode, tail))
    print(f"S03 WAVE finished={utc_now()} failures={len(failures)}", flush=True)
    # Failure shards remain part of the matrix; only a missing shard aborts orchestration.
    missing = [task.task_id for task in tasks if not shard_path(shard_dir, task).is_file()]
    if missing:
        raise RuntimeError(f"wave ended without shards: {missing}")


def main() -> int:
    args = parse_args()
    if os.environ.get("STEINER_SOLVER_STACK_ID") != EXPECTED_STACK_ID:
        raise SystemExit("run through scripts/steiner/run_with_scip804.sh --python")
    if args.max_workers < 1 or args.max_workers > 6:
        raise SystemExit("--max-workers must be in 1..6")
    config_path = resolve_path(args.config)
    config = load_s03_config(config_path)
    digest = config_sha256(config)
    formal_tasks, ramp_tasks = expand_tasks(config)
    all_tasks = {task.task_id: task for task in formal_tasks + ramp_tasks}
    run_dir = resolve_path(args.artifact_root) / "s03" / str(config["experiment_id"])
    shard_dir = run_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    if args.run_one:
        task = all_tasks.get(args.run_one)
        if task is None:
            raise SystemExit(f"unknown S03 task: {args.run_one}")
        return run_one(task=task, config=config, digest=digest, shard_dir=shard_dir)

    manifest_path = run_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "stage": "S03",
        "experiment_id": config["experiment_id"],
        "config_path": str(config_path.relative_to(REPO)),
        "config_sha256": digest,
        "solver_stack_id": config["solver_stack_id"],
        "split": config["split"],
        "protocol_id": config["protocol_id"],
        "solver_seed": config["solver_seed"],
        "started_at_utc": utc_now(),
        "status": "running",
        "formal_task_ids": [task.task_id for task in formal_tasks],
        "ramp_task_ids": [task.task_id for task in ramp_tasks],
    }
    atomic_write_json(manifest_path, manifest)
    try:
        if not args.aggregate_only:
            if not NATIVE_PROBE.is_file():
                raise RuntimeError(
                    "build/steiner_s03_sb_probe is missing; run "
                    "CONDA_PREFIX=/home/duweiyue25/conda/envs/rl4scip make steiner-s03-probe"
                )
            if not args.skip_ramp:
                offset = 0
                for workers, count in zip(
                    config["worker_ramp"]["workers"],
                    config["worker_ramp"]["tasks_per_wave"],
                ):
                    wave = ramp_tasks[offset : offset + int(count)]
                    offset += int(count)
                    launch_wave(
                        wave, workers=int(workers), script_args=args,
                        shard_dir=shard_dir, digest=digest,
                    )
            launch_wave(
                formal_tasks,
                workers=args.max_workers,
                script_args=args,
                shard_dir=shard_dir,
                digest=digest,
            )
        summary = aggregate_results(config, formal_tasks, ramp_tasks, shard_dir)
        atomic_write_json(resolve_path(args.summary_output), summary)
        manifest["status"] = "completed" if not summary["completion"]["missing_formal_tasks"] else "failed"
        manifest["finished_at_utc"] = utc_now()
        manifest["summary_output"] = str(resolve_path(args.summary_output).relative_to(REPO))
        manifest["gate_pass"] = summary["gate"]["overall_pass"]
        atomic_write_json(manifest_path, manifest)
        print(json.dumps(summary["gate"], sort_keys=True), flush=True)
        return 0 if summary["gate"]["overall_pass"] else 2
    except BaseException:
        manifest["status"] = "failed"
        manifest["finished_at_utc"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
