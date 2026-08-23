#!/usr/bin/env python3
"""Run resumable phase-8 SCIP branching experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RESULT_FIELDS = (
    "instance_id",
    "split",
    "size",
    "protocol",
    "method",
    "solver_method",
    "seed",
    "status",
    "objective",
    "primal_bound",
    "dual_bound",
    "gap",
    "wall_time",
    "presolve_time",
    "solve_time",
    "solve_time_after_presolve",
    "nodes",
    "lp_iterations",
    "primal_dual_integral",
    "first_solution_time",
    "branch_decisions",
    "rl_inference_total",
    "rl_inference_mean",
    "rl_inference_max",
    "fallback_count",
    "illegal_actions",
    "legality_checks",
    "n_vars",
    "n_int_vars",
    "n_constraints",
    "active_branching_rule",
    "node_selection_rule",
    "threads",
    "time_limit",
    "node_limit",
    "has_solution",
    "solution_feasible",
    "return_code",
    "json_path",
    "log_path",
    "branch_log_path",
)


@dataclass(frozen=True)
class Job:
    index: int
    protocol: str
    instance: dict[str, Any]
    method: dict[str, Any]
    seed: int
    json_path: Path
    log_path: Path
    branch_log_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "binary",
        "output_dir",
        "time_limit",
        "node_limit",
        "threads",
        "workers",
        "seeds",
        "protocols",
        "instances",
        "methods",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"experiment config is missing {sorted(missing)}")
    if config["threads"] != 1:
        raise ValueError("formal comparisons require threads=1")
    if not config["seeds"] or not config["instances"] or not config["methods"]:
        raise ValueError("seeds, instances, and methods must be non-empty")
    valid_protocols = {"project-production-v1"}
    if not set(config["protocols"]).issubset(valid_protocols):
        raise ValueError("unsupported experiment protocol; only project-production-v1 is allowed")
    return config


def make_jobs(config: dict[str, Any], output_dir: Path) -> list[Job]:
    jobs: list[Job] = []
    raw_dir = output_dir / "raw"
    for protocol in config["protocols"]:
        for instance in config["instances"]:
            for method in config["methods"]:
                allowed = method.get("instances")
                if allowed is not None and instance["instance_id"] not in allowed:
                    continue
                for seed in config["seeds"]:
                    stem = (
                        f"{protocol}__{instance['instance_id']}__"
                        f"{method['name']}__seed{int(seed)}"
                    )
                    directory = raw_dir / protocol / instance["instance_id"]
                    branch_log = (
                        directory / f"{stem}.branches.csv"
                        if method["branching"].startswith("rl-")
                        else None
                    )
                    jobs.append(
                        Job(
                            index=len(jobs),
                            protocol=protocol,
                            instance=instance,
                            method=method,
                            seed=int(seed),
                            json_path=directory / f"{stem}.json",
                            log_path=directory / f"{stem}.log",
                            branch_log_path=branch_log,
                        )
                    )
    return jobs


def job_time_limit(job: Job, config: dict[str, Any]) -> float:
    """Prefer per-instance time_limit, fall back to global config."""
    value = job.instance.get("time_limit", config["time_limit"])
    return float(value)


def command_for(job: Job, config: dict[str, Any], binary: Path) -> list[str]:
    command = [
        str(binary),
        "--instance-id",
        str(job.instance["cli_id"]),
        "--scip-profile",
        str(config.get("scip_profile", "configs/scip/project-production-v1.set")),
        "--branching",
        job.method["branching"],
        "--seed",
        str(job.seed),
        "--time-limit",
        str(job_time_limit(job, config)),
        "--node-limit",
        str(config["node_limit"]),
        "--threads",
        str(config["threads"]),
        "--output-json",
        str(job.json_path),
    ]
    if job.method["branching"].startswith("rl-"):
        command.extend(
            [
                "--rl-model",
                job.method["model"],
                "--rl-device",
                job.method.get("device", "cpu"),
                "--rl-fallback",
                job.method.get("fallback", "relpscost"),
                "--rl-max-depth",
                str(job.method.get("max_depth", -1)),
                "--rl-min-candidates",
                str(job.method.get("min_candidates", 1)),
                "--rl-log",
                str(job.branch_log_path),
            ]
        )
        if any(
            key in job.method
            for key in (
                "prim_lambda",
                "prim_min_depth",
                "prim_require_grown",
                "prim_features",
                "bias_mode",
            )
        ):
            raise ValueError("legacy Prim bias switches have been removed from the official runner")
    return command


def persistent_failure(job: Job, config: dict[str, Any], status: str, message: str) -> None:
    payload = {
        "instance_id": job.instance["cli_id"],
        "method": job.method["branching"],
        "protocol": job.protocol,
        "seed": job.seed,
        "status": status,
        "time_limit": job_time_limit(job, config),
        "node_limit": config["node_limit"],
        "runner_error": message,
    }
    job.json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_cuda_devices() -> list[str]:
    raw = os.environ.get("PHASE8_CUDA_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES") or "0"
    devices = [item.strip() for item in raw.split(",") if item.strip() != ""]
    return devices or ["0"]


def execute_job(
    job: Job,
    config: dict[str, Any],
    binary: Path,
    resume: bool,
    cuda_locks: dict[str, threading.Lock],
    cuda_devices: list[str],
    cuda_counter: list[int],
    cuda_counter_lock: threading.Lock,
    large_semaphore: threading.Semaphore | None,
) -> tuple[Job, int]:
    job.json_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and job.json_path.is_file():
        return job, 0
    command = command_for(job, config, binary)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    assigned_device: str | None = None
    if job.method.get("device") == "cuda":
        with cuda_counter_lock:
            assigned_device = cuda_devices[cuda_counter[0] % len(cuda_devices)]
            cuda_counter[0] += 1
        environment["CUDA_VISIBLE_DEVICES"] = assigned_device
    limit = job_time_limit(job, config)
    timeout = limit + float(config.get("process_grace_seconds", 300.0))
    use_large_slot = large_semaphore is not None and job.instance.get("size") == "large"

    def run() -> int:
        with job.log_path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=environment,
                timeout=timeout,
                check=False,
            )
        return int(completed.returncode)

    def run_with_device() -> int:
        if assigned_device is None:
            return run()
        with cuda_locks[assigned_device]:
            return run()

    try:
        if use_large_slot:
            with large_semaphore:
                return_code = run_with_device()
        else:
            return_code = run_with_device()
    except subprocess.TimeoutExpired as error:
        persistent_failure(job, config, "process_timeout", str(error))
        return_code = 124
    except Exception as error:
        persistent_failure(job, config, "process_error", f"{type(error).__name__}: {error}")
        return_code = 1
    if not job.json_path.is_file():
        persistent_failure(job, config, "process_error", f"solver return code {return_code}")
    return job, int(return_code)


def result_row(job: Job, return_code: int) -> dict[str, Any]:
    with job.json_path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    return {
        "instance_id": job.instance["instance_id"],
        "split": job.instance["split"],
        "size": job.instance["size"],
        "protocol": job.protocol,
        "method": job.method["name"],
        "solver_method": raw.get("method", job.method["branching"]),
        "seed": job.seed,
        "status": raw.get("status", "process_error"),
        "objective": raw.get("objective"),
        "primal_bound": raw.get("primal_bound"),
        "dual_bound": raw.get("dual_bound"),
        "gap": raw.get("final_gap"),
        "wall_time": raw.get("wall_clock_time"),
        "presolve_time": raw.get("presolve_time"),
        "solve_time": raw.get("solving_time"),
        "solve_time_after_presolve": raw.get("solve_time_after_presolve"),
        "nodes": raw.get("nodes"),
        "lp_iterations": raw.get("lp_iterations"),
        "primal_dual_integral": raw.get("primal_dual_integral"),
        "first_solution_time": raw.get("first_solution_time"),
        "branch_decisions": raw.get("branch_decisions", 0),
        "rl_inference_total": raw.get("rl_inference_total", 0.0),
        "rl_inference_mean": raw.get("rl_inference_mean", 0.0),
        "rl_inference_max": raw.get("rl_inference_max", 0.0),
        "fallback_count": raw.get("fallback_count", 0),
        "illegal_actions": raw.get("custom_illegal_actions", 0),
        "legality_checks": raw.get("custom_legality_checks", 0),
        "n_vars": raw.get("number_of_variables"),
        "n_int_vars": raw.get("number_of_integer_variables"),
        "n_constraints": raw.get("number_of_constraints"),
        "active_branching_rule": raw.get("active_branching_rule"),
        "node_selection_rule": raw.get("node_selection_rule"),
        "threads": raw.get("threads"),
        "time_limit": raw.get("time_limit"),
        "node_limit": raw.get("node_limit"),
        "has_solution": raw.get("has_solution", False),
        "solution_feasible": raw.get("solution_feasible", False),
        "return_code": return_code,
        "json_path": str(job.json_path),
        "log_path": str(job.log_path),
        "branch_log_path": "" if job.branch_log_path is None else str(job.branch_log_path),
    }


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    binary = Path(config["binary"]).resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    for method in config["methods"]:
        if method["branching"].startswith("rl-") and not Path(method["model"]).is_file():
            raise FileNotFoundError(method["model"])

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs(config, output_dir)
    size_rank = {"small": 0, "medium": 1, "large": 2}
    jobs = sorted(
        jobs,
        key=lambda item: (size_rank.get(str(item.instance.get("size")), 9), item.index),
    )
    jobs = [
        Job(
            index=index,
            protocol=job.protocol,
            instance=job.instance,
            method=job.method,
            seed=job.seed,
            json_path=job.json_path,
            log_path=job.log_path,
            branch_log_path=job.branch_log_path,
        )
        for index, job in enumerate(jobs)
    ]
    (output_dir / "experiment_plan.json").write_text(
        json.dumps(
            {
                "config": str(args.config),
                "jobs": len(jobs),
                "time_limit_default": config["time_limit"],
                "instance_time_limits": {
                    item["instance_id"]: item.get("time_limit", config["time_limit"])
                    for item in config["instances"]
                },
                "seeds": config["seeds"],
                "protocols": config["protocols"],
                "instances": [item["instance_id"] for item in config["instances"]],
                "methods": [item["name"] for item in config["methods"]],
                "max_concurrent_large": config.get("max_concurrent_large"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    workers = args.workers if args.workers is not None else int(config["workers"])
    if workers <= 0:
        raise ValueError("workers must be positive")
    cuda_devices = parse_cuda_devices()
    cuda_locks = {device: threading.Lock() for device in cuda_devices}
    cuda_counter: list[int] = [0]
    cuda_counter_lock = threading.Lock()
    max_large = config.get("max_concurrent_large")
    large_semaphore = (
        threading.Semaphore(int(max_large)) if max_large is not None else None
    )
    print(
        f"phase8 runner: jobs={len(jobs)} workers={workers} "
        f"cuda_devices={','.join(cuda_devices)} "
        f"default_time_limit={config['time_limit']} "
        f"max_concurrent_large={max_large}",
        flush=True,
    )
    completed: dict[int, tuple[Job, int]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                execute_job,
                job,
                config,
                binary,
                args.resume,
                cuda_locks,
                cuda_devices,
                cuda_counter,
                cuda_counter_lock,
                large_semaphore,
            ): job
            for job in jobs
        }
        for count, future in enumerate(as_completed(futures), start=1):
            job, return_code = future.result()
            completed[job.index] = (job, return_code)
            rows = [
                result_row(*completed[index])
                for index in sorted(completed)
            ]
            write_results(output_dir / "raw_results.csv", rows)
            print(
                f"[{count}/{len(jobs)}] {job.protocol} {job.instance['instance_id']} "
                f"{job.method['name']} seed={job.seed} "
                f"tl={job_time_limit(job, config):.0f}s rc={return_code}",
                flush=True,
            )

    failures = [item for item in completed.values() if item[1] != 0]
    print(f"completed={len(completed)} failures={len(failures)}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
