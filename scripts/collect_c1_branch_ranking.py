#!/usr/bin/env python3
"""C1 parallel collector: expert ranking states on train-split only."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from rl_branching.ranking.sb_ranking_collection import RankingJob, run_ranking_job

FORBIDDEN_SPLITS = {"test", "transfer"}


def _load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text()) or {}
    required = ("output_dir", "instances", "seeds", "policies")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"missing keys in collect config: {missing}")
    return raw


def _assert_train_only(instances: list[str]) -> None:
    for item in instances:
        path = Path(item)
        parts = {p.lower() for p in path.parts}
        if parts & FORBIDDEN_SPLITS:
            raise ValueError(
                f"C1 forbids test/transfer instances, got: {item}. "
                "Use only data/instances/train/..."
            )
        if "validation" in parts or "val" in parts:
            raise ValueError(
                f"C1 ranking collection must not use validation for labels: {item}"
            )


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    job = RankingJob(
        instance=payload["instance"],
        seed=int(payload["seed"]),
        worker_id=int(payload["worker_id"]),
        policy=str(payload["policy"]),
    )
    return run_ranking_job(
        job,
        output_dir=Path(payload["output_dir"]),
        time_limit=float(payload["time_limit"]),
        node_limit=int(payload["node_limit"]),
        protocol=str(payload["protocol"]),
        max_decisions=int(payload["max_decisions"]),
        epsilon=float(payload["epsilon"]),
        store_graph=bool(payload["store_graph"]),
        max_depth_record=int(payload["max_depth_record"]),
        teacher_mode=str(payload["teacher_mode"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shards").mkdir(parents=True, exist_ok=True)
    (output_dir / "collect_config.yaml").write_text(args.config.read_text())

    workers = int(args.workers or config.get("workers", 4))
    time_limit = float(config.get("time_limit", 180.0))
    node_limit = int(config.get("node_limit", 100))
    protocol = str(config.get("protocol", "production-scip"))
    max_decisions = int(config.get("max_decisions_per_episode", 100))
    epsilon = float(config.get("epsilon", 0.3))
    store_graph = bool(config.get("store_graph", False))
    max_depth_record = int(config.get("max_depth_record", 12))
    teacher_mode = str(config.get("teacher_mode", "auto"))
    instances = [str(item) for item in config["instances"]]
    seeds = [int(seed) for seed in config["seeds"]]
    policies = [str(p) for p in config["policies"]]
    _assert_train_only(instances)

    jobs: list[dict[str, Any]] = []
    index = 0
    for instance in instances:
        for seed in seeds:
            for policy in policies:
                shard = (
                    output_dir
                    / "shards"
                    / (
                        f"worker{index % workers}__{Path(instance).stem}"
                        f"__seed{seed}__{policy}.pkl"
                    )
                )
                if args.resume and shard.is_file():
                    index += 1
                    continue
                jobs.append(
                    {
                        "instance": instance,
                        "seed": seed,
                        "policy": policy,
                        "worker_id": index % workers,
                        "output_dir": str(output_dir),
                        "time_limit": time_limit,
                        "node_limit": node_limit,
                        "protocol": protocol,
                        "max_decisions": max_decisions,
                        "epsilon": epsilon,
                        "store_graph": store_graph,
                        "max_depth_record": max_depth_record,
                        "teacher_mode": teacher_mode,
                    }
                )
                index += 1

    print(
        f"[c1-collect] instances={len(instances)} seeds={len(seeds)} "
        f"policies={policies} teacher={teacher_mode} pending_jobs={len(jobs)} "
        f"workers={workers}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    if jobs:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"[c1-collect] {result.get('instance')} "
                    f"seed={result.get('seed')} policy={result.get('policy')} "
                    f"samples={result.get('n_samples')} status={result.get('status')} "
                    f"wall={result.get('wall_time', 0):.1f}s"
                    + (f" error={result['error']}" if result.get("error") else ""),
                    flush=True,
                )

    shard_paths = sorted((output_dir / "shards").glob("*.pkl"))
    total_samples = 0
    summaries = []
    for path in shard_paths:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        summary = payload.get("summary", {})
        summaries.append({"shard": str(path), **summary})
        total_samples += int(summary.get("n_samples", 0))

    manifest = {
        "phase": "C1",
        "format": "c1_branch_ranking_v1",
        "output_dir": str(output_dir),
        "n_shards": len(shard_paths),
        "n_samples": total_samples,
        "workers": workers,
        "time_limit": time_limit,
        "node_limit": node_limit,
        "protocol": protocol,
        "max_decisions_per_episode": max_decisions,
        "epsilon": epsilon,
        "store_graph": store_graph,
        "max_depth_record": max_depth_record,
        "teacher_mode": teacher_mode,
        "instances": instances,
        "seeds": seeds,
        "policies": policies,
        "summaries": summaries,
        "new_results": results,
        "gate_30k": bool(total_samples >= 30000),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "n_shards": len(shard_paths),
                "n_samples": total_samples,
                "gate_30k": total_samples >= 30000,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
