#!/usr/bin/env python3
"""Collect/resume checksum-addressed S05 strong-branch teacher shards."""

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

import numpy as np


REPO = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from steiner_branching.data.generate import GeneratorConfig, generate_graph  # noqa: E402
from steiner_branching.learning.imitation import atomic_write_json  # noqa: E402
from steiner_branching.learning.teacher_data import (  # noqa: E402
    EXPECTED_STACK_ID,
    TeacherSample,
    TeacherTask,
    assert_no_split_leakage,
    expand_pilot_tasks,
    immutable_array,
    load_s05_config,
    load_teacher_sample,
    s05_config_sha256,
    task_sha256,
    write_teacher_sample,
)
from steiner_branching.milp.mcf import build_mcf  # noqa: E402
from steiner_branching.solver.bipartite_observation import (  # noqa: E402
    SteinerNodeBipartite,
    with_legal_edge_actions,
)
from steiner_branching.solver.branchability import configure_p1  # noqa: E402
from steiner_branching.solver.strong_branching import (  # noqa: E402
    OrderedStrongBranchObservation,
    StrongBranchingTeacher,
    align_labels_to_probindices,
)


DEFAULT_CONFIG = REPO / "configs/steiner/experiments/s05_teacher_il_pilot_v3.yml"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-one", help=argparse.SUPPRESS)
    return parser.parse_args()


def require_audited_tag(tag: str) -> str:
    process = subprocess.run(
        ["git", "rev-parse", f"refs/tags/{tag}^{{}}"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"required audited tag {tag!r} is missing; S05 collection remains blocked"
        )
    target = process.stdout.strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", target, "HEAD"], cwd=REPO, check=False
    )
    if ancestor.returncode != 0:
        raise RuntimeError(f"required audited tag {tag!r} is not in the current history")
    return target


def task_envelope_path(run_root: Path, task: TeacherTask) -> Path:
    return run_root / "tasks" / task.task_id / "task.json"


def _load_envelope(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def load_valid_task(
    path: Path, task: TeacherTask, config_digest: str, run_root: Path, git_commit: str
) -> dict[str, Any] | None:
    envelope = _load_envelope(path)
    if envelope is None or envelope.get("status") not in {"completed", "root_solved"}:
        return None
    if envelope.get("config_sha256") != config_digest:
        return None
    if envelope.get("git_commit") != git_commit:
        return None
    if envelope.get("task_sha256") != task_sha256(task, config_digest):
        return None
    if envelope.get("task") != task.to_dict() or not isinstance(envelope.get("shards"), list):
        return None
    if envelope.get("states_observed") != len(envelope["shards"]):
        return None
    seen_state_indices: set[int] = set()
    try:
        for item in envelope["shards"]:
            shard = run_root / item["path"]
            sample = load_teacher_sample(shard, expected_file_sha256=item["file_sha256"])
            if (
                sample.semantic_sha256 != item["semantic_sha256"]
                or sample.task_id != task.task_id
                or sample.split != task.split
                or sample.family != task.family
                or sample.teacher_seed != task.teacher_seed
                or sample.graph_sha256 != envelope.get("graph_sha256")
                or sample.state_index in seen_state_indices
            ):
                return None
            seen_state_indices.add(sample.state_index)
        if seen_state_indices != set(range(len(envelope["shards"]))):
            return None
    except (KeyError, OSError, ValueError):
        return None
    return envelope


def _choose_action(sample: TeacherSample) -> tuple[int, str]:
    valid = sample.labels.fully_valid
    if bool(valid.any()):
        masked = np.where(valid, sample.labels.scores, -np.inf)
        return int(sample.state.candidate_indices[int(np.argmax(masked))]), "strong"
    fractions = sample.state.variable_features[sample.state.candidate_indices, 9]
    selected = int(np.argmin(np.abs(fractions - 0.5)))
    return int(sample.state.candidate_indices[selected]), "most_infeasible_fallback"


def collect_task(
    task: TeacherTask,
    config: dict[str, Any],
    config_digest: str,
    run_root: Path,
    git_commit: str,
) -> dict[str, Any]:
    envelope_path = task_envelope_path(run_root, task)
    cached = load_valid_task(envelope_path, task, config_digest, run_root, git_commit)
    if cached is not None:
        return cached
    task_dir = envelope_path.parent
    shard_dir = task_dir / "states"
    shard_dir.mkdir(parents=True, exist_ok=True)
    envelope: dict[str, Any] = {
        "schema_version": 1,
        "stage": "S05",
        "config_sha256": config_digest,
        "git_commit": git_commit,
        "task_sha256": task_sha256(task, config_digest),
        "task": task.to_dict(),
        "started_at_utc": utc_now(),
        "status": "running",
        "shards": [],
        "trajectory_actions": [],
    }
    atomic_write_json(envelope_path, envelope)
    try:
        graph = generate_graph(
            GeneratorConfig(
                family=task.family,
                n_nodes=task.n_nodes,
                n_terminals=task.n_terminals,
                seed=task.generator_seed,
            )
        )
        build = build_mcf(graph, configure_correctness_profile=False, hide_output=True)
        runtime_config = dict(config)
        runtime_config["solver_seed"] = task.teacher_seed
        effective_parameters = configure_p1(build.model, runtime_config, "relpscost")
        teacher_config = config["teacher"]
        observations = OrderedStrongBranchObservation(
            SteinerNodeBipartite(),
            StrongBranchingTeacher(
                iteration_limit=int(teacher_config["iteration_limit_per_candidate"]),
                candidate_limit=int(teacher_config["candidate_limit"]),
                idempotent=bool(teacher_config["idempotent"]),
                infeasible_gain_cap=float(teacher_config["infeasible_gain_cap"]),
            ),
        )
        import ecole

        environment = ecole.environment.Branching(
            observation_function=observations, pseudo_candidates=False
        )
        environment.seed(task.teacher_seed)
        observation, action_set, _reward, done, _info = environment.reset(
            ecole.scip.Model.from_pyscipopt(build.model)
        )
        state_index = 0
        mapped_candidates = 0
        observed_candidates = 0
        invalid_children = 0
        while not done and state_index < int(teacher_config["max_states_per_task"]):
            if observation is None or action_set is None:
                raise RuntimeError("branch state omitted observation or legal actions")
            state = with_legal_edge_actions(observation["state"], action_set, build.metadata)
            labels = align_labels_to_probindices(observation["teacher"], action_set)
            pseudocosts = np.asarray(observation["pseudocosts"], dtype=np.float64)
            if pseudocosts.ndim != 1 or state.candidate_indices.size == 0:
                raise RuntimeError("pseudocost observation/action set is malformed")
            if int(state.candidate_indices.max()) >= pseudocosts.size:
                raise RuntimeError("pseudocost rows do not cover legal actions")
            sample = TeacherSample(
                task_id=task.task_id,
                split=task.split,
                family=task.family,
                graph_sha256=graph.graph_sha256,
                teacher_seed=task.teacher_seed,
                state_index=state_index,
                state=state,
                labels=labels,
                pseudocost_scores=immutable_array(
                    pseudocosts[state.candidate_indices], np.float64
                ),
            )
            shard_path = shard_dir / f"state-{state_index:05d}.npz"
            hashes = write_teacher_sample(shard_path, sample)
            envelope["shards"].append(
                {
                    "path": str(shard_path.relative_to(run_root)),
                    **hashes,
                    "state_valid": sample.labels.state_valid,
                    "all_tie": sample.labels.all_tie(
                        float(teacher_config["tie_relative_tolerance"])
                    ),
                    "candidate_count": state.candidate_count,
                    "invalid_child_candidates": int((~sample.labels.fully_valid).sum()),
                    "teacher_seconds": sample.labels.elapsed_seconds,
                }
            )
            observed_candidates += state.candidate_count
            mapped_candidates += len(state.candidate_edge_ids)
            invalid_children += int((~sample.labels.fully_valid).sum())
            action, source = _choose_action(sample)
            envelope["trajectory_actions"].append(
                {"state_index": state_index, "probindex": action, "source": source}
            )
            state_index += 1
            observation, action_set, _reward, done, _info = environment.step(action)
        envelope.update(
            {
                "status": "root_solved" if state_index == 0 and done else "completed",
                "finished_at_utc": utc_now(),
                "graph_sha256": graph.graph_sha256,
                "metadata_sha256": build.metadata.sha256,
                "states_observed": state_index,
                "max_states_reached": state_index == int(teacher_config["max_states_per_task"]),
                "candidates_observed": observed_candidates,
                "candidates_mapped": mapped_candidates,
                "invalid_child_candidates": invalid_children,
                "effective_parameters": effective_parameters,
            }
        )
    except BaseException as error:
        envelope.update(
            {
                "status": "solver_error",
                "finished_at_utc": utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
    atomic_write_json(envelope_path, envelope)
    return envelope


def aggregate_manifest(
    *,
    config: dict[str, Any],
    config_path: Path,
    config_digest: str,
    tasks: tuple[TeacherTask, ...],
    run_root: Path,
    audited_tag_target: str,
    git_commit: str,
) -> dict[str, Any]:
    envelopes = [
        _load_envelope(task_envelope_path(run_root, task)) or {"status": "missing", "shards": []}
        for task in tasks
    ]
    samples: list[TeacherSample] = []
    shard_records: list[dict[str, Any]] = []
    for envelope in envelopes:
        for item in envelope.get("shards", []):
            sample = load_teacher_sample(
                run_root / item["path"], expected_file_sha256=item["file_sha256"]
            )
            samples.append(sample)
            shard_records.append(dict(item))
    assert_no_split_leakage(samples)
    expected_states = len(tasks) * int(config["teacher"]["max_states_per_task"])
    valid_states = sum(sample.labels.state_valid for sample in samples)
    all_tie = sum(
        sample.labels.state_valid
        and sample.labels.all_tie(float(config["teacher"]["tie_relative_tolerance"]))
        for sample in samples
    )
    candidates = sum(sample.state.candidate_count for sample in samples)
    return {
        "schema_version": 1,
        "stage": "S05",
        "experiment_id": config["experiment_id"],
        "config_path": str(config_path.relative_to(REPO)),
        "config_sha256": config_digest,
        "git_commit": git_commit,
        "required_s04_audited_tag": config["required_s04_audited_tag"],
        "s04_audited_tag_target": audited_tag_target,
        "created_at_utc": utc_now(),
        "status": "completed" if all(envelope.get("status") in {"completed", "root_solved"} for envelope in envelopes) else "failed",
        "task_ids": [task.task_id for task in tasks],
        "task_statuses": {task.task_id: envelope.get("status", "missing") for task, envelope in zip(tasks, envelopes)},
        "shards": shard_records,
        "counts": {
            "tasks": len(tasks),
            "expected_states": expected_states,
            "observed_states": len(samples),
            "valid_states": valid_states,
            "all_tie_valid_states": all_tie,
            "candidates_observed": candidates,
            "candidates_mapped": candidates,
        },
        "diagnostics": {
            "valid_state_fraction_of_expected": valid_states / expected_states,
            "all_tie_fraction_of_valid": all_tie / valid_states if valid_states else 1.0,
            "action_mapping_rate": 1.0 if candidates else 0.0,
            "split_leakage_count": 0,
        },
        "formal_gate_evaluated": False,
    }


def invoke_task(
    task: TeacherTask, *, config_path: Path
) -> tuple[str, int, str]:
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path), "--run-one", task.task_id],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    return task.task_id, process.returncode, (process.stdout + process.stderr)[-4000:]


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_s05_config(config_path)
    digest = s05_config_sha256(config)
    tasks = expand_pilot_tasks(config)
    if args.dry_run:
        print(json.dumps({"config_sha256": digest, "tasks": [task.to_dict() for task in tasks]}, indent=2))
        return 0
    if os.environ.get("STEINER_SOLVER_STACK_ID") != EXPECTED_STACK_ID:
        raise SystemExit("run through scripts/steiner/run_with_scip804.sh --python")
    if not 1 <= args.max_workers <= 6:
        raise SystemExit("--max-workers must be in 1..6")
    audited_target = require_audited_tag(str(config["required_s04_audited_tag"]))
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=True
    ).stdout.strip()
    run_root = resolve_path(config["artifacts"]["raw_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    task_by_id = {task.task_id: task for task in tasks}
    if args.run_one:
        task = task_by_id.get(args.run_one)
        if task is None:
            raise SystemExit(f"unknown S05 task: {args.run_one}")
        envelope = collect_task(task, config, digest, run_root, git_commit)
        print(f"S05 WRITE status={envelope['status']} task={task.task_id}", flush=True)
        return 0 if envelope["status"] in {"completed", "root_solved"} else 1

    pending = [
        task for task in tasks
        if load_valid_task(
            task_envelope_path(run_root, task), task, digest, run_root, git_commit
        ) is None
    ]
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(invoke_task, task, config_path=config_path): task for task in pending}
        for future in as_completed(futures):
            task_id, returncode, tail = future.result()
            print(tail, end="" if tail.endswith("\n") else "\n", flush=True)
            if returncode != 0:
                failures.append(task_id)
    manifest = aggregate_manifest(
        config=config,
        config_path=config_path,
        config_digest=digest,
        tasks=tasks,
        run_root=run_root,
        audited_tag_target=audited_target,
        git_commit=git_commit,
    )
    atomic_write_json(run_root / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"], "failures": failures}, sort_keys=True))
    return 0 if manifest["status"] == "completed" and not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
