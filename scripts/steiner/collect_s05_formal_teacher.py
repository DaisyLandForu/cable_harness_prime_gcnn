#!/usr/bin/env python3
"""Collect and freeze the audited S05 formal teacher dataset."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO / "python"
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from collect_s05_teacher import (  # noqa: E402
    _load_envelope,
    collect_task,
    load_valid_task,
    require_audited_tag,
    task_envelope_path,
    utc_now,
)
from steiner_branching.learning.formal_protocol import (  # noqa: E402
    FORMAL_AUDIT_RECORD_PATH,
    FORMAL_CONFIG_FILE_SHA256,
    FORMAL_CONFIG_PATH,
    FormalTaskPlan,
    expand_formal_tasks,
    formal_config_sha256,
    load_s05_formal_config,
)
from steiner_branching.learning.imitation import atomic_write_json  # noqa: E402
from steiner_branching.learning.teacher_data import (  # noqa: E402
    EXPECTED_STACK_ID,
    TeacherSample,
    file_sha256,
    load_teacher_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(FORMAL_CONFIG_PATH))
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-one", help=argparse.SUPPRESS)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def cpu_resource_preflight(config: dict[str, Any]) -> dict[str, Any]:
    cpu_count = len(os.sched_getaffinity(0))
    mem_total_kib = next(
        int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemTotal:")
    )
    ram_gib = mem_total_kib / 1024**2
    required_cpu = int(config["teacher"]["required_cpu_cores"])
    required_ram = int(config["teacher"]["required_ram_gib"])
    if cpu_count < required_cpu or ram_gib < required_ram:
        raise RuntimeError(
            f"formal teacher resources insufficient: cpus={cpu_count}/{required_cpu}, "
            f"ram_gib={ram_gib:.3f}/{required_ram}"
        )
    return {
        "cpu_affinity_count": cpu_count,
        "ram_total_gib": ram_gib,
        "workers": int(config["teacher"]["workers"]),
    }


def invoke_task(task_id: str, config_path: Path) -> tuple[str, int, str]:
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path), "--run-one", task_id],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    return task_id, process.returncode, (process.stdout + process.stderr)[-4000:]


def _sample_record(
    *, sample: TeacherSample, item: dict[str, Any], plan: FormalTaskPlan
) -> dict[str, Any]:
    if (
        sample.task_id != plan.task.task_id
        or sample.split != plan.task.split
        or sample.family != plan.task.family
        or sample.teacher_seed != plan.task.teacher_seed
    ):
        raise ValueError(f"formal shard/task identity mismatch: {sample.task_id}")
    return {
        "path": item["path"],
        "file_sha256": item["file_sha256"],
        "semantic_sha256": item["semantic_sha256"],
        "task_id": sample.task_id,
        "role": plan.role,
        "split": sample.split,
        "family": sample.family,
        "bucket_id": plan.task.bucket_id,
        "graph_sha256": sample.graph_sha256,
        "teacher_seed": sample.teacher_seed,
        "state_index": sample.state_index,
        "state_valid": sample.labels.state_valid,
        "all_tie": bool(item["all_tie"]),
        "candidate_count": sample.state.candidate_count,
    }


def _quota_for(config: dict[str, Any], key: tuple[str, str, str | None]) -> int:
    role, family, bucket = key
    selection = config["state_selection"]
    if role == "train":
        return int(selection["train_quotas"][family][str(bucket)])
    if role == "validation_select":
        return int(selection["validation_select_quota_per_family"])
    if role == "validation_gate":
        return int(selection["validation_gate_quota_per_family"])
    raise ValueError(f"unknown formal role: {role}")


def select_formal_records(
    config: dict[str, Any], records: list[dict[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    graph_roles: dict[str, str] = {}
    semantic_roles: dict[str, str] = {}
    grouped: dict[tuple[str, str, str | None], list[dict[str, Any]]] = {}
    duplicates: list[dict[str, Any]] = []
    seen_semantic: set[str] = set()
    for record in records:
        previous_role = graph_roles.setdefault(record["graph_sha256"], record["role"])
        if previous_role != record["role"]:
            raise ValueError("base graph lineage crosses formal roles")
        semantic = str(record["semantic_sha256"])
        previous_semantic_role = semantic_roles.setdefault(semantic, record["role"])
        if previous_semantic_role != record["role"]:
            raise ValueError("semantic teacher state crosses formal roles")
        if not record["state_valid"]:
            continue
        if semantic in seen_semantic:
            duplicates.append(record)
            continue
        seen_semantic.add(semantic)
        key = (
            record["role"],
            record["family"],
            record["bucket_id"] if record["role"] == "train" else None,
        )
        grouped.setdefault(key, []).append(record)

    selected = {role: [] for role in ("train", "validation_select", "validation_gate")}
    expected_keys: list[tuple[str, str, str | None]] = []
    for family, buckets in config["state_selection"]["train_quotas"].items():
        expected_keys.extend(("train", family, bucket) for bucket in buckets)
    for role in ("validation_select", "validation_gate"):
        expected_keys.extend((role, family, None) for family in config["state_selection"]["train_quotas"])
    shortages: list[str] = []
    for key in expected_keys:
        eligible = sorted(
            grouped.get(key, []),
            key=lambda value: (
                int(value["state_index"]),
                str(value["graph_sha256"]),
                int(value["teacher_seed"]),
                str(value["semantic_sha256"]),
            ),
        )
        quota = _quota_for(config, key)
        if len(eligible) < quota:
            shortages.append(f"{key}: {len(eligible)}/{quota}")
        selected[key[0]].extend(eligible[:quota])
    for role in selected:
        selected[role].sort(
            key=lambda value: (
                str(value["family"]), str(value["bucket_id"]),
                int(value["state_index"]), str(value["graph_sha256"]),
                int(value["teacher_seed"]), str(value["semantic_sha256"]),
            )
        )
    return selected, [{"shortage": value} for value in shortages] + duplicates


def aggregate_formal_manifest(
    *,
    config: dict[str, Any],
    config_path: Path,
    config_digest: str,
    plans: tuple[FormalTaskPlan, ...],
    run_root: Path,
    audited_tag_target: str,
    git_commit: str,
    resource_preflight: dict[str, Any],
) -> dict[str, Any]:
    plan_by_id = {plan.task.task_id: plan for plan in plans}
    envelopes = [
        _load_envelope(task_envelope_path(run_root, plan.task))
        or {"status": "missing", "shards": []}
        for plan in plans
    ]
    records: list[dict[str, Any]] = []
    valid_states = all_tie = candidates_observed = candidates_mapped = 0
    for plan, envelope in zip(plans, envelopes):
        for item in envelope.get("shards", []):
            sample = load_teacher_sample(
                run_root / item["path"], expected_file_sha256=item["file_sha256"]
            )
            records.append(_sample_record(sample=sample, item=item, plan=plan_by_id[sample.task_id]))
            valid_states += int(sample.labels.state_valid)
            all_tie += int(sample.labels.state_valid and bool(item["all_tie"]))
            candidates_observed += sample.state.candidate_count
            candidates_mapped += len(sample.state.candidate_edge_ids)
    selected, selection_issues = select_formal_records(config, records)
    expected_states = int(config["gate"]["expected_max_states"])
    statuses = [envelope.get("status", "missing") for envelope in envelopes]
    terminal = all(status in {"completed", "root_solved"} for status in statuses)
    failures = [
        plan.task.task_id for plan, status in zip(plans, statuses)
        if status not in {"completed", "root_solved"}
    ]
    expected_selected = config["state_selection"]["selected_state_counts"]
    selection_counts = {role: len(values) for role, values in selected.items()}
    valid_fraction = valid_states / expected_states
    tie_fraction = all_tie / valid_states if valid_states else 1.0
    mapping_rate = candidates_mapped / candidates_observed if candidates_observed else 0.0
    checks = {
        "all_tasks_terminal": terminal,
        "zero_task_failures": not failures,
        "teacher_valid_fraction": valid_fraction >= float(config["gate"]["min_teacher_valid_state_fraction"]),
        "teacher_all_tie_fraction": tie_fraction <= float(config["gate"]["max_teacher_all_tie_valid_state_fraction"]),
        "action_mapping_rate": mapping_rate == float(config["gate"]["required_action_mapping_rate"]),
        "all_state_quotas_met": selection_counts == expected_selected and not any("shortage" in issue for issue in selection_issues),
        "zero_split_or_role_leakage": True,
    }
    gate_pass = all(checks.values())
    return {
        "schema_version": 2,
        "stage": "S05",
        "experiment_id": config["experiment_id"],
        "config_path": str(config_path.relative_to(REPO)),
        "config_sha256": config_digest,
        "protocol_yaml_file_sha256": FORMAL_CONFIG_FILE_SHA256,
        "protocol_audit_record": str(FORMAL_AUDIT_RECORD_PATH.relative_to(REPO)),
        "protocol_audit_record_sha256": file_sha256(FORMAL_AUDIT_RECORD_PATH),
        "git_commit": git_commit,
        "required_s04_audited_tag": config["required_s04_audited_tag"],
        "s04_audited_tag_target": audited_tag_target,
        "resource_preflight": resource_preflight,
        "created_at_utc": utc_now(),
        "status": "completed" if gate_pass else "failed",
        "teacher_gate": {"status": "PASS" if gate_pass else "FAIL", "checks": checks},
        "task_ids": [plan.task.task_id for plan in plans],
        "task_roles": {plan.task.task_id: plan.role for plan in plans},
        "task_statuses": {plan.task.task_id: status for plan, status in zip(plans, statuses)},
        "failures": failures,
        "shards": records,
        "selection": selected,
        "selection_issues": selection_issues,
        "counts": {
            "base_graphs": 105,
            "tasks": len(plans),
            "expected_states": expected_states,
            "observed_states": len(records),
            "valid_states": valid_states,
            "all_tie_valid_states": all_tie,
            "candidates_observed": candidates_observed,
            "candidates_mapped": candidates_mapped,
            "selected_states": selection_counts,
        },
        "diagnostics": {
            "valid_state_fraction_of_expected": valid_fraction,
            "all_tie_fraction_of_valid": tie_fraction,
            "action_mapping_rate": mapping_rate,
            "split_leakage_count": 0,
            "role_leakage_count": 0,
        },
        "formal_gate_evaluated": False,
    }


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_s05_formal_config(config_path, require_activation=True)
    digest = formal_config_sha256(config)
    plans = expand_formal_tasks(config)
    if args.dry_run:
        print(json.dumps({
            "config_file_sha256": file_sha256(config_path),
            "config_sha256": digest,
            "tasks": [{"role": plan.role, **plan.task.to_dict()} for plan in plans],
        }, indent=2))
        return 0
    if os.environ.get("STEINER_SOLVER_STACK_ID") != EXPECTED_STACK_ID:
        raise SystemExit("run through scripts/steiner/run_with_scip804.sh --python")
    if not 1 <= args.max_workers <= int(config["teacher"]["workers"]):
        raise SystemExit("--max-workers must be in 1..6")
    resource_preflight = cpu_resource_preflight(config)
    audited_target = require_audited_tag(str(config["required_s04_audited_tag"]))
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=True
    ).stdout.strip()
    run_root = resolve_path(config["artifacts"]["raw_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    plan_by_id = {plan.task.task_id: plan for plan in plans}
    if args.run_one:
        plan = plan_by_id.get(args.run_one)
        if plan is None:
            raise SystemExit(f"unknown S05 formal task: {args.run_one}")
        envelope = collect_task(plan.task, config, digest, run_root, git_commit)
        print(f"S05 FORMAL WRITE status={envelope['status']} task={plan.task.task_id}", flush=True)
        return 0 if envelope["status"] in {"completed", "root_solved"} else 1
    pending = [
        plan for plan in plans
        if load_valid_task(
            task_envelope_path(run_root, plan.task), plan.task, digest, run_root, git_commit
        ) is None
    ]
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(invoke_task, plan.task.task_id, config_path): plan for plan in pending
        }
        for future in as_completed(futures):
            task_id, returncode, tail = future.result()
            print(tail, end="" if tail.endswith("\n") else "\n", flush=True)
            if returncode != 0:
                failures.append(task_id)
    manifest = aggregate_formal_manifest(
        config=config,
        config_path=config_path,
        config_digest=digest,
        plans=plans,
        run_root=run_root,
        audited_tag_target=audited_target,
        git_commit=git_commit,
        resource_preflight=resource_preflight,
    )
    atomic_write_json(run_root / "manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"], "teacher_gate": manifest["teacher_gate"],
        "counts": manifest["counts"], "failures": failures,
    }, sort_keys=True))
    return 0 if manifest["status"] == "completed" and not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
