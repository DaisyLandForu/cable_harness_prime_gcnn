#!/usr/bin/env python3
"""Collect the audited fresh S05 formal-v3 validation candidate pool."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from steiner_branching.data.generate import SYNTHETIC_FAMILIES  # noqa: E402
from steiner_branching.learning.formal_v3 import (  # noqa: E402
    V3_CONFIG_FILE_SHA256,
    V3_CONFIG_PATH,
    V3_EXPERIMENT_ID,
    V3_RAW_ROOT,
    V3_TEACHER_MANIFEST,
    V3TaskPlan,
    expand_v3_tasks,
    load_v3_protocol,
    require_v3_implementation_identity,
    v3_teacher_runtime_config,
)
from steiner_branching.learning.imitation import atomic_write_json  # noqa: E402
from steiner_branching.learning.teacher_data import (  # noqa: E402
    EXPECTED_STACK_ID,
    file_sha256,
    load_teacher_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-one", help=argparse.SUPPRESS)
    return parser.parse_args()


def cpu_resource_preflight(config: dict[str, Any]) -> dict[str, Any]:
    observed_cpus = len(os.sched_getaffinity(0))
    total_kib = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        if line.startswith("MemTotal:")
    )
    observed_ram = total_kib / 1024**2
    teacher = config["solver_and_teacher"]
    required_cpus = int(teacher["required_cpu_cores"])
    required_ram = int(teacher["required_ram_gib"])
    if observed_cpus < required_cpus or observed_ram < required_ram:
        raise RuntimeError(
            f"formal-v3 resources insufficient: cpus={observed_cpus}/{required_cpus}, "
            f"ram_gib={observed_ram:.3f}/{required_ram}"
        )
    return {
        "cpu_affinity_count": observed_cpus,
        "ram_total_gib": observed_ram,
        "workers": int(teacher["workers"]),
    }


def invoke_task(task_id: str) -> tuple[str, int, str]:
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--run-one", task_id],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    return task_id, process.returncode, (process.stdout + process.stderr)[-4000:]


def _record_for_shard(
    plan: V3TaskPlan, item: dict[str, Any], run_root: Path
) -> dict[str, Any]:
    sample = load_teacher_sample(
        run_root / item["path"], expected_file_sha256=str(item["file_sha256"])
    )
    if any((
        sample.task_id != plan.task.task_id,
        sample.split != "validation_iid",
        sample.family != plan.task.family,
        sample.teacher_seed != plan.task.teacher_seed,
        sample.graph_sha256 != plan.graph_sha256,
        sample.state_index != int(Path(item["path"]).stem.split("-")[-1]),
    )):
        raise ValueError(f"formal-v3 shard identity mismatch: {item['path']}")
    return {
        "path": str(item["path"]),
        "file_sha256": str(item["file_sha256"]),
        "semantic_sha256": sample.semantic_sha256,
        "task_id": sample.task_id,
        "role": "validation_gate_candidate",
        "split": sample.split,
        "family": sample.family,
        "bucket_id": plan.task.bucket_id,
        "candidate_rank": plan.candidate_rank,
        "generator_seed": plan.task.generator_seed,
        "graph_sha256": sample.graph_sha256,
        "teacher_seed": sample.teacher_seed,
        "state_index": sample.state_index,
        "state_valid": sample.labels.state_valid,
        "all_tie": sample.labels.all_tie(1.0e-9),
        "candidate_count": sample.state.candidate_count,
    }


def aggregate_teacher_manifest(
    *,
    config: dict[str, Any],
    plans: tuple[V3TaskPlan, ...],
    run_root: Path,
    git_commit: str,
    audited_tag_target: str,
    resources: dict[str, Any],
) -> dict[str, Any]:
    envelopes = []
    records: list[dict[str, Any]] = []
    graph_candidates: dict[str, dict[str, Any]] = {}
    graph_task_statuses: dict[str, list[str]] = defaultdict(list)
    graph_candidates_observed: Counter[str] = Counter()
    graph_candidates_mapped: Counter[str] = Counter()
    for plan in plans:
        envelope = _load_envelope(task_envelope_path(run_root, plan.task))
        if envelope is None:
            envelope = {"status": "missing", "shards": []}
        envelopes.append(envelope)
        graph_task_statuses[plan.graph_sha256].append(str(envelope.get("status", "missing")))
        graph_candidates_observed[plan.graph_sha256] += int(
            envelope.get("candidates_observed", 0)
        )
        graph_candidates_mapped[plan.graph_sha256] += int(
            envelope.get("candidates_mapped", 0)
        )
        if envelope.get("graph_sha256") not in {None, plan.graph_sha256}:
            raise ValueError("formal-v3 task generated an unexpected graph hash")
        graph_candidates.setdefault(plan.graph_sha256, {
            "family": plan.task.family,
            "bucket_id": plan.task.bucket_id,
            "candidate_rank": plan.candidate_rank,
            "generator_seed": plan.task.generator_seed,
            "graph_sha256": plan.graph_sha256,
        })
        for item in envelope.get("shards", []):
            records.append(_record_for_shard(plan, item, run_root))

    statuses = [str(envelope.get("status", "missing")) for envelope in envelopes]
    terminal_values = {"completed", "root_solved"}
    failures = [
        plan.task.task_id
        for plan, status in zip(plans, statuses)
        if status not in terminal_values
    ]
    semantics = [str(item["semantic_sha256"]) for item in records]
    duplicate_semantics = sorted(
        semantic for semantic, count in Counter(semantics).items() if count != 1
    )
    valid_records = [item for item in records if item["state_valid"]]
    all_tie = sum(bool(item["all_tie"]) for item in valid_records)
    candidates_observed = sum(
        int(envelope.get("candidates_observed", 0)) for envelope in envelopes
    )
    candidates_mapped = sum(
        int(envelope.get("candidates_mapped", 0)) for envelope in envelopes
    )
    expected_states = int(config["fresh_candidate_pool"]["expected_max_states"])
    valid_fraction = len(valid_records) / expected_states
    tie_fraction = all_tie / len(valid_records) if valid_records else 1.0
    mapping_rate = candidates_mapped / candidates_observed if candidates_observed else 0.0

    valid_by_graph: Counter[str] = Counter(
        str(item["graph_sha256"]) for item in valid_records
    )
    eligible = []
    for graph_sha256, candidate in graph_candidates.items():
        task_statuses = graph_task_statuses[graph_sha256]
        observed = graph_candidates_observed[graph_sha256]
        mapped = graph_candidates_mapped[graph_sha256]
        checks = {
            "three_tasks_terminal": len(task_statuses) == 3
            and all(status in terminal_values for status in task_statuses),
            "zero_task_failures": len(task_statuses) == 3
            and all(status in terminal_values for status in task_statuses),
            "minimum_12_unique_valid_states": valid_by_graph[graph_sha256] >= 12,
            "action_mapping_rate": observed > 0 and mapped == observed,
        }
        eligible.append({
            **candidate,
            "task_statuses": task_statuses,
            "valid_unique_states": valid_by_graph[graph_sha256],
            "candidates_observed": observed,
            "candidates_mapped": mapped,
            "checks": checks,
            "eligible": all(checks.values()),
        })
    eligible.sort(key=lambda item: (str(item["family"]), int(item["candidate_rank"])))
    eligible_counts = Counter(
        str(item["family"]) for item in eligible if item["eligible"]
    )
    quality = config["teacher_quality_gate"]
    checks = {
        "all_240_tasks_terminal": len(statuses) == 240
        and all(status in terminal_values for status in statuses),
        "zero_task_failures": not failures,
        "teacher_valid_fraction": valid_fraction
        >= float(quality["min_teacher_valid_state_fraction"]),
        "teacher_all_tie_fraction": tie_fraction
        <= float(quality["max_teacher_all_tie_valid_state_fraction"]),
        "action_mapping_rate": mapping_rate
        == float(quality["required_action_mapping_rate"]),
        "semantic_sha256_unique": not duplicate_semantics,
        "all_candidate_graphs_present": len(graph_candidates) == 80,
        "at_least_six_eligible_lineages_per_family": all(
            eligible_counts[family] >= 6 for family in SYNTHETIC_FAMILIES
        ),
        "zero_split_or_role_leakage": all(
            item["split"] == "validation_iid"
            and item["role"] == "validation_gate_candidate"
            for item in records
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "stage": "S05",
        "experiment_id": V3_EXPERIMENT_ID,
        "status": "completed" if passed else "failed",
        "created_at_utc": utc_now(),
        "implementation_run_head": git_commit,
        "protocol_yaml_sha256": V3_CONFIG_FILE_SHA256,
        "activation_record": "docs/steiner/audits/S05_FORMAL_V3_ACTIVATION_RECORD.json",
        "candidate_manifest": "configs/steiner/experiments/s05_formal_v3_candidate_graphs.json",
        "candidate_manifest_sha256": config["fresh_candidate_pool"]["manifest_sha256"],
        "s04_audited_tag_target": audited_tag_target,
        "resource_preflight": resources,
        "teacher_gate": {"status": "PASS" if passed else "FAIL", "checks": checks},
        "task_ids": [plan.task.task_id for plan in plans],
        "task_statuses": {
            plan.task.task_id: status for plan, status in zip(plans, statuses)
        },
        "failures": failures,
        "duplicate_semantic_sha256": duplicate_semantics,
        "shards": records,
        "lineage_eligibility": eligible,
        "counts": {
            "candidate_graphs": len(graph_candidates),
            "tasks": len(plans),
            "expected_states": expected_states,
            "observed_states": len(records),
            "valid_states": len(valid_records),
            "all_tie_valid_states": all_tie,
            "candidates_observed": candidates_observed,
            "candidates_mapped": candidates_mapped,
            "eligible_lineages_by_family": {
                family: eligible_counts[family] for family in SYNTHETIC_FAMILIES
            },
        },
        "diagnostics": {
            "valid_state_fraction_of_expected": valid_fraction,
            "all_tie_fraction_of_valid": tie_fraction,
            "action_mapping_rate": mapping_rate,
        },
        "selection_performed": False,
        "checkpoint_loaded": False,
        "formal_gate_evaluated": False,
        "s06_authorized": False,
        "test_and_final_accessed": False,
    }


def main() -> int:
    args = parse_args()
    config = load_v3_protocol(require_activation=True)
    plans = expand_v3_tasks(config)
    if args.dry_run:
        print(json.dumps({
            "protocol_yaml_sha256": V3_CONFIG_FILE_SHA256,
            "tasks": [
                {
                    "candidate_rank": plan.candidate_rank,
                    "expected_graph_sha256": plan.graph_sha256,
                    **plan.task.to_dict(),
                }
                for plan in plans
            ],
        }, indent=2))
        return 0
    if os.environ.get("STEINER_SOLVER_STACK_ID") != EXPECTED_STACK_ID:
        raise SystemExit("run through scripts/steiner/run_with_scip804.sh --python")
    max_workers = int(config["solver_and_teacher"]["workers"])
    if not 1 <= args.max_workers <= max_workers:
        raise SystemExit(f"--max-workers must be in 1..{max_workers}")
    if V3_TEACHER_MANIFEST.exists():
        raise SystemExit(f"refusing to overwrite formal-v3 evidence: {V3_TEACHER_MANIFEST}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    require_v3_implementation_identity(head)
    resources = cpu_resource_preflight(config)
    audited_tag_target = require_audited_tag("steiner-s04-audited-v2")
    runtime = v3_teacher_runtime_config(config)
    V3_RAW_ROOT.mkdir(parents=True, exist_ok=True)
    plan_by_id = {plan.task.task_id: plan for plan in plans}
    if args.run_one:
        plan = plan_by_id.get(args.run_one)
        if plan is None:
            raise SystemExit(f"unknown formal-v3 task: {args.run_one}")
        envelope = collect_task(
            plan.task, runtime, V3_CONFIG_FILE_SHA256, V3_RAW_ROOT, head
        )
        if envelope.get("graph_sha256") not in {None, plan.graph_sha256}:
            raise SystemExit("formal-v3 generated graph hash differs from the frozen candidate")
        print(
            f"S05 FORMAL-V3 WRITE status={envelope['status']} task={plan.task.task_id}",
            flush=True,
        )
        return 0 if envelope["status"] in {"completed", "root_solved"} else 1

    pending = [
        plan
        for plan in plans
        if load_valid_task(
            task_envelope_path(V3_RAW_ROOT, plan.task),
            plan.task,
            V3_CONFIG_FILE_SHA256,
            V3_RAW_ROOT,
            head,
        ) is None
    ]
    failures = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(invoke_task, plan.task.task_id): plan for plan in pending
        }
        for future in as_completed(futures):
            task_id, returncode, tail = future.result()
            print(tail, end="" if tail.endswith("\n") else "\n", flush=True)
            if returncode != 0:
                failures.append(task_id)
    manifest = aggregate_teacher_manifest(
        config=config,
        plans=plans,
        run_root=V3_RAW_ROOT,
        git_commit=head,
        audited_tag_target=audited_tag_target,
        resources=resources,
    )
    atomic_write_json(V3_TEACHER_MANIFEST, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "teacher_gate": manifest["teacher_gate"],
        "counts": manifest["counts"],
        "failures": failures,
    }, sort_keys=True))
    return 0 if manifest["status"] == "completed" and not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
