#!/usr/bin/env python3
"""Create the audited S05 formal-v2 640-state view from sealed v1 evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from steiner_branching.config import StrictConfigError, load_yaml_mapping  # noqa: E402
from steiner_branching.learning.formal_protocol import (  # noqa: E402
    FORMAL_V2_ACTIVATION_PATH,
    FORMAL_V2_CONFIG_FILE_SHA256,
    FORMAL_V2_CONFIG_PATH,
    FORMAL_V2_EXPERIMENT_ID,
    FORMAL_V2_MANIFEST,
    FORMAL_V2_SOURCE_MANIFEST,
    FORMAL_V2_SOURCE_MANIFEST_SHA256,
    load_formal_v2_activation,
    load_s05_formal_v2_config,
)
from steiner_branching.learning.imitation import atomic_write_json  # noqa: E402
from steiner_branching.learning.teacher_data import file_sha256  # noqa: E402


RECORD_FIELDS = {
    "path", "file_sha256", "semantic_sha256", "task_id", "role", "split",
    "family", "bucket_id", "graph_sha256", "teacher_seed", "state_index",
    "state_valid", "all_tie", "candidate_count",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_key(
    record: Mapping[str, Any], fields: list[str], bucket_rank: Mapping[str, int]
) -> tuple[Any, ...]:
    values: list[Any] = []
    for field in fields:
        if field == "bucket_order":
            bucket = str(record["bucket_id"])
            if bucket not in bucket_rank:
                raise KeyError(f"unknown bucket_id for bucket_order: {bucket}")
            values.append(bucket_rank[bucket])
        else:
            if field not in record:
                raise KeyError(field)
            values.append(record[field])
    return tuple(values)


def select_v2_train_records(
    records: list[dict[str, Any]], revision: Mapping[str, Any]
) -> list[dict[str, Any]]:
    policy = revision["state_selection"]
    bucket_rank = {name: index for index, name in enumerate(policy["bucket_order"])}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for record in records:
        if record["role"] != "train" or not record["state_valid"]:
            continue
        semantic = str(record["semantic_sha256"])
        if semantic in seen:
            raise ValueError(f"duplicate semantic_sha256: {semantic}")
        seen.add(semantic)
        grouped.setdefault((str(record["family"]), str(record["bucket_id"])), []).append(record)

    selected: list[dict[str, Any]] = []
    for family, targets in policy["initial_bucket_targets"].items():
        family_selected: list[dict[str, Any]] = []
        remaining: list[dict[str, Any]] = []
        for bucket, target in targets.items():
            eligible = sorted(
                grouped.get((family, bucket), []),
                key=lambda item: _canonical_key(item, policy["primary_order"], bucket_rank),
            )
            family_selected.extend(eligible[: int(target)])
            remaining.extend(eligible[int(target):])
        shortage = int(policy["train_quota_per_family"]) - len(family_selected)
        if shortage < 0:
            raise ValueError(f"initial bucket targets exceed family quota: {family}")
        remaining.sort(
            key=lambda item: _canonical_key(item, policy["fallback_order"], bucket_rank)
        )
        family_selected.extend(remaining[:shortage])
        if len(family_selected) != int(policy["train_quota_per_family"]):
            raise ValueError(f"family has fewer than 128 eligible states: {family}")
        selected.extend(family_selected)
    if len(selected) != 640 or len({item["semantic_sha256"] for item in selected}) != 640:
        raise ValueError("formal-v2 train selection is not 640 unique states")
    return selected


def _validate_source_manifest(path: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if file_sha256(path) != FORMAL_V2_SOURCE_MANIFEST_SHA256:
        raise ValueError("sealed formal-v1 manifest checksum changed")
    required = {
        "schema_version": 2,
        "stage": "S05",
        "experiment_id": "s05-teacher-il-formal-v1",
        "status": "failed",
        "git_commit": "13a8a83761f660c96cb238471de2e2c0a2d54a8b",
        "formal_gate_evaluated": False,
        "failures": [],
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise ValueError(f"sealed formal-v1 manifest {key} changed")
    checks = manifest.get("teacher_gate", {}).get("checks", {})
    if manifest.get("teacher_gate", {}).get("status") != "FAIL":
        raise ValueError("formal-v1 FAIL evidence was rewritten")
    if {key for key, value in checks.items() if value is not True} != {"all_state_quotas_met"}:
        raise ValueError("formal-v1 failed-check identity changed")
    counts = manifest.get("counts", {})
    expected_counts = {
        "base_graphs": 105, "tasks": 315, "expected_states": 5040,
        "observed_states": 3431, "valid_states": 3207,
        "all_tie_valid_states": 14, "candidates_observed": 88549,
        "candidates_mapped": 88549,
    }
    for key, value in expected_counts.items():
        if counts.get(key) != value:
            raise ValueError(f"sealed formal-v1 count changed: {key}")
    if len(manifest.get("task_ids", [])) != 315 or len(manifest.get("task_statuses", {})) != 315:
        raise ValueError("sealed formal-v1 task denominator changed")
    if set(manifest["task_statuses"].values()) - {"completed", "root_solved"}:
        raise ValueError("sealed formal-v1 task is non-terminal")
    records = manifest.get("shards")
    if not isinstance(records, list) or len(records) != 3431:
        raise ValueError("sealed formal-v1 shard count changed")
    graph_roles: dict[str, str] = {}
    semantic_roles: dict[str, str] = {}
    seen_semantics: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != RECORD_FIELDS:
            raise ValueError(f"sealed shard schema changed at index {index}")
        role = str(record["role"])
        split = str(record["split"])
        if role not in {"train", "validation_select", "validation_gate"}:
            raise ValueError(f"illegal role at shard {index}")
        if split != ("train" if role == "train" else "validation_iid"):
            raise ValueError(f"split/role mismatch at shard {index}")
        graph = str(record["graph_sha256"])
        if graph_roles.setdefault(graph, role) != role:
            raise ValueError("base-graph lineage crosses roles")
        semantic = str(record["semantic_sha256"])
        if semantic_roles.setdefault(semantic, role) != role:
            raise ValueError("semantic identity crosses roles")
        if semantic in seen_semantics:
            raise ValueError(f"duplicate semantic_sha256: {semantic}")
        seen_semantics.add(semantic)
        shard_path = path.parent / str(record["path"])
        if file_sha256(shard_path) != record["file_sha256"]:
            raise ValueError(f"sealed shard checksum changed: {record['path']}")
    return records


def build_v2_manifest(source_path: Path = FORMAL_V2_SOURCE_MANIFEST) -> dict[str, Any]:
    load_formal_v2_activation()
    effective = load_s05_formal_v2_config(require_activation=True)
    revision = load_yaml_mapping(FORMAL_V2_CONFIG_PATH)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    records = _validate_source_manifest(source_path, source)
    train = select_v2_train_records(records, revision)
    validation_select = source["selection"]["validation_select"]
    validation_gate = source["selection"]["validation_gate"]
    if len(validation_select) != 160 or len(validation_gate) != 320:
        raise ValueError("sealed validation selection changed")
    shard_by_semantic = {item["semantic_sha256"]: item for item in records}
    for role, selected in (
        ("validation_select", validation_select), ("validation_gate", validation_gate)
    ):
        for item in selected:
            if item != shard_by_semantic.get(item.get("semantic_sha256")):
                raise ValueError(f"sealed {role} selection is not an exact shard record")
            if item["role"] != role or not item["state_valid"]:
                raise ValueError(f"sealed {role} selection eligibility changed")
    expected_buckets = revision["state_selection"]["expected_train_bucket_counts_from_sealed_manifest"]
    actual_buckets = {
        family: {
            bucket: sum(
                item["family"] == family and item["bucket_id"] == bucket for item in train
            )
            for bucket in buckets
        }
        for family, buckets in expected_buckets.items()
    }
    if actual_buckets != expected_buckets:
        raise ValueError(f"formal-v2 bucket composition changed: {actual_buckets}")
    family_counts = dict(Counter(str(item["family"]) for item in train))
    if set(family_counts.values()) != {128} or len(family_counts) != 5:
        raise ValueError("formal-v2 family composition changed")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=True
    ).stdout.strip()
    return {
        "schema_version": 1,
        "stage": "S05",
        "experiment_id": FORMAL_V2_EXPERIMENT_ID,
        "status": "completed",
        "created_at_utc": utc_now(),
        "implementation_run_head": head,
        "revision_yaml_path": str(FORMAL_V2_CONFIG_PATH.relative_to(REPO)),
        "revision_yaml_sha256": FORMAL_V2_CONFIG_FILE_SHA256,
        "effective_config_sha256": formal_config_digest(effective),
        "activation_record": str(FORMAL_V2_ACTIVATION_PATH.relative_to(REPO)),
        "activation_record_sha256": file_sha256(FORMAL_V2_ACTIVATION_PATH),
        "source_manifest": str(source_path.relative_to(REPO)),
        "source_manifest_sha256": FORMAL_V2_SOURCE_MANIFEST_SHA256,
        "source_root": str(source_path.parent.relative_to(REPO)),
        "source_formal_v1_status": "FAIL_RETAINED",
        "teacher_tasks_rerun": False,
        "selection_policy": revision["state_selection"],
        "selection": {
            "train": train,
            "validation_select": validation_select,
            "validation_gate": validation_gate,
        },
        "counts": {
            "train": 640, "validation_select": 160, "validation_gate": 320,
            "family": family_counts, "train_bucket": actual_buckets,
        },
        "teacher_gate": {
            "status": "PASS",
            "scope": "sealed_v1_quality_checks_plus_audited_v2_selection_feasibility",
            "checks": {
                "source_v1_failure_retained": True,
                "all_source_tasks_terminal": True,
                "zero_source_task_failures": True,
                "teacher_valid_fraction": True,
                "teacher_all_tie_fraction": True,
                "action_mapping_rate": True,
                "v2_all_state_quotas_met": True,
                "zero_split_or_role_leakage": True,
                "all_referenced_shard_checksums_verified": True,
                "validation_selection_unchanged": True,
            },
        },
        "formal_gate_evaluated": False,
        "s06_authorized": False,
        "test_and_final_accessed": False,
    }


def formal_config_digest(config: Mapping[str, Any]) -> str:
    from steiner_branching.learning.formal_protocol import formal_config_sha256
    return formal_config_sha256(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", default=str(FORMAL_V2_SOURCE_MANIFEST))
    parser.add_argument("--output", default=str(FORMAL_V2_MANIFEST))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source_manifest)
    output = Path(args.output)
    source = source if source.is_absolute() else REPO / source
    output = output if output.is_absolute() else REPO / output
    if output.exists():
        raise SystemExit(f"refusing to overwrite formal-v2 manifest: {output}")
    manifest = build_v2_manifest(source)
    atomic_write_json(output, manifest)
    print(json.dumps({
        "status": "PASS",
        "manifest": str(output),
        "manifest_sha256": file_sha256(output),
        "counts": manifest["counts"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
