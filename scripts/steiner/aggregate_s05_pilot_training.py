#!/usr/bin/env python3
"""Fail-closed aggregation of disjoint S05 pilot training seed reports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from steiner_branching.learning.imitation import atomic_write_json  # noqa: E402
from steiner_branching.learning.teacher_data import (  # noqa: E402
    file_sha256,
    load_s05_config,
    s05_config_sha256,
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
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="seed-shard report path; repeat once per parallel training job",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def _current_identity(required_tag: str) -> tuple[str, str]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=True
    ).stdout.strip()
    tag = subprocess.run(
        ["git", "rev-parse", f"refs/tags/{required_tag}^{{}}"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", tag, head], cwd=REPO, check=False
    ).returncode != 0:
        raise ValueError("required S04 audited tag is outside current history")
    return head, tag


def _load_report(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"training report is not an object: {path}")
    return raw


def aggregate_reports(
    *,
    config: dict[str, Any],
    input_paths: list[Path],
    output_path: Path,
    git_commit: str,
    audited_target: str,
) -> dict[str, Any]:
    if not input_paths or len(set(input_paths)) != len(input_paths):
        raise ValueError("aggregation inputs must be non-empty and unique")
    expected_seeds = tuple(int(seed) for seed in config["training"]["pilot_seeds"])
    expected_counts = tuple(
        int(count) for count in config["training"]["learning_curve_train_states"]
    )
    config_digest = s05_config_sha256(config)
    manifest_path = resolve_path(config["artifacts"]["raw_root"]) / "manifest.json"
    manifest_digest = file_sha256(manifest_path)
    reports = [_load_report(path) for path in input_paths]
    common_keys = (
        "schema_version",
        "stage",
        "experiment_id",
        "git_commit",
        "s04_audited_tag_target",
        "config_sha256",
        "data_manifest",
        "data_manifest_sha256",
        "train_valid_states",
        "validation_valid_states",
        "expected_training_seeds",
        "offline_baselines",
        "cuda_determinism",
        "formal_gate_evaluated",
    )
    reference = reports[0]
    for path, report in zip(input_paths, reports):
        if report.get("report_kind") != "pilot_seed_shard":
            raise ValueError(f"input is not a pilot seed shard: {path}")
        for key in common_keys:
            if report.get(key) != reference.get(key):
                raise ValueError(f"training shard mismatch for {key}: {path}")
        if report.get("schema_version") != 1 or report.get("stage") != "S05":
            raise ValueError(f"training shard schema/stage mismatch: {path}")
        if report.get("experiment_id") != config["experiment_id"]:
            raise ValueError(f"training shard experiment mismatch: {path}")
        if report.get("git_commit") != git_commit:
            raise ValueError(f"training shard Git commit mismatch: {path}")
        if report.get("s04_audited_tag_target") != audited_target:
            raise ValueError(f"training shard S04 audit target mismatch: {path}")
        if report.get("config_sha256") != config_digest:
            raise ValueError(f"training shard config mismatch: {path}")
        if report.get("data_manifest_sha256") != manifest_digest:
            raise ValueError(f"training shard teacher manifest mismatch: {path}")
        if report.get("expected_training_seeds") != list(expected_seeds):
            raise ValueError(f"training shard expected-seed set changed: {path}")
        if report.get("formal_gate_evaluated") is not False:
            raise ValueError(f"pilot shard incorrectly claims a formal Gate: {path}")

    seen_seeds: set[int] = set()
    seen_runs: set[tuple[int, int]] = set()
    runs: list[dict[str, Any]] = []
    for path, report in zip(input_paths, reports):
        selected = tuple(int(seed) for seed in report.get("selected_training_seeds", []))
        if not selected or len(set(selected)) != len(selected):
            raise ValueError(f"training shard seed selection is empty/duplicated: {path}")
        if not set(selected).issubset(expected_seeds):
            raise ValueError(f"training shard contains an unregistered seed: {path}")
        overlap = seen_seeds.intersection(selected)
        if overlap:
            raise ValueError(f"training seed appears in multiple shards: {sorted(overlap)}")
        seen_seeds.update(selected)
        local_runs: set[tuple[int, int]] = set()
        for run in report.get("runs", []):
            key = (int(run["training_seed"]), int(run["train_state_count"]))
            if key[0] not in selected or key[1] not in expected_counts:
                raise ValueError(f"training shard contains an unexpected run {key}: {path}")
            if key in local_runs or key in seen_runs:
                raise ValueError(f"duplicate training run {key}: {path}")
            local_runs.add(key)
            seen_runs.add(key)
            runs.append(dict(run))
        expected_local = {(seed, count) for seed in selected for count in expected_counts}
        if local_runs != expected_local:
            missing = sorted(expected_local - local_runs)
            raise ValueError(f"training shard is missing registered runs {missing}: {path}")

    if seen_seeds != set(expected_seeds):
        raise ValueError(
            f"pilot seed shards are incomplete: missing={sorted(set(expected_seeds) - seen_seeds)}"
        )
    expected_runs = {(seed, count) for seed in expected_seeds for count in expected_counts}
    if seen_runs != expected_runs:
        raise ValueError("pilot training run matrix is incomplete")

    sources = [
        {"path": str(path), "sha256": file_sha256(path)}
        for path in input_paths
    ]
    failures = sum(run.get("status") != "completed" for run in runs)
    aggregate = {
        key: reference[key]
        for key in common_keys
    }
    aggregate.update(
        {
            "report_kind": "pilot_aggregate",
            "selected_training_seeds": list(expected_seeds),
            "started_at_utc": min(str(report["started_at_utc"]) for report in reports),
            "finished_at_utc": utc_now(),
            "source_reports": sources,
            "runs": sorted(
                runs,
                key=lambda run: (int(run["train_state_count"]), int(run["training_seed"])),
            ),
            "status": "completed" if failures == 0 else "failed",
        }
    )
    atomic_write_json(output_path, aggregate)
    return aggregate


def main() -> int:
    args = parse_args()
    config = load_s05_config(resolve_path(args.config))
    git_commit, audited_target = _current_identity(str(config["required_s04_audited_tag"]))
    output_path = (
        resolve_path(args.output)
        if args.output
        else resolve_path(config["artifacts"]["report_root"]) / "pilot_training_report.json"
    )
    report = aggregate_reports(
        config=config,
        input_paths=[resolve_path(value) for value in args.input],
        output_path=output_path,
        git_commit=git_commit,
        audited_target=audited_target,
    )
    print(
        json.dumps(
            {"status": report["status"], "runs": len(report["runs"]), "report": str(output_path)},
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
