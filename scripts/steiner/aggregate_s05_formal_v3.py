#!/usr/bin/env python3
"""Aggregate five frozen-checkpoint evaluations for S05 formal-v3."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO / "python"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (PYTHON_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aggregate_s05_formal_training import evaluate_formal_gate  # noqa: E402
from steiner_branching.data.generate import SYNTHETIC_FAMILIES  # noqa: E402
from steiner_branching.learning.formal_v3 import (  # noqa: E402
    V3_EXPERIMENT_ID,
    V3_MODEL_SEEDS,
    V3_SELECTED_MANIFEST,
    load_v3_protocol,
    load_v3_selection_seal,
)
from steiner_branching.learning.imitation import atomic_write_json  # noqa: E402
from steiner_branching.learning.teacher_data import file_sha256  # noqa: E402


REPORT_ROOT = REPO / "results/steiner/s05/s05-teacher-il-formal-v3-confirmatory"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable formal-v3 evidence {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"formal-v3 evidence must be a mapping: {path}")
    return value


def main() -> int:
    seal = load_v3_selection_seal()
    config = load_v3_protocol(require_activation=True)
    selected = _read_json(V3_SELECTED_MANIFEST)
    selected_digest = file_sha256(V3_SELECTED_MANIFEST)
    if any((
        selected_digest != seal["selected_manifest_sha256"],
        selected.get("pre_model_access_gate", {}).get("status") != "PASS",
        selected.get("formal_gate_evaluated") is not False,
        selected.get("test_and_final_accessed") is not False,
    )):
        raise ValueError("formal-v3 selected manifest is not aggregate-eligible")

    paths = [
        REPORT_ROOT / "evaluation_shards" / f"seed-{seed:03d}.json"
        for seed in V3_MODEL_SEEDS
    ]
    reports = [_read_json(path) for path in paths]
    for seed, path, report in zip(V3_MODEL_SEEDS, paths, reports):
        frozen = selected["frozen_checkpoints"][str(seed)]
        if any((
            report.get("schema_version") != 1,
            report.get("stage") != "S05",
            report.get("experiment_id") != V3_EXPERIMENT_ID,
            report.get("report_kind") != "frozen_checkpoint_evaluation_shard",
            report.get("status") != "completed",
            int(report.get("training_seed", -1)) != seed,
            int(report.get("model_seed", -1)) != seed,
            report.get("implementation_run_head") != seal["implementation_run_head"],
            report.get("protocol_yaml_sha256") != seal["protocol_yaml_sha256"],
            report.get("selected_manifest_sha256") != selected_digest,
            report.get("checkpoint_manifest_sha256") != frozen["manifest_sha256"],
            report.get("checkpoint_model_sha256") != frozen["model_sha256"],
            report.get("repeat_inference_max_absolute_error") != 0.0,
            report.get("reload_max_absolute_error") != 0.0,
            report.get("reload_state_dict_bit_exact") is not True,
            report.get("formal_gate_evaluated") is not False,
            report.get("test_and_final_accessed") is not False,
        )):
            raise ValueError(f"formal-v3 seed report is incomplete or changed: {path}")
        if file_sha256(REPO / frozen["manifest_path"]) != frozen["manifest_sha256"]:
            raise ValueError(f"formal-v3 frozen manifest changed for seed {seed}")
        if file_sha256(REPO / frozen["model_path"]) != frozen["model_sha256"]:
            raise ValueError(f"formal-v3 frozen model changed for seed {seed}")

    gate_config = {
        "state_selection": {
            "train_quotas": {family: {} for family in SYNTHETIC_FAMILIES}
        },
        "statistics": dict(config["statistics"]),
        "gate": {
            "max_seed_regret_coefficient_of_variation": config["scientific_gate"][
                "max_seed_regret_coefficient_of_variation"
            ]
        },
    }
    gate = evaluate_formal_gate(gate_config, selected, reports)
    output = REPORT_ROOT / "formal_v3_evaluation_report.json"
    summary = REPORT_ROOT / "S05_FORMAL_V3_GATE_SUMMARY.json"
    if output.exists() or summary.exists():
        raise RuntimeError("refusing to overwrite formal-v3 aggregate evidence")
    aggregate = {
        "schema_version": 1,
        "stage": "S05",
        "experiment_id": V3_EXPERIMENT_ID,
        "created_at_utc": utc_now(),
        "evaluation_git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
            capture_output=True, check=True,
        ).stdout.strip(),
        "implementation_run_head": seal["implementation_run_head"],
        "protocol_yaml_sha256": seal["protocol_yaml_sha256"],
        "selected_manifest_sha256": selected_digest,
        "seed_report_sha256": {
            str(seed): file_sha256(path)
            for seed, path in zip(V3_MODEL_SEEDS, paths)
        },
        "gate": gate,
        "formal_gate_evaluated": True,
        "s05_local_gate_status": gate["status"],
        "s05_audited_tag_authorized": False,
        "s06_authorized": False,
        "test_and_final_accessed": False,
    }
    atomic_write_json(output, aggregate)
    atomic_write_json(summary, aggregate)
    print(json.dumps({"status": gate["status"], "report": str(output)}, sort_keys=True))
    return 0 if gate["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
