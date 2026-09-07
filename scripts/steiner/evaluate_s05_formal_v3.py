#!/usr/bin/env python3
"""Evaluate one frozen formal-v2 checkpoint on the sealed formal-v3 Gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from steiner_branching.data.generate import SYNTHETIC_FAMILIES  # noqa: E402
from steiner_branching.learning.formal_v3 import (  # noqa: E402
    V3_EXPERIMENT_ID,
    V3_MODEL_SEEDS,
    V3_RAW_ROOT,
    V3_SELECTED_MANIFEST,
    load_v3_protocol,
    load_v3_selection_seal,
)
from steiner_branching.learning.imitation import (  # noqa: E402
    atomic_write_json,
    enable_cuda_determinism,
    load_checkpoint_bundle,
    model_predictions,
    normalized_sb_regret,
    offline_baseline_predictions,
    ranking_metrics,
    seed_everything,
)
from steiner_branching.learning.teacher_data import (  # noqa: E402
    file_sha256,
    load_teacher_sample,
)


MODEL_CONFIG = REPO / "configs/steiner/models/b0_milp_gcnn_v1.yml"
REPORT_ROOT = REPO / "results/steiner/s05/s05-teacher-il-formal-v3-confirmatory"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", required=True, type=int)
    return parser.parse_args()


def _random_predictions(samples: list[Any], *, seed: int) -> list[np.ndarray]:
    predictions = []
    for sample in samples:
        values = []
        for probindex in sample.state.candidate_indices:
            digest = hashlib.sha256(
                f"{seed}:{sample.semantic_sha256}:{int(probindex)}".encode("ascii")
            ).digest()
            values.append(int.from_bytes(digest[:8], "big") / 2**64)
        predictions.append(np.asarray(values, dtype=np.float64))
    return predictions


def _load_selected_samples(
    selected: dict[str, Any],
) -> tuple[list[Any], list[dict[str, Any]]]:
    records = selected.get("selection", {}).get("validation_gate")
    if not isinstance(records, list) or len(records) != 320:
        raise ValueError("formal-v3 selected Gate must contain exactly 320 records")
    source_root = REPO / str(selected.get("source_root"))
    samples = []
    seen_semantics: set[str] = set()
    family_counts: dict[str, int] = {family: 0 for family in SYNTHETIC_FAMILIES}
    graph_families: dict[str, str] = {}
    for record in records:
        semantic = str(record["semantic_sha256"])
        if semantic in seen_semantics:
            raise ValueError("formal-v3 selected Gate contains duplicate semantics")
        sample = load_teacher_sample(
            source_root / str(record["path"]),
            expected_file_sha256=str(record["file_sha256"]),
        )
        if any((
            not sample.labels.state_valid,
            sample.semantic_sha256 != semantic,
            sample.graph_sha256 != record["graph_sha256"],
            sample.family != record["family"],
            sample.teacher_seed != int(record["teacher_seed"]),
            sample.state_index != int(record["state_index"]),
        )):
            raise ValueError("formal-v3 selected shard identity/validity changed")
        seen_semantics.add(semantic)
        family_counts[sample.family] += 1
        previous = graph_families.setdefault(sample.graph_sha256, sample.family)
        if previous != sample.family:
            raise ValueError("formal-v3 lineage family identity changed")
        samples.append(sample)
    if family_counts != {family: 64 for family in SYNTHETIC_FAMILIES}:
        raise ValueError("formal-v3 selected family state counts changed")
    lineage_counts = {
        family: sum(value == family for value in graph_families.values())
        for family in SYNTHETIC_FAMILIES
    }
    if len(graph_families) != 30 or lineage_counts != {
        family: 6 for family in SYNTHETIC_FAMILIES
    }:
        raise ValueError("formal-v3 selected lineage matrix changed")
    return samples, records


def _state_records(
    samples: list[Any],
    model_scores: list[np.ndarray],
    random_scores: list[np.ndarray],
    *,
    tolerance: float,
) -> list[dict[str, Any]]:
    values = []
    for sample, model, random in zip(samples, model_scores, random_scores):
        values.append({
            "task_id": sample.task_id,
            "semantic_sha256": sample.semantic_sha256,
            "graph_sha256": sample.graph_sha256,
            "family": sample.family,
            "teacher_seed": sample.teacher_seed,
            "state_index": sample.state_index,
            "model_regret": normalized_sb_regret(
                sample.labels.scores, int(np.argmax(model)), tie_tolerance=tolerance
            ),
            "random_regret": normalized_sb_regret(
                sample.labels.scores, int(np.argmax(random)), tie_tolerance=tolerance
            ),
        })
    return values


def _max_error(left: list[np.ndarray], right: list[np.ndarray]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("formal-v3 prediction sets are empty or misaligned")
    return max(
        float(np.max(np.abs(first - second)))
        for first, second in zip(left, right)
    )


def main() -> int:
    args = parse_args()
    seed = int(args.model_seed)
    if seed not in V3_MODEL_SEEDS:
        raise SystemExit(f"model seed must be one of {V3_MODEL_SEEDS}")

    # This is the mandatory pre-model barrier. Nothing above this line reads a
    # checkpoint manifest via torch or loads a model payload.
    seal = load_v3_selection_seal()
    config = load_v3_protocol(require_activation=True)
    selected = json.loads(V3_SELECTED_MANIFEST.read_text(encoding="utf-8"))
    if any((
        selected.get("status") != "completed",
        selected.get("pre_model_access_gate", {}).get("status") != "PASS",
        selected.get("checkpoint_loaded") is not False,
        selected.get("formal_gate_evaluated") is not False,
        selected.get("test_and_final_accessed") is not False,
        file_sha256(V3_SELECTED_MANIFEST) != seal["selected_manifest_sha256"],
    )):
        raise ValueError("formal-v3 selected manifest failed the pre-model barrier")
    samples, records = _load_selected_samples(selected)
    del records

    expected_workspace = ":4096:8"
    cuda_contract = enable_cuda_determinism(
        expected_workspace_config=expected_workspace
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("formal-v3 model evaluation must see exactly one CUDA device")
    device = torch.device("cuda:0")
    gpu_model = torch.cuda.get_device_name(device)
    seed_everything(seed)

    checkpoint = selected["frozen_checkpoints"][str(seed)]
    manifest_path = REPO / checkpoint["manifest_path"]
    if file_sha256(manifest_path) != checkpoint["manifest_sha256"]:
        raise ValueError("formal-v3 frozen checkpoint manifest changed after sealing")
    if file_sha256(REPO / checkpoint["model_path"]) != checkpoint["model_sha256"]:
        raise ValueError("formal-v3 frozen checkpoint payload changed after sealing")

    output = REPORT_ROOT / "evaluation_shards" / f"seed-{seed:03d}.json"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite formal-v3 seed evidence: {output}")
    report: dict[str, Any] = {
        "schema_version": 1,
        "stage": "S05",
        "experiment_id": V3_EXPERIMENT_ID,
        "report_kind": "frozen_checkpoint_evaluation_shard",
        "status": "running",
        "started_at_utc": utc_now(),
        "evaluation_git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
            capture_output=True, check=True,
        ).stdout.strip(),
        "implementation_run_head": seal["implementation_run_head"],
        "protocol_yaml_sha256": seal["protocol_yaml_sha256"],
        "selected_manifest_sha256": seal["selected_manifest_sha256"],
        "training_seed": seed,
        "model_seed": seed,
        "checkpoint_manifest": checkpoint["manifest_path"],
        "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
        "checkpoint_model_sha256": checkpoint["model_sha256"],
        "gpu_model": gpu_model,
        "cuda_determinism": cuda_contract,
        "formal_gate_evaluated": False,
        "test_and_final_accessed": False,
    }
    atomic_write_json(output, report)
    try:
        first_model, first_stats, first_manifest = load_checkpoint_bundle(
            manifest_path, model_config_path=MODEL_CONFIG, device=device
        )
        first = model_predictions(first_model, samples, first_stats, device=device)
        repeated = model_predictions(first_model, samples, first_stats, device=device)
        repeat_error = _max_error(first, repeated)

        second_model, second_stats, second_manifest = load_checkpoint_bundle(
            manifest_path, model_config_path=MODEL_CONFIG, device=device
        )
        state_dict_exact = first_stats == second_stats and all(
            torch.equal(value, second_model.state_dict()[name])
            for name, value in first_model.state_dict().items()
        )
        reloaded = model_predictions(second_model, samples, second_stats, device=device)
        reload_error = _max_error(first, reloaded)
        if first_manifest != second_manifest or not state_dict_exact:
            raise RuntimeError("formal-v3 checkpoint reload is not bit exact")
        if repeat_error != 0.0 or reload_error != 0.0:
            raise RuntimeError(
                "formal-v3 deterministic inference failed: "
                f"repeat={repeat_error}, reload={reload_error}"
            )
        random_scores = _random_predictions(
            samples, seed=int(config["confirmatory_evaluation"]["random_baseline_seed"])
        )
        tolerance = float(config["solver_and_teacher"]["tie_relative_tolerance"])
        report.update({
            "status": "completed",
            "finished_at_utc": utc_now(),
            "validation_gate_state_count": len(samples),
            "validation_gate_metrics": ranking_metrics(
                samples, first, tie_tolerance=tolerance
            ),
            "offline_gate_baselines": {
                "random": ranking_metrics(samples, random_scores, tie_tolerance=tolerance),
                "most_infeasible": ranking_metrics(
                    samples,
                    offline_baseline_predictions(
                        samples, baseline="most_infeasible", seed=0
                    ),
                    tie_tolerance=tolerance,
                ),
                "pseudocost": ranking_metrics(
                    samples,
                    offline_baseline_predictions(samples, baseline="pseudocost", seed=0),
                    tie_tolerance=tolerance,
                ),
            },
            "gate_state_records": _state_records(
                samples, first, random_scores, tolerance=tolerance
            ),
            "repeat_inference_max_absolute_error": repeat_error,
            "reload_max_absolute_error": reload_error,
            "reload_state_dict_bit_exact": state_dict_exact,
        })
    except BaseException as error:
        report.update({
            "status": "failed",
            "finished_at_utc": utc_now(),
            "error": f"{type(error).__name__}: {error}",
        })
    atomic_write_json(output, report)
    print(json.dumps({"status": report["status"], "seed": seed, "report": str(output)}))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
