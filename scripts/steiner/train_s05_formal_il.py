#!/usr/bin/env python3
"""Train one audited S05 formal seed on one V100."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
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

from steiner_branching.learning.formal_protocol import (  # noqa: E402
    FORMAL_CONFIG_PATH,
    FORMAL_TRAINING_SEEDS,
    formal_config_sha256,
    load_s05_formal_config,
)
from steiner_branching.learning.imitation import (  # noqa: E402
    atomic_write_json,
    enable_cuda_determinism,
    fit_train_normalization,
    load_checkpoint_bundle,
    model_predictions,
    normalized_sb_regret,
    offline_baseline_predictions,
    ranking_metrics,
    save_checkpoint_bundle,
    seed_everything,
    train_one_epoch,
)
from steiner_branching.learning.teacher_data import file_sha256, load_teacher_sample  # noqa: E402
from steiner_branching.models.milp_gcnn import MilpBipartiteGCNN, load_b0_config  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(FORMAL_CONFIG_PATH))
    parser.add_argument("--data-manifest")
    parser.add_argument("--training-seed", required=True, type=int)
    return parser.parse_args()


def require_git_identity(tag: str) -> tuple[str, str]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=True
    ).stdout.strip()
    target_process = subprocess.run(
        ["git", "rev-parse", f"refs/tags/{tag}^{{}}"], cwd=REPO,
        text=True, capture_output=True, check=False,
    )
    if target_process.returncode != 0:
        raise RuntimeError(f"required audited tag {tag!r} is missing")
    target = target_process.stdout.strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", target, head], cwd=REPO, check=False
    ).returncode != 0:
        raise RuntimeError("S04 audited tag is outside current history")
    return head, target


def formal_random_predictions(samples: list[Any], *, seed: int) -> list[np.ndarray]:
    predictions: list[np.ndarray] = []
    for sample in samples:
        values = []
        for probindex in sample.state.candidate_indices:
            digest = hashlib.sha256(
                f"{seed}:{sample.semantic_sha256}:{int(probindex)}".encode("ascii")
            ).digest()
            values.append(int.from_bytes(digest[:8], "big") / 2**64)
        predictions.append(np.asarray(values, dtype=np.float64))
    return predictions


def _load_selected(
    manifest_path: Path, manifest: dict[str, Any], role: str
) -> tuple[list[Any], list[dict[str, Any]]]:
    raw_records = manifest.get("selection", {}).get(role)
    if not isinstance(raw_records, list):
        raise ValueError(f"formal manifest has no {role} selection")
    samples = []
    records = []
    for item in raw_records:
        sample = load_teacher_sample(
            manifest_path.parent / item["path"], expected_file_sha256=item["file_sha256"]
        )
        if (
            sample.semantic_sha256 != item["semantic_sha256"]
            or sample.graph_sha256 != item["graph_sha256"]
            or sample.family != item["family"]
            or not sample.labels.state_valid
        ):
            raise ValueError("formal selected shard identity/validity mismatch")
        samples.append(sample)
        records.append(item)
    return samples, records


def _state_gate_records(
    samples: list[Any], records: list[dict[str, Any]],
    model_predictions_raw: list[np.ndarray], random_predictions: list[np.ndarray],
    *, tolerance: float,
) -> list[dict[str, Any]]:
    result = []
    for sample, record, model_scores, random_scores in zip(
        samples, records, model_predictions_raw, random_predictions
    ):
        teacher = sample.labels.scores
        result.append({
            "task_id": sample.task_id,
            "semantic_sha256": sample.semantic_sha256,
            "graph_sha256": sample.graph_sha256,
            "family": sample.family,
            "teacher_seed": sample.teacher_seed,
            "state_index": sample.state_index,
            "model_regret": normalized_sb_regret(
                teacher, int(np.argmax(model_scores)), tie_tolerance=tolerance
            ),
            "random_regret": normalized_sb_regret(
                teacher, int(np.argmax(random_scores)), tie_tolerance=tolerance
            ),
        })
    return result


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_s05_formal_config(config_path, require_activation=True)
    seed = int(args.training_seed)
    if seed not in FORMAL_TRAINING_SEEDS:
        raise SystemExit(f"formal training seed must be one of {FORMAL_TRAINING_SEEDS}")
    training = config["training"]
    cuda_contract = enable_cuda_determinism(
        expected_workspace_config=str(training["cublas_workspace_config"])
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("formal seed job must see exactly one CUDA device")
    device = torch.device("cuda:0")
    observed_gpu = torch.cuda.get_device_name(device)
    if observed_gpu != training["gpu_model"]:
        raise RuntimeError(
            f"formal GPU model mismatch: expected {training['gpu_model']!r}, got {observed_gpu!r}"
        )
    git_commit, audited_target = require_git_identity(config["required_s04_audited_tag"])
    manifest_path = (
        resolve_path(args.data_manifest)
        if args.data_manifest
        else resolve_path(config["artifacts"]["raw_root"]) / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_digest = formal_config_sha256(config)
    if (
        manifest.get("schema_version") != 2
        or manifest.get("experiment_id") != config["experiment_id"]
        or manifest.get("status") != "completed"
        or manifest.get("teacher_gate", {}).get("status") != "PASS"
        or manifest.get("formal_gate_evaluated") is not False
        or manifest.get("config_sha256") != config_digest
        or manifest.get("git_commit") != git_commit
        or manifest.get("s04_audited_tag_target") != audited_target
    ):
        raise ValueError("formal teacher manifest identity/Gate is not eligible")
    train_samples, train_records = _load_selected(manifest_path, manifest, "train")
    select_samples, select_records = _load_selected(manifest_path, manifest, "validation_select")
    gate_samples, gate_records = _load_selected(manifest_path, manifest, "validation_gate")
    expected_counts = config["state_selection"]["selected_state_counts"]
    if {
        "train": len(train_samples),
        "validation_select": len(select_samples),
        "validation_gate": len(gate_samples),
    } != expected_counts:
        raise ValueError("formal selected dataset counts changed")
    del train_records, select_records
    role_graphs = [set(sample.graph_sha256 for sample in values) for values in (
        train_samples, select_samples, gate_samples
    )]
    if any(role_graphs[i] & role_graphs[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("formal selected base graph lineage crosses roles")

    b0_path = resolve_path(config["model_config"])
    architecture = load_b0_config(b0_path)["architecture"]
    stats = fit_train_normalization(train_samples)
    tolerance = float(config["teacher"]["tie_relative_tolerance"])
    seed_everything(seed)
    model = MilpBipartiteGCNN(
        embedding_dim=int(architecture["embedding_dim"]),
        hidden_dim=int(architecture["hidden_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    report_path = resolve_path(config["artifacts"]["report_root"]) / "formal_training_shards" / f"seed-{seed:03d}.json"
    checkpoint_dir = resolve_path(training["checkpoint_root"]) / f"seed-{seed:03d}"
    if report_path.exists() or checkpoint_dir.exists():
        raise RuntimeError(
            f"formal seed {seed} already has an artifact; refusing to overwrite failed or completed evidence"
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "stage": "S05",
        "experiment_id": config["experiment_id"],
        "report_kind": "formal_seed_shard",
        "status": "running",
        "started_at_utc": utc_now(),
        "git_commit": git_commit,
        "config_sha256": config_digest,
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": file_sha256(manifest_path),
        "s04_audited_tag_target": audited_target,
        "training_seed": seed,
        "train_state_count": len(train_samples),
        "validation_select_state_count": len(select_samples),
        "validation_gate_state_count": len(gate_samples),
        "gpu_model": observed_gpu,
        "cuda_determinism": cuda_contract,
        "history": [],
        "formal_gate_evaluated": False,
    }
    atomic_write_json(report_path, report)
    best_regret = math.inf
    best_epoch = -1
    best_metrics = None
    best_state = None
    try:
        for epoch in range(int(training["epochs"])):
            loss = train_one_epoch(
                model, train_samples, stats, optimizer, device=device,
                temperature=float(training["target_temperature"]),
                gradient_clip_norm=float(training["gradient_clip_norm"]),
            )
            selection_metrics = ranking_metrics(
                select_samples,
                model_predictions(model, select_samples, stats, device=device),
                tie_tolerance=tolerance,
            )
            report["history"].append({"epoch": epoch, "train_loss": loss, **selection_metrics})
            if float(selection_metrics["normalized_sb_regret"]) < best_regret:
                best_regret = float(selection_metrics["normalized_sb_regret"])
                best_epoch = epoch
                best_metrics = dict(selection_metrics)
                best_state = copy.deepcopy(model.state_dict())
            atomic_write_json(report_path, report)
        if best_state is None or best_metrics is None:
            raise RuntimeError("formal training produced no checkpoint")
        model.load_state_dict(best_state, strict=True)
        checkpoint_manifest = save_checkpoint_bundle(
            output_dir=checkpoint_dir,
            model=model,
            stats=stats,
            manifest_fields={
                "config_sha256": config_digest,
                "data_manifest_sha256": file_sha256(manifest_path),
                "git_commit": git_commit,
                "training_seed": seed,
                "train_state_count": len(train_samples),
                "validation_metrics": best_metrics,
                "model_config_sha256": file_sha256(b0_path),
                "bipartite_schema_id": config["bipartite_schema_id"],
                "solver_stack_id": config["solver_stack_id"],
                "pytorch_version": torch.__version__,
                "cuda_determinism": cuda_contract,
            },
        )
        restored, restored_stats, _ = load_checkpoint_bundle(
            checkpoint_manifest, model_config_path=b0_path, device=device
        )
        if restored_stats != stats or any(
            not torch.equal(value, restored.state_dict()[name])
            for name, value in model.state_dict().items()
        ):
            raise RuntimeError("formal checkpoint reload is not bit exact")
        original_predictions = model_predictions(model, gate_samples, stats, device=device)
        restored_predictions = model_predictions(restored, gate_samples, restored_stats, device=device)
        reload_error = max(
            float(np.max(np.abs(left - right)))
            for left, right in zip(original_predictions, restored_predictions)
        )
        if reload_error != 0.0:
            raise RuntimeError(f"formal checkpoint reload parity failed: {reload_error}")
        random_predictions = formal_random_predictions(
            gate_samples, seed=int(config["statistics"]["random_baseline_seed"])
        )
        report.update({
            "status": "completed",
            "finished_at_utc": utc_now(),
            "best_epoch": best_epoch,
            "validation_select_metrics": best_metrics,
            "validation_gate_metrics": ranking_metrics(
                gate_samples, original_predictions, tie_tolerance=tolerance
            ),
            "offline_gate_baselines": {
                "random": ranking_metrics(gate_samples, random_predictions, tie_tolerance=tolerance),
                "most_infeasible": ranking_metrics(
                    gate_samples,
                    offline_baseline_predictions(gate_samples, baseline="most_infeasible", seed=0),
                    tie_tolerance=tolerance,
                ),
                "pseudocost": ranking_metrics(
                    gate_samples,
                    offline_baseline_predictions(gate_samples, baseline="pseudocost", seed=0),
                    tie_tolerance=tolerance,
                ),
            },
            "gate_state_records": _state_gate_records(
                gate_samples, gate_records, original_predictions, random_predictions,
                tolerance=tolerance,
            ),
            "checkpoint_manifest": str(checkpoint_manifest),
            "checkpoint_manifest_sha256": file_sha256(checkpoint_manifest),
            "reload_max_absolute_error": reload_error,
            "reload_state_dict_bit_exact": True,
        })
    except BaseException as error:
        report.update({
            "status": "failed",
            "finished_at_utc": utc_now(),
            "error": f"{type(error).__name__}: {error}",
        })
    atomic_write_json(report_path, report)
    print(json.dumps({"status": report["status"], "seed": seed, "report": str(report_path)}))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
