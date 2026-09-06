#!/usr/bin/env python3
"""Train the preregistered multi-seed S05 B0 listwise imitation pilot."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
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

from steiner_branching.learning.imitation import (  # noqa: E402
    atomic_write_json,
    fit_train_normalization,
    load_checkpoint_bundle,
    model_predictions,
    offline_baseline_predictions,
    ranking_metrics,
    save_checkpoint_bundle,
    seed_everything,
    train_one_epoch,
)
from steiner_branching.learning.teacher_data import (  # noqa: E402
    assert_no_split_leakage,
    file_sha256,
    load_s05_config,
    load_teacher_sample,
    s05_config_sha256,
)
from steiner_branching.models.milp_gcnn import (  # noqa: E402
    MilpBipartiteGCNN,
    load_b0_config,
)


DEFAULT_CONFIG = REPO / "configs/steiner/experiments/s05_teacher_il_pilot_v2.yml"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-manifest")
    parser.add_argument(
        "--training-seed",
        action="append",
        type=int,
        dest="training_seeds",
        help=(
            "run only this preregistered pilot seed; repeat to place multiple "
            "disjoint seeds in one job"
        ),
    )
    return parser.parse_args()


def select_training_seeds(
    registered: list[int], requested: list[int] | None
) -> tuple[int, ...]:
    expected = tuple(int(seed) for seed in registered)
    if requested is None:
        return expected
    selected = tuple(int(seed) for seed in requested)
    if not selected:
        raise ValueError("at least one training seed must be selected")
    if len(set(selected)) != len(selected):
        raise ValueError("training seed selection contains duplicates")
    unknown = sorted(set(selected) - set(expected))
    if unknown:
        raise ValueError(f"training seeds are not registered for the pilot: {unknown}")
    return selected


def training_report_path(
    report_root: Path, selected: tuple[int, ...], expected: tuple[int, ...]
) -> Path:
    if selected == expected:
        return report_root / "pilot_training_report.json"
    suffix = "-".join(str(seed) for seed in sorted(selected))
    return report_root / "pilot_training_shards" / f"seeds-{suffix}.json"


def require_current_git_identity(tag: str) -> tuple[str, str]:
    tag_process = subprocess.run(
        ["git", "rev-parse", f"refs/tags/{tag}^{{}}"], cwd=REPO,
        text=True, capture_output=True, check=False,
    )
    if tag_process.returncode != 0:
        raise RuntimeError(f"required audited tag {tag!r} is missing")
    tag_target = tag_process.stdout.strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", tag_target, "HEAD"], cwd=REPO, check=False
    ).returncode != 0:
        raise RuntimeError(f"required audited tag {tag!r} is outside current history")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=True
    ).stdout.strip()
    return head, tag_target


def load_dataset(manifest_path: Path) -> tuple[dict[str, Any], list[Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("stage") != "S05":
        raise ValueError("teacher manifest schema/stage mismatch")
    if manifest.get("status") != "completed" or manifest.get("formal_gate_evaluated") is not False:
        raise ValueError("teacher pilot manifest is incomplete or misclassified")
    samples = []
    for item in manifest.get("shards", []):
        sample = load_teacher_sample(
            manifest_path.parent / item["path"],
            expected_file_sha256=item["file_sha256"],
        )
        if sample.semantic_sha256 != item["semantic_sha256"]:
            raise ValueError("teacher manifest semantic checksum mismatch")
        samples.append(sample)
    assert_no_split_leakage(samples)
    if any(sample.split not in {"train", "validation_iid"} for sample in samples):
        raise ValueError("S05 training encountered a forbidden split")
    return manifest, samples


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_s05_config(config_path)
    config_digest = s05_config_sha256(config)
    git_commit, audited_target = require_current_git_identity(
        str(config["required_s04_audited_tag"])
    )
    manifest_path = (
        resolve_path(args.data_manifest)
        if args.data_manifest
        else resolve_path(config["artifacts"]["raw_root"]) / "manifest.json"
    )
    data_manifest, samples = load_dataset(manifest_path)
    if data_manifest.get("config_sha256") != config_digest:
        raise ValueError("teacher data/config fingerprint mismatch")
    if data_manifest.get("git_commit") != git_commit:
        raise ValueError("teacher data was collected from a different code commit")
    if data_manifest.get("s04_audited_tag_target") != audited_target:
        raise ValueError("teacher data was collected under a different S04 audit target")
    data_digest = file_sha256(manifest_path)
    train_samples = sorted(
        (sample for sample in samples if sample.split == "train" and sample.labels.state_valid),
        key=lambda sample: (sample.graph_sha256, sample.teacher_seed, sample.state_index),
    )
    validation_samples = sorted(
        (sample for sample in samples if sample.split == "validation_iid" and sample.labels.state_valid),
        key=lambda sample: (sample.graph_sha256, sample.teacher_seed, sample.state_index),
    )
    if not validation_samples:
        raise RuntimeError("S05 pilot has no valid validation teacher states")
    training = config["training"]
    expected_training_seeds = tuple(int(seed) for seed in training["pilot_seeds"])
    selected_training_seeds = select_training_seeds(
        training["pilot_seeds"], args.training_seeds
    )
    if training["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("S05 training requires the configured CUDA environment")
    device = torch.device("cuda:0")
    b0_config_path = resolve_path(config["model_config"])
    b0_config = load_b0_config(b0_config_path)
    architecture = b0_config["architecture"]
    tolerance = float(config["teacher"]["tie_relative_tolerance"])
    report: dict[str, Any] = {
        "schema_version": 1,
        "stage": "S05",
        "experiment_id": config["experiment_id"],
        "started_at_utc": utc_now(),
        "git_commit": git_commit,
        "s04_audited_tag_target": audited_target,
        "config_sha256": config_digest,
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": data_digest,
        "train_valid_states": len(train_samples),
        "validation_valid_states": len(validation_samples),
        "report_kind": (
            "pilot_complete"
            if selected_training_seeds == expected_training_seeds
            else "pilot_seed_shard"
        ),
        "selected_training_seeds": list(selected_training_seeds),
        "expected_training_seeds": list(expected_training_seeds),
        "offline_baselines": {},
        "runs": [],
        "formal_gate_evaluated": False,
    }
    for baseline in ("random", "most_infeasible", "pseudocost"):
        report["offline_baselines"][baseline] = ranking_metrics(
            validation_samples,
            offline_baseline_predictions(validation_samples, baseline=baseline, seed=20260902),
            tie_tolerance=tolerance,
        )

    checkpoint_root = resolve_path(training["checkpoint_root"])
    failures = 0
    for state_count in training["learning_curve_train_states"]:
        state_count = int(state_count)
        if len(train_samples) < state_count:
            for seed in selected_training_seeds:
                report["runs"].append(
                    {
                        "training_seed": int(seed),
                        "train_state_count": state_count,
                        "status": "skipped_insufficient_train_states",
                    }
                )
                failures += 1
            continue
        selected_train = train_samples[:state_count]
        stats = fit_train_normalization(selected_train)
        for seed in selected_training_seeds:
            seed = int(seed)
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
            best_regret = math.inf
            best_epoch = -1
            best_metrics: dict[str, Any] | None = None
            best_state: dict[str, torch.Tensor] | None = None
            history: list[dict[str, Any]] = []
            try:
                for epoch in range(int(training["epochs"])):
                    loss = train_one_epoch(
                        model,
                        selected_train,
                        stats,
                        optimizer,
                        device=device,
                        temperature=float(training["target_temperature"]),
                        gradient_clip_norm=float(training["gradient_clip_norm"]),
                    )
                    metrics = ranking_metrics(
                        validation_samples,
                        model_predictions(model, validation_samples, stats, device=device),
                        tie_tolerance=tolerance,
                    )
                    history.append({"epoch": epoch, "train_loss": loss, **metrics})
                    if metrics["normalized_sb_regret"] < best_regret:
                        best_regret = float(metrics["normalized_sb_regret"])
                        best_epoch = epoch
                        best_metrics = dict(metrics)
                        best_state = copy.deepcopy(model.state_dict())
                if best_state is None or best_metrics is None:
                    raise RuntimeError("training produced no eligible validation checkpoint")
                model.load_state_dict(best_state, strict=True)
                output_dir = checkpoint_root / f"states-{state_count:06d}" / f"seed-{seed:03d}"
                checkpoint_manifest = save_checkpoint_bundle(
                    output_dir=output_dir,
                    model=model,
                    stats=stats,
                    manifest_fields={
                        "config_sha256": config_digest,
                        "data_manifest_sha256": data_digest,
                        "git_commit": git_commit,
                        "training_seed": seed,
                        "train_state_count": state_count,
                        "validation_metrics": best_metrics,
                        "model_config_sha256": file_sha256(b0_config_path),
                        "bipartite_schema_id": config["bipartite_schema_id"],
                        "solver_stack_id": config["solver_stack_id"],
                        "pytorch_version": torch.__version__,
                    },
                )
                reloaded, reloaded_stats, _reload_manifest = load_checkpoint_bundle(
                    checkpoint_manifest, model_config_path=b0_config_path, device=device
                )
                original = model_predictions(model, validation_samples[:1], stats, device=device)[0]
                restored = model_predictions(
                    reloaded, validation_samples[:1], reloaded_stats, device=device
                )[0]
                reload_error = float(np.max(np.abs(original - restored)))
                if reload_error != 0.0:
                    raise RuntimeError(f"checkpoint reload parity failed: {reload_error}")
                report["runs"].append(
                    {
                        "training_seed": seed,
                        "train_state_count": state_count,
                        "status": "completed",
                        "best_epoch": best_epoch,
                        "validation_metrics": best_metrics,
                        "checkpoint_manifest": str(checkpoint_manifest),
                        "checkpoint_manifest_sha256": file_sha256(checkpoint_manifest),
                        "reload_max_absolute_error": reload_error,
                        "history": history,
                    }
                )
            except BaseException as error:
                report["runs"].append(
                    {
                        "training_seed": seed,
                        "train_state_count": state_count,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                        "history": history,
                    }
                )
                failures += 1
    report["finished_at_utc"] = utc_now()
    report["status"] = "completed" if failures == 0 else "failed"
    report_path = training_report_path(
        resolve_path(config["artifacts"]["report_root"]),
        selected_training_seeds,
        expected_training_seeds,
    )
    atomic_write_json(report_path, report)
    print(json.dumps({"status": report["status"], "runs": len(report["runs"]), "report": str(report_path)}, sort_keys=True))
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
