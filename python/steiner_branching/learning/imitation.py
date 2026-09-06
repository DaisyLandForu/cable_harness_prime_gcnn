"""Train-only normalization, listwise IL, ranking metrics, and reload manifests."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from ..models.milp_gcnn import MilpBipartiteGCNN, load_b0_config, score_state
from ..solver.bipartite_observation import MilpBipartiteState, make_bipartite_state
from .teacher_data import TeacherSample, file_sha256


CUBLAS_WORKSPACE_CONFIG = ":4096:8"


@dataclass(frozen=True)
class NormalizationStats:
    constraint_mean: tuple[float, ...]
    constraint_std: tuple[float, ...]
    variable_mean: tuple[float, ...]
    variable_std: tuple[float, ...]
    edge_mean: tuple[float, ...]
    edge_std: tuple[float, ...]
    fitted_state_count: int
    fitted_split: str = "train"

    def __post_init__(self) -> None:
        if len(self.constraint_mean) != 5 or len(self.constraint_std) != 5:
            raise ValueError("constraint normalization must have width 5")
        if len(self.variable_mean) != 19 or len(self.variable_std) != 19:
            raise ValueError("variable normalization must have width 19")
        if len(self.edge_mean) != 1 or len(self.edge_std) != 1:
            raise ValueError("edge normalization must have width 1")
        if self.fitted_split != "train" or self.fitted_state_count < 1:
            raise ValueError("normalization must be fitted from non-empty train states only")
        values = self.constraint_mean + self.constraint_std + self.variable_mean + self.variable_std + self.edge_mean + self.edge_std
        if not all(math.isfinite(value) for value in values):
            raise ValueError("normalization contains a non-finite value")
        if any(value <= 0.0 for value in self.constraint_std + self.variable_std + self.edge_std):
            raise ValueError("normalization standard deviations must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "fitted_split": self.fitted_split,
            "fitted_state_count": self.fitted_state_count,
            "constraint_mean": list(self.constraint_mean),
            "constraint_std": list(self.constraint_std),
            "variable_mean": list(self.variable_mean),
            "variable_std": list(self.variable_std),
            "edge_mean": list(self.edge_mean),
            "edge_std": list(self.edge_std),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NormalizationStats":
        expected = {
            "schema_version", "fitted_split", "fitted_state_count", "constraint_mean",
            "constraint_std", "variable_mean", "variable_std", "edge_mean", "edge_std",
        }
        if set(raw) != expected or raw["schema_version"] != 1:
            raise ValueError("normalization manifest fields/schema mismatch")
        return cls(
            constraint_mean=tuple(float(value) for value in raw["constraint_mean"]),
            constraint_std=tuple(float(value) for value in raw["constraint_std"]),
            variable_mean=tuple(float(value) for value in raw["variable_mean"]),
            variable_std=tuple(float(value) for value in raw["variable_std"]),
            edge_mean=tuple(float(value) for value in raw["edge_mean"]),
            edge_std=tuple(float(value) for value in raw["edge_std"]),
            fitted_state_count=int(raw["fitted_state_count"]),
            fitted_split=str(raw["fitted_split"]),
        )


def _moments(arrays: Sequence[np.ndarray], width: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not arrays or any(array.ndim != 2 or array.shape[1] != width for array in arrays):
        raise ValueError("normalization arrays are empty or have the wrong width")
    joined = np.concatenate(arrays, axis=0).astype(np.float64, copy=False)
    mean = joined.mean(axis=0)
    std = joined.std(axis=0)
    std = np.where(std < 1.0e-8, 1.0, std)
    return tuple(map(float, mean)), tuple(map(float, std))


def fit_train_normalization(samples: Sequence[TeacherSample]) -> NormalizationStats:
    if not samples or any(sample.split != "train" for sample in samples):
        raise ValueError("normalization input must contain train samples only")
    c_mean, c_std = _moments([sample.state.constraint_features for sample in samples], 5)
    v_mean, v_std = _moments([sample.state.variable_features for sample in samples], 19)
    e_mean, e_std = _moments([sample.state.edge_features for sample in samples], 1)
    return NormalizationStats(
        constraint_mean=c_mean,
        constraint_std=c_std,
        variable_mean=v_mean,
        variable_std=v_std,
        edge_mean=e_mean,
        edge_std=e_std,
        fitted_state_count=len(samples),
    )


def normalize_state(state: MilpBipartiteState, stats: NormalizationStats) -> MilpBipartiteState:
    return make_bipartite_state(
        constraint_features=(state.constraint_features - np.asarray(stats.constraint_mean)) / np.asarray(stats.constraint_std),
        variable_features=(state.variable_features - np.asarray(stats.variable_mean)) / np.asarray(stats.variable_std),
        edge_indices=state.edge_indices,
        edge_features=(state.edge_features - np.asarray(stats.edge_mean)) / np.asarray(stats.edge_std),
        variable_names=state.variable_names,
        variable_global_ids=state.variable_global_ids,
        constraint_global_ids=state.constraint_global_ids,
        candidate_indices=state.candidate_indices,
        candidate_edge_ids=state.candidate_edge_ids,
    )


def listwise_cross_entropy(
    logits: torch.Tensor, teacher_scores: np.ndarray, *, temperature: float
) -> torch.Tensor:
    if logits.ndim != 1 or logits.numel() != int(teacher_scores.size) or logits.numel() < 2:
        raise ValueError("listwise logits must align with at least two candidates")
    if temperature <= 0.0 or not np.isfinite(teacher_scores).all():
        raise ValueError("listwise teacher scores/temperature are invalid")
    targets = torch.softmax(
        torch.tensor(
            np.array(teacher_scores, copy=True), dtype=logits.dtype, device=logits.device
        ) / temperature,
        dim=0,
    )
    return -(targets * functional.log_softmax(logits, dim=0)).sum()


def normalized_sb_regret(scores: np.ndarray, selected: int, *, tie_tolerance: float) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("normalized SB regret requires a finite candidate list")
    if not 0 <= selected < values.size:
        raise ValueError("selected candidate is outside the score list")
    span = float(values.max() - values.min())
    scale = max(1.0, float(np.max(np.abs(values))))
    if span <= tie_tolerance * scale:
        return 0.0
    return float((values.max() - values[selected]) / span)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_rank_correlation(predicted: np.ndarray, teacher: np.ndarray) -> float | None:
    predicted = np.asarray(predicted, dtype=np.float64)
    teacher = np.asarray(teacher, dtype=np.float64)
    if predicted.shape != teacher.shape or predicted.ndim != 1 or predicted.size < 2:
        raise ValueError("rank vectors must be aligned one-dimensional candidate lists")
    if not np.isfinite(predicted).all() or not np.isfinite(teacher).all():
        raise ValueError("rank vectors must be finite")
    left = _average_ranks(predicted)
    right = _average_ranks(teacher)
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def ranking_metrics(
    samples: Sequence[TeacherSample],
    predictions: Sequence[np.ndarray],
    *,
    tie_tolerance: float,
) -> dict[str, Any]:
    if len(samples) != len(predictions):
        raise ValueError("samples and predictions must align")
    regrets: list[float] = []
    top1: list[float] = []
    top3: list[float] = []
    correlations: list[float] = []
    for sample, raw_prediction in zip(samples, predictions):
        if not sample.labels.state_valid:
            continue
        teacher = sample.labels.scores
        predicted = np.asarray(raw_prediction, dtype=np.float64)
        if predicted.shape != teacher.shape or not np.isfinite(predicted).all():
            raise ValueError("prediction is non-finite or candidate-misaligned")
        selected = int(np.argmax(predicted))
        scale = max(1.0, float(np.max(np.abs(teacher))))
        optimal = set(np.flatnonzero(teacher.max() - teacher <= tie_tolerance * scale))
        ranking = np.argsort(-predicted, kind="mergesort")
        top1.append(float(selected in optimal))
        top3.append(float(bool(optimal & set(map(int, ranking[: min(3, ranking.size)])))))
        regrets.append(normalized_sb_regret(teacher, selected, tie_tolerance=tie_tolerance))
        correlation = spearman_rank_correlation(predicted, teacher)
        if correlation is not None:
            correlations.append(correlation)
    if not regrets:
        raise ValueError("no valid teacher states for ranking metrics")
    return {
        "states": len(regrets),
        "top1_accuracy": float(np.mean(top1)),
        "top3_accuracy": float(np.mean(top3)),
        "normalized_sb_regret": float(np.mean(regrets)),
        "rank_correlation": float(np.mean(correlations)) if correlations else None,
        "rank_correlation_states": len(correlations),
    }


def offline_baseline_predictions(
    samples: Sequence[TeacherSample], *, baseline: str, seed: int
) -> list[np.ndarray]:
    generator = np.random.default_rng(seed)
    predictions: list[np.ndarray] = []
    for sample in samples:
        n = sample.state.candidate_count
        if baseline == "random":
            predictions.append(generator.random(n))
        elif baseline == "most_infeasible":
            fractions = sample.state.variable_features[sample.state.candidate_indices, 9]
            predictions.append(-np.abs(fractions.astype(np.float64) - 0.5))
        elif baseline == "pseudocost":
            values = np.asarray(sample.pseudocost_scores, dtype=np.float64)
            if not np.isfinite(values).any():
                predictions.append(np.zeros(n, dtype=np.float64))
            else:
                floor = float(np.nanmin(values[np.isfinite(values)])) - 1.0
                predictions.append(np.nan_to_num(values, nan=floor))
        else:
            raise ValueError(f"unknown offline baseline: {baseline}")
    return predictions


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def enable_cuda_determinism(*, expected_workspace_config: str) -> dict[str, Any]:
    """Enable and verify the frozen S05 CUDA determinism contract."""
    if expected_workspace_config != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError("configured cuBLAS workspace policy is not registered")
    observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if observed != expected_workspace_config:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be set to "
            f"{expected_workspace_config!r} before the Python process starts; got {observed!r}"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("PyTorch deterministic algorithms did not remain enabled")
    return {
        "cublas_workspace_config": observed,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }


def train_one_epoch(
    model: MilpBipartiteGCNN,
    samples: Sequence[TeacherSample],
    stats: NormalizationStats,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    temperature: float,
    gradient_clip_norm: float,
) -> float:
    eligible = [sample for sample in samples if sample.labels.state_valid]
    if not eligible:
        raise ValueError("training set contains no valid teacher states")
    model.train()
    losses: list[float] = []
    for sample in eligible:
        optimizer.zero_grad(set_to_none=True)
        logits = score_state(model, normalize_state(sample.state, stats), device=device)
        loss = listwise_cross_entropy(logits, sample.labels.scores, temperature=temperature)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite S05 training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def model_predictions(
    model: MilpBipartiteGCNN,
    samples: Sequence[TeacherSample],
    stats: NormalizationStats,
    *,
    device: torch.device,
) -> list[np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for sample in samples:
            logits = score_state(model, normalize_state(sample.state, stats), device=device)
            values = logits.detach().cpu().numpy().astype(np.float64, copy=True)
            if not np.isfinite(values).all():
                raise FloatingPointError("non-finite S05 model prediction")
            predictions.append(values)
    return predictions


def atomic_write_json(path: Path | str, value: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def save_checkpoint_bundle(
    *,
    output_dir: Path | str,
    model: MilpBipartiteGCNN,
    stats: NormalizationStats,
    manifest_fields: Mapping[str, Any],
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "model.pt"
    temporary = directory / ".model.pt.tmp"
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, checkpoint)
    normalization = directory / "normalization.json"
    atomic_write_json(normalization, stats.to_dict())
    manifest = {
        "schema_version": 2 if "cuda_determinism" in manifest_fields else 1,
        "model_id": "b0_milp_gcnn_v1",
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_sha256(checkpoint),
        "normalization": normalization.name,
        "normalization_sha256": file_sha256(normalization),
        **dict(manifest_fields),
    }
    atomic_write_json(directory / "manifest.json", manifest)
    return directory / "manifest.json"


def load_checkpoint_bundle(
    manifest_path: Path | str,
    *,
    model_config_path: Path | str,
    device: torch.device,
) -> tuple[MilpBipartiteGCNN, NormalizationStats, dict[str, Any]]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required_v1 = {
        "schema_version", "model_id", "checkpoint", "checkpoint_sha256",
        "normalization", "normalization_sha256", "config_sha256", "data_manifest_sha256",
        "git_commit", "training_seed", "train_state_count", "validation_metrics",
        "model_config_sha256", "bipartite_schema_id", "solver_stack_id", "pytorch_version",
    }
    required_v2 = required_v1 | {"cuda_determinism"}
    expected_fields = required_v2 if manifest.get("schema_version") == 2 else required_v1
    if (
        set(manifest) != expected_fields
        or manifest.get("schema_version") not in {1, 2}
        or manifest.get("model_id") != "b0_milp_gcnn_v1"
    ):
        raise ValueError("checkpoint manifest fields/schema mismatch")
    checkpoint = path.parent / manifest["checkpoint"]
    normalization = path.parent / manifest["normalization"]
    if file_sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    if file_sha256(normalization) != manifest["normalization_sha256"]:
        raise ValueError("normalization checksum mismatch")
    if file_sha256(model_config_path) != manifest["model_config_sha256"]:
        raise ValueError("model config checksum mismatch")
    if manifest["bipartite_schema_id"] != "milp_bipartite_v1":
        raise ValueError("checkpoint bipartite schema mismatch")
    if manifest["solver_stack_id"] != "scip804-ecole081-pyscipopt430":
        raise ValueError("checkpoint solver stack mismatch")
    if manifest["pytorch_version"] != torch.__version__:
        raise ValueError("checkpoint PyTorch version mismatch")
    if manifest["schema_version"] == 2:
        expected_determinism = {
            "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        }
        if manifest["cuda_determinism"] != expected_determinism:
            raise ValueError("checkpoint CUDA determinism contract mismatch")
    stats = NormalizationStats.from_dict(json.loads(normalization.read_text(encoding="utf-8")))
    config = load_b0_config(model_config_path)
    architecture = config["architecture"]
    model = MilpBipartiteGCNN(
        embedding_dim=int(architecture["embedding_dim"]),
        hidden_dim=int(architecture["hidden_dim"]),
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, stats, manifest
