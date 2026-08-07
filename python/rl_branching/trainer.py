import csv
import json
import os
import random
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import torch
import yaml

from .candidate_features import (
    AVIATION_VARIABLE_CATEGORIES,
    ECOLE_VARIABLE_FEATURE_NAMES,
    CandidateState,
    RunningFeatureNormalizer,
    extract_candidate_state,
)
from .candidate_model import CandidateQNetwork, export_torchscript
from .dqn import DoubleDQNLearner, UpdateMetrics
from .environment import BBMDPBranchingEnv
from .observation import GLOBAL_FEATURE_NAMES
from .replay import NStepAccumulator, OneStepExperience, ReplayBuffer
from .training_config import MLPTrainingConfig


HISTORY_FIELDS = (
    "event",
    "episode",
    "gradient_step",
    "loss",
    "td_error",
    "epsilon",
    "reward",
    "episode_nodes",
    "episode_solving_time",
    "validation_nodes",
    "validation_time",
    "replay_size",
    "q_value_mean",
    "q_value_std",
    "selected_candidate_rank",
    "cpu_memory_mb",
    "gpu_memory_mb",
    "status",
    "instance",
)


@dataclass(frozen=True)
class EpisodeMetrics:
    method: str
    instance: str
    seed: int
    policy_seed: int
    status: str
    nodes: int
    solving_time: float
    reward: float
    transitions: int
    mean_rank: float


class HistoryWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("w", newline="")
        self._writer = csv.DictWriter(self._stream, fieldnames=HISTORY_FIELDS)
        self._writer.writeheader()

    def write(self, **values) -> None:
        row = {field: values.get(field, "") for field in HISTORY_FIELDS}
        self._writer.writerow(row)
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


def _memory_metrics(device: torch.device) -> tuple[float, float]:
    cpu_mb = psutil.Process().memory_info().rss / 1048576
    gpu_mb = torch.cuda.memory_allocated(device) / 1048576 if device.type == "cuda" else 0.0
    return cpu_mb, gpu_mb


def _set_deterministic(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _compact_next(transition) -> Optional[CandidateState]:
    if transition.next_observation is None or transition.next_action_set.size == 0:
        return None
    return extract_candidate_state(transition.next_observation, transition.next_action_set)


def evaluate_episode(
    instance: str,
    seed: int,
    environment_config,
    method: str,
    learner: Optional[DoubleDQNLearner],
    policy_seed: int,
) -> EpisodeMetrics:
    rng = np.random.default_rng(policy_seed)
    env = BBMDPBranchingEnv(replace(environment_config, seed=int(seed)))
    state = env.reset(instance)
    total_reward = 0.0
    transitions = 0
    ranks: list[float] = []
    while not (state.terminated or state.truncated):
        compact = extract_candidate_state(state.observation, state.action_set)
        if method == "random":
            position = int(rng.integers(compact.candidate_count))
            action = int(compact.actions[position])
        else:
            if learner is None:
                raise ValueError(f"method {method} requires a learner")
            action, position, rank, _ = learner.select_action(compact, epsilon=0.0, rng=rng)
            ranks.append(rank)
        transition = env.step(action)
        total_reward += transition.reward
        transitions += 1
        state = env.current_state
    metrics = EpisodeMetrics(
        method=method,
        instance=str(instance),
        seed=int(seed),
        policy_seed=int(policy_seed),
        status=str(state.info.get("status", "unknown")),
        nodes=int(state.info.get("node_count", 0)),
        solving_time=float(state.info.get("solving_time", 0.0)),
        reward=total_reward,
        transitions=transitions,
        mean_rank=float(np.mean(ranks)) if ranks else 0.0,
    )
    env.close()
    return metrics


def _evaluation_policy_seed(base_seed: int, solver_seed: int, method: str, instance_index: int) -> int:
    method_offsets = {"random": 11, "untrained": 23, "rl": 37}
    if method not in method_offsets:
        raise ValueError(f"unknown evaluation method {method}")
    return int(base_seed * 1_000_003 + solver_seed * 101 + instance_index * 10_007 + method_offsets[method])


def _save_checkpoint(path: Path, learner: DoubleDQNLearner, config: MLPTrainingConfig) -> None:
    torch.save(
        {
            "model_state_dict": learner.online.state_dict(),
            "target_state_dict": learner.target.state_dict(),
            "optimizer_state_dict": learner.optimizer.state_dict(),
            "gradient_step": learner.gradient_step,
            "hidden_sizes": list(config.model.hidden_sizes),
            "config": config.to_dict(),
        },
        path,
    )


def _load_model(path: Path, device: torch.device) -> CandidateQNetwork:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = CandidateQNetwork(checkpoint["hidden_sizes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _feature_schema() -> dict:
    return {
        "schema_version": 2,
        "candidate_variable_features": list(ECOLE_VARIABLE_FEATURE_NAMES),
        "global_features": list(GLOBAL_FEATURE_NAMES),
        "aviation_variable_categories": list(AVIATION_VARIABLE_CATEGORIES),
        "input_width": len(ECOLE_VARIABLE_FEATURE_NAMES)
        + len(GLOBAL_FEATURE_NAMES)
        + len(AVIATION_VARIABLE_CATEGORIES),
        "action_semantics": "one scalar Q per current fractional LP candidate",
        "category_name_rule": "strip repeated t_ prefixes, then match <category>_",
        "excluded_features": {
            "solving_time": "wall-clock dependent and not reproducible for an identical SCIP state"
        },
    }


def train_candidate_mlp(config_path: Path | str) -> dict:
    config_path = Path(config_path)
    config = MLPTrainingConfig.from_yaml(config_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output_dir / "config.yaml")
    (output_dir / "feature_schema.json").write_text(json.dumps(_feature_schema(), indent=2) + "\n")

    _set_deterministic(config.seed)
    device = _resolve_device(config.device)
    rng = np.random.default_rng(config.seed)
    online = CandidateQNetwork(config.model.hidden_sizes)
    target = CandidateQNetwork(config.model.hidden_sizes)
    learner = DoubleDQNLearner(
        online=online,
        target=target,
        device=device,
        learning_rate=config.optimization.learning_rate,
        gamma=config.optimization.gamma,
        gradient_clip=config.optimization.gradient_clip,
        target_update_interval=config.optimization.target_update_interval,
    )
    replay = ReplayBuffer(config.optimization.replay_capacity, config.seed)
    normalizer = RunningFeatureNormalizer()
    normalizer_frozen = False
    observed_states = 0
    history = HistoryWriter(output_dir / "training_history.csv")
    latest_update: Optional[UpdateMetrics] = None
    episode = 0
    best_validation = float("inf")
    validations_without_improvement = 0
    next_validation = config.evaluation.interval_steps
    start_wall = time.monotonic()

    try:
        while learner.gradient_step < config.optimization.total_gradient_steps:
            instance = config.train_instances[episode % len(config.train_instances)]
            env_seed = config.seed + episode
            env = BBMDPBranchingEnv(replace(config.environment, seed=env_seed))
            state = env.reset(instance)
            accumulator = NStepAccumulator(config.optimization.n_step, config.optimization.gamma)
            episode_reward = 0.0
            episode_ranks: list[float] = []
            while not (state.terminated or state.truncated):
                compact = extract_candidate_state(state.observation, state.action_set)
                if not normalizer_frozen:
                    normalizer.update(compact)
                    observed_states += 1
                    if observed_states >= config.normalization_warmup_states:
                        statistics = normalizer.statistics()
                        learner.online.set_normalization(statistics)
                        learner.target.set_normalization(statistics)
                        normalizer_frozen = True

                epsilon = config.exploration.epsilon(learner.gradient_step)
                action, position, rank, q_values = learner.select_action(compact, epsilon, rng)
                episode_ranks.append(rank)
                transition = env.step(action)
                next_compact = _compact_next(transition)
                emitted = accumulator.append(
                    OneStepExperience(
                        state=compact,
                        action_position=position,
                        reward=transition.reward,
                        next_state=next_compact,
                        bootstrap_mask=transition.bootstrap_mask,
                    )
                )
                for experience in emitted:
                    replay.add(experience)
                episode_reward += transition.reward
                state = env.current_state

                if normalizer_frozen and len(replay) >= config.optimization.min_replay_size:
                    for _ in range(config.optimization.updates_per_env_step):
                        if learner.gradient_step >= config.optimization.total_gradient_steps:
                            break
                        latest_update = learner.update(replay.sample(config.optimization.batch_size))
                        if learner.gradient_step % config.log_interval_steps == 0:
                            cpu_mb, gpu_mb = _memory_metrics(device)
                            history.write(
                                event="train_update",
                                episode=episode,
                                gradient_step=learner.gradient_step,
                                loss=latest_update.loss,
                                td_error=latest_update.td_error,
                                epsilon=epsilon,
                                replay_size=len(replay),
                                q_value_mean=latest_update.q_mean,
                                q_value_std=latest_update.q_std,
                                selected_candidate_rank=rank,
                                cpu_memory_mb=cpu_mb,
                                gpu_memory_mb=gpu_mb,
                                instance=instance,
                            )
            final_info = state.info
            env.close()
            cpu_mb, gpu_mb = _memory_metrics(device)
            history.write(
                event="episode",
                episode=episode,
                gradient_step=learner.gradient_step,
                loss="" if latest_update is None else latest_update.loss,
                td_error="" if latest_update is None else latest_update.td_error,
                epsilon=config.exploration.epsilon(learner.gradient_step),
                reward=episode_reward,
                episode_nodes=final_info.get("node_count", 0),
                episode_solving_time=final_info.get("solving_time", 0.0),
                replay_size=len(replay),
                selected_candidate_rank=float(np.mean(episode_ranks)) if episode_ranks else 0.0,
                cpu_memory_mb=cpu_mb,
                gpu_memory_mb=gpu_mb,
                status=final_info.get("status", "unknown"),
                instance=instance,
            )
            episode += 1

            if learner.gradient_step >= next_validation or learner.gradient_step >= config.optimization.total_gradient_steps:
                validation = []
                for instance_index, validation_instance in enumerate(config.validation_instances):
                    for seed in config.evaluation.seeds:
                        validation.append(
                            evaluate_episode(
                                validation_instance,
                                seed,
                                config.environment,
                                "rl",
                                learner,
                                _evaluation_policy_seed(config.seed, seed, "rl", instance_index),
                            )
                        )
                validation_nodes = float(np.mean([result.nodes for result in validation]))
                validation_time = float(np.mean([result.solving_time for result in validation]))
                history.write(
                    event="validation",
                    episode=episode,
                    gradient_step=learner.gradient_step,
                    validation_nodes=validation_nodes,
                    validation_time=validation_time,
                    replay_size=len(replay),
                    status=";".join(result.status for result in validation),
                    instance=";".join(config.validation_instances),
                )
                if validation_nodes < best_validation:
                    best_validation = validation_nodes
                    validations_without_improvement = 0
                    _save_checkpoint(output_dir / "best_model.pt", learner, config)
                    export_torchscript(learner.online, output_dir / "best_model_scripted.pt")
                else:
                    validations_without_improvement += 1
                next_validation += config.evaluation.interval_steps
                if validations_without_improvement >= config.evaluation.early_stopping_patience:
                    break
    finally:
        history.close()

    if not normalizer_frozen:
        statistics = normalizer.statistics()
        learner.online.set_normalization(statistics)
        learner.target.set_normalization(statistics)
    (output_dir / "normalization.json").write_text(json.dumps(normalizer.to_json(), indent=2) + "\n")
    _save_checkpoint(output_dir / "last_model.pt", learner, config)
    export_torchscript(learner.online, output_dir / "last_model_scripted.pt")
    if not (output_dir / "best_model.pt").exists():
        _save_checkpoint(output_dir / "best_model.pt", learner, config)
        export_torchscript(learner.online, output_dir / "best_model_scripted.pt")

    best_model = _load_model(output_dir / "best_model.pt", device)
    best_target = CandidateQNetwork(config.model.hidden_sizes).to(device)
    best_learner = DoubleDQNLearner(
        best_model,
        best_target,
        device,
        config.optimization.learning_rate,
        config.optimization.gamma,
        config.optimization.gradient_clip,
        config.optimization.target_update_interval,
    )
    torch.manual_seed(config.seed + 1_000_000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed + 1_000_000)
    untrained_model = CandidateQNetwork(config.model.hidden_sizes).to(device)
    untrained_model.set_normalization(normalizer.statistics())
    untrained_target = CandidateQNetwork(config.model.hidden_sizes).to(device)
    untrained_learner = DoubleDQNLearner(
        untrained_model,
        untrained_target,
        device,
        config.optimization.learning_rate,
        config.optimization.gamma,
        config.optimization.gradient_clip,
        config.optimization.target_update_interval,
    )

    comparisons = []
    for instance_index, instance in enumerate(config.validation_instances):
        for seed in config.evaluation.compare_seeds:
            for method, evaluation_learner in (
                ("random", None),
                ("untrained", untrained_learner),
                ("rl", best_learner),
            ):
                comparisons.append(
                    evaluate_episode(
                        instance,
                        seed,
                        config.environment,
                        method,
                        evaluation_learner,
                        _evaluation_policy_seed(config.seed, seed, method, instance_index),
                    )
                )
    with (output_dir / "evaluation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EpisodeMetrics.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(result.__dict__ for result in comparisons)

    summary = {
        "run_name": config.run_name,
        "gradient_steps": learner.gradient_step,
        "episodes": episode,
        "replay_size": len(replay),
        "best_validation_nodes": best_validation,
        "wall_time": time.monotonic() - start_wall,
        "device": str(device),
        "torch_version": torch.__version__,
        "evaluation": [result.__dict__ for result in comparisons],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def reevaluate_candidate_mlp(artifact_dir: Path | str) -> dict:
    evaluation_start = time.monotonic()
    artifact_dir = Path(artifact_dir)
    config = MLPTrainingConfig.from_yaml(artifact_dir / "config.yaml")
    _set_deterministic(config.seed)
    device = _resolve_device(config.device)

    best_model = _load_model(artifact_dir / "best_model.pt", device)
    best_target = CandidateQNetwork(config.model.hidden_sizes).to(device)
    best_learner = DoubleDQNLearner(
        best_model,
        best_target,
        device,
        config.optimization.learning_rate,
        config.optimization.gamma,
        config.optimization.gradient_clip,
        config.optimization.target_update_interval,
    )
    normalization = {
        "variable_mean": best_model.variable_mean.detach().cpu().numpy(),
        "variable_std": best_model.variable_std.detach().cpu().numpy(),
        "global_mean": best_model.global_mean.detach().cpu().numpy(),
        "global_std": best_model.global_std.detach().cpu().numpy(),
    }

    torch.manual_seed(config.seed + 1_000_000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed + 1_000_000)
    untrained_model = CandidateQNetwork(config.model.hidden_sizes).to(device)
    untrained_model.set_normalization(normalization)
    untrained_target = CandidateQNetwork(config.model.hidden_sizes).to(device)
    untrained_learner = DoubleDQNLearner(
        untrained_model,
        untrained_target,
        device,
        config.optimization.learning_rate,
        config.optimization.gamma,
        config.optimization.gradient_clip,
        config.optimization.target_update_interval,
    )

    comparisons = []
    for instance_index, instance in enumerate(config.validation_instances):
        for seed in config.evaluation.compare_seeds:
            for method, evaluation_learner in (
                ("random", None),
                ("untrained", untrained_learner),
                ("rl", best_learner),
            ):
                policy_seed = _evaluation_policy_seed(config.seed, seed, method, instance_index)
                comparisons.append(
                    evaluate_episode(
                        instance,
                        seed,
                        config.environment,
                        method,
                        evaluation_learner,
                        policy_seed,
                    )
                )

    with (artifact_dir / "evaluation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EpisodeMetrics.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(result.__dict__ for result in comparisons)
    history_path = artifact_dir / "training_history.csv"
    with history_path.open() as stream:
        history_rows = list(csv.DictReader(stream))
    episode_rows = [row for row in history_rows if row["event"] == "episode"]
    validation_rows = [row for row in history_rows if row["event"] == "validation"]
    last_checkpoint = torch.load(
        artifact_dir / "last_model.pt", map_location="cpu", weights_only=False
    )
    summary_path = artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.pop("wall_time", None)
    summary.update(
        {
            "run_name": config.run_name,
            "gradient_steps": int(last_checkpoint["gradient_step"]),
            "episodes": len(episode_rows),
            "replay_size": int(episode_rows[-1]["replay_size"]),
            "best_validation_nodes": min(
                float(row["validation_nodes"]) for row in validation_rows
            ),
            "device": config.device,
            "torch_version": torch.__version__,
            "training_wall_time_estimate": (
                history_path.stat().st_mtime
                - (artifact_dir / "feature_schema.json").stat().st_mtime
            ),
            "evaluation_wall_time": time.monotonic() - evaluation_start,
            "summary_reconstructed_from_artifacts": True,
        }
    )
    summary["evaluation_protocol"] = {
        "policy_seed_formula": "base_seed*1000003 + solver_seed*101 + instance_index*10007 + method_offset",
        "method_offsets": {"random": 11, "untrained": 23, "rl": 37},
        "untrained_initialization_seed": config.seed + 1_000_000,
    }
    summary["evaluation"] = [result.__dict__ for result in comparisons]
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary
