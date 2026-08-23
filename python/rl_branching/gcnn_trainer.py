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

from .candidate_features import AVIATION_VARIABLE_CATEGORIES
from .graph_features import GRAPH_VARIABLE_FEATURE_NAMES
from .environment import BBMDPBranchingEnv
from .gcnn_config import GCNNTrainingConfig
from .gcnn_dqn import GraphDoubleDQNLearner, GraphUpdateMetrics
from .gcnn_model import BipartiteGCNNQNetwork, export_gcnn_torchscript
from .graph_features import (
    AVIATION_CONSTRAINT_CATEGORIES,
    GraphState,
    RunningGraphNormalizer,
    training_graph_state,
)
from .graph_replay import DualPoolGraphReplay
from .observation import EDGE_FEATURE_NAMES, EXTENDED_ROW_FEATURE_NAMES, GLOBAL_FEATURE_NAMES
from .replay import NStepAccumulator, OneStepExperience


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
    "forward_time",
    "cpu_memory_mb",
    "gpu_memory_mb",
    "status",
    "instance",
)


@dataclass(frozen=True)
class GraphEpisodeMetrics:
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
    inference_time: float


class HistoryWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=HISTORY_FIELDS)
        self._writer.writeheader()

    def write(self, **values) -> None:
        self._writer.writerow({field: values.get(field, "") for field in HISTORY_FIELDS})
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


def _set_deterministic(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _pool_for_instance(instance: str) -> str:
    return "large" if Path(instance).stem.startswith("real_02") else "medium"


def _can_sample_logical_batch(replay: DualPoolGraphReplay) -> bool:
    if len(replay.large) >= replay.large_sample_quota:
        return len(replay.medium) >= replay.medium_sample_quota
    return len(replay.medium) >= replay.medium_sample_quota + replay.large_sample_quota


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for GCNN but unavailable")
    return torch.device(name)


def _memory(device: torch.device) -> tuple[float, float]:
    cpu = psutil.Process().memory_info().rss / 1048576
    gpu = torch.cuda.memory_allocated(device) / 1048576 if device.type == "cuda" else 0.0
    return cpu, gpu


def _model(config: GCNNTrainingConfig) -> BipartiteGCNNQNetwork:
    return BipartiteGCNNQNetwork(
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        distributional_bins=config.model.output_bins,
        z_min=config.model.z_min,
        z_max=config.model.z_max,
        use_aviation_categories=config.model.use_aviation_categories,
        use_global_features=config.model.use_global_features,
    )


def _learner(config: GCNNTrainingConfig, device: torch.device) -> GraphDoubleDQNLearner:
    return GraphDoubleDQNLearner(
        online=_model(config),
        target=_model(config),
        device=device,
        learning_rate=config.optimization.learning_rate,
        gamma=config.optimization.gamma,
        gradient_clip=config.optimization.gradient_clip,
        target_tau=config.optimization.target_tau,
        hl_gauss_sigma=config.model.hl_gauss_sigma,
    )


def _save_checkpoint(
    path: Path,
    learner: GraphDoubleDQNLearner,
    config: GCNNTrainingConfig,
) -> None:
    torch.save(
        {
            "model_state_dict": learner.online.state_dict(),
            "target_state_dict": learner.target.state_dict(),
            "optimizer_state_dict": learner.optimizer.state_dict(),
            "gradient_step": learner.gradient_step,
            "config": config.to_dict(),
        },
        path,
    )


def load_gcnn_model(
    checkpoint_path: Path | str,
    config: GCNNTrainingConfig,
    device: torch.device,
) -> BipartiteGCNNQNetwork:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = _model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _schema(config: GCNNTrainingConfig) -> dict:
    return {
        "schema_version": 1,
        "architecture": "variable-to-constraint then constraint-to-variable index_add message passing",
        "variable_features": list(GRAPH_VARIABLE_FEATURE_NAMES),
        "constraint_features": list(EXTENDED_ROW_FEATURE_NAMES),
        "edge_features": list(EDGE_FEATURE_NAMES),
        "global_features": list(GLOBAL_FEATURE_NAMES),
        "aviation_variable_categories": list(AVIATION_VARIABLE_CATEGORIES),
        "aviation_constraint_categories": list(AVIATION_CONSTRAINT_CATEGORIES),
        "action_semantics": "one scalar Q per current fractional LP candidate",
        "candidate_indices_reference": "transformed SCIP variable order",
        "loss_mode": config.model.loss_mode,
        "distributional_bins": config.model.output_bins,
        "use_aviation_categories": config.model.use_aviation_categories,
        "use_global_features": config.model.use_global_features,
        "hl_gauss": {
            "transform": "z=log2(-Q), Q=-2**z",
            "z_min": config.model.z_min,
            "z_max": config.model.z_max,
            "sigma": config.model.hl_gauss_sigma,
        },
        "excluded_features": {
            "solving_time": "wall-clock dependent and not reproducible for identical SCIP states"
        },
    }


def _policy_seed(base: int, solver_seed: int, method: str, instance_index: int) -> int:
    offsets = {"random": 11, "untrained": 23, "rl": 37}
    return int(base * 1_000_003 + solver_seed * 101 + instance_index * 10_007 + offsets[method])


def evaluate_gcnn_episode(
    instance: str,
    seed: int,
    environment_config,
    method: str,
    learner: Optional[GraphDoubleDQNLearner],
    policy_seed: int,
) -> GraphEpisodeMetrics:
    rng = np.random.default_rng(policy_seed)
    env = BBMDPBranchingEnv(replace(environment_config, seed=int(seed)))
    state = env.reset(instance)
    graph = None if state.observation is None else training_graph_state(state.observation, state.action_set)
    reward = 0.0
    transitions = 0
    ranks = []
    inference = 0.0
    while not (state.terminated or state.truncated):
        if graph is None:
            raise RuntimeError("live GCNN environment state has no graph")
        if method == "random":
            position = int(rng.integers(graph.candidate_count))
            action = int(graph.actions[position])
        else:
            if learner is None:
                raise ValueError(f"method {method} requires a learner")
            start = time.monotonic()
            action, position, rank, _ = learner.select_action(graph, 0.0, rng)
            if learner.device.type == "cuda":
                torch.cuda.synchronize(learner.device)
            inference += time.monotonic() - start
            ranks.append(rank)
        transition = env.step(action)
        reward += transition.reward
        transitions += 1
        state = env.current_state
        graph = (
            None
            if transition.next_observation is None or transition.next_action_set.size == 0
            else training_graph_state(transition.next_observation, transition.next_action_set)
        )
    result = GraphEpisodeMetrics(
        method=method,
        instance=str(instance),
        seed=int(seed),
        policy_seed=int(policy_seed),
        status=str(state.info.get("status", "unknown")),
        nodes=int(state.info.get("node_count", 0)),
        solving_time=float(state.info.get("solving_time", 0.0)),
        reward=float(reward),
        transitions=transitions,
        mean_rank=float(np.mean(ranks)) if ranks else 0.0,
        inference_time=inference,
    )
    env.close()
    return result


def _save_parity_state(path: Path, state: GraphState) -> None:
    np.savez_compressed(
        path,
        row_features=state.row_features,
        variable_features=state.variable_features,
        edge_indices=state.edge_indices,
        edge_features=state.edge_features,
        global_features=state.global_features,
        variable_categories=state.variable_categories,
        row_categories=state.row_categories,
        candidate_indices=state.actions,
        candidate_names=np.asarray(state.candidate_names),
    )


def train_gcnn(config_path: Path | str) -> dict:
    config_path = Path(config_path)
    config = GCNNTrainingConfig.from_yaml(config_path)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output / "config.yaml")
    (output / "feature_schema.json").write_text(
        json.dumps(_schema(config), indent=2) + "\n", encoding="utf-8"
    )

    _set_deterministic(config.seed)
    device = _device(config.device)
    rng = np.random.default_rng(config.seed)
    learner = _learner(config, device)
    replay = DualPoolGraphReplay(
        config.seed,
        alpha=config.optimization.per_alpha,
        beta_start=config.optimization.per_beta_start,
        beta_steps=config.optimization.per_beta_steps,
        epsilon=config.optimization.per_epsilon,
    )
    shared_norm = Path(config.normalization_path) if config.normalization_path else Path()
    if shared_norm.is_file():
        raw = json.loads(shared_norm.read_text(encoding="utf-8"))
        statistics = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in raw.items()
            if key.endswith("_mean") or key.endswith("_std")
        }
        normalizer = RunningGraphNormalizer.from_statistics(statistics)
        learner.online.set_normalization(normalizer.statistics())
        learner.target.set_normalization(normalizer.statistics())
        normalizer_frozen = True
    else:
        normalizer = RunningGraphNormalizer()
        normalizer_frozen = False
    observed_states = 0
    history = HistoryWriter(output / "training_history.csv")
    latest: Optional[GraphUpdateMetrics] = None
    episode = 0
    best_validation = float("inf")
    stale_validations = 0
    next_validation = config.evaluation.interval_steps
    start_wall = time.monotonic()

    try:
        while learner.gradient_step < config.optimization.total_gradient_steps:
            instance = config.train_instances[episode % len(config.train_instances)]
            env = BBMDPBranchingEnv(replace(config.environment, seed=config.seed + episode))
            state = env.reset(instance)
            graph = None if state.observation is None else training_graph_state(state.observation, state.action_set)
            accumulator = NStepAccumulator(
                config.optimization.n_step, config.optimization.gamma
            )
            episode_reward = 0.0
            ranks = []
            while not (state.terminated or state.truncated):
                if graph is None:
                    raise RuntimeError("live training state has no graph")
                if not (output / "parity_observation.npz").exists():
                    _save_parity_state(output / "parity_observation.npz", graph)
                if not normalizer_frozen:
                    normalizer.update(graph)
                    observed_states += 1
                    if observed_states >= config.normalization_warmup_states:
                        statistics = normalizer.statistics()
                        learner.online.set_normalization(statistics)
                        learner.target.set_normalization(statistics)
                        normalizer.freeze()
                        normalizer_frozen = True

                epsilon = config.exploration.epsilon(learner.gradient_step)
                action, position, rank, _ = learner.select_action(
                    graph,
                    epsilon,
                    rng,
                    config.exploration.boltzmann_temperature,
                )
                ranks.append(rank)
                transition = env.step(action)
                next_graph = (
                    None
                    if transition.next_observation is None
                    or transition.next_action_set.size == 0
                    else training_graph_state(
                        transition.next_observation, transition.next_action_set
                    )
                )
                for experience in accumulator.append(
                    OneStepExperience(
                        graph,
                        position,
                        transition.reward,
                        next_graph,
                        transition.bootstrap_mask,
                    )
                ):
                    replay.add(experience, _pool_for_instance(instance))
                episode_reward += transition.reward
                graph = next_graph
                state = env.current_state

                if normalizer_frozen and _can_sample_logical_batch(replay):
                    for _ in range(config.optimization.updates_per_env_step):
                        if learner.gradient_step >= config.optimization.total_gradient_steps:
                            break
                        sample = replay.sample_logical_batch(learner.gradient_step)
                        forward_start = time.monotonic()
                        latest = learner.update(sample)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        forward_time = time.monotonic() - forward_start
                        replay.update_priorities(sample.handles, latest.priorities)
                        if learner.gradient_step % config.log_interval_steps == 0:
                            cpu_mb, gpu_mb = _memory(device)
                            history.write(
                                event="train_update",
                                episode=episode,
                                gradient_step=learner.gradient_step,
                                loss=latest.loss,
                                td_error=latest.td_error,
                                epsilon=epsilon,
                                replay_size=len(replay),
                                q_value_mean=latest.q_mean,
                                q_value_std=latest.q_std,
                                selected_candidate_rank=rank,
                                forward_time=forward_time,
                                cpu_memory_mb=cpu_mb,
                                gpu_memory_mb=gpu_mb,
                                instance=instance,
                            )
            final_info = state.info
            env.close()
            cpu_mb, gpu_mb = _memory(device)
            history.write(
                event="episode",
                episode=episode,
                gradient_step=learner.gradient_step,
                loss="" if latest is None else latest.loss,
                td_error="" if latest is None else latest.td_error,
                epsilon=config.exploration.epsilon(learner.gradient_step),
                reward=episode_reward,
                episode_nodes=final_info.get("node_count", 0),
                episode_solving_time=final_info.get("solving_time", 0.0),
                replay_size=len(replay),
                selected_candidate_rank=float(np.mean(ranks)) if ranks else 0.0,
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
                            evaluate_gcnn_episode(
                                validation_instance,
                                seed,
                                config.environment,
                                "rl",
                                learner,
                                _policy_seed(config.seed, seed, "rl", instance_index),
                            )
                        )
                validation_nodes = float(np.mean([item.nodes for item in validation]))
                validation_time = float(np.mean([item.solving_time for item in validation]))
                history.write(
                    event="validation",
                    episode=episode,
                    gradient_step=learner.gradient_step,
                    validation_nodes=validation_nodes,
                    validation_time=validation_time,
                    replay_size=len(replay),
                    status=";".join(item.status for item in validation),
                    instance=";".join(config.validation_instances),
                )
                if validation_nodes < best_validation:
                    best_validation = validation_nodes
                    stale_validations = 0
                    _save_checkpoint(output / "best_model.pt", learner, config)
                    export_gcnn_torchscript(
                        learner.online, output / "best_model_scripted.pt"
                    )
                else:
                    stale_validations += 1
                next_validation += config.evaluation.interval_steps
                if stale_validations >= config.evaluation.early_stopping_patience:
                    break
    finally:
        history.close()

    if not normalizer_frozen:
        statistics = normalizer.statistics()
        learner.online.set_normalization(statistics)
        learner.target.set_normalization(statistics)
    (output / "normalization.json").write_text(
        json.dumps(normalizer.to_json(), indent=2) + "\n", encoding="utf-8"
    )
    _save_checkpoint(output / "last_model.pt", learner, config)
    export_gcnn_torchscript(learner.online, output / "last_model_scripted.pt")
    if not (output / "best_model.pt").exists():
        _save_checkpoint(output / "best_model.pt", learner, config)
        export_gcnn_torchscript(learner.online, output / "best_model_scripted.pt")

    best_model = load_gcnn_model(output / "best_model.pt", config, device)
    best_learner = _learner(config, device)
    best_learner.online.load_state_dict(best_model.state_dict())
    best_learner.target.load_state_dict(best_model.state_dict())

    torch.manual_seed(config.seed + 1_000_000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed + 1_000_000)
    untrained = _learner(config, device)
    untrained.online.set_normalization(normalizer.statistics())
    untrained.target.set_normalization(normalizer.statistics())

    comparisons = []
    for instance_index, instance in enumerate(config.validation_instances):
        for seed in config.evaluation.compare_seeds:
            for method, evaluation_learner in (
                ("random", None),
                ("untrained", untrained),
                ("rl", best_learner),
            ):
                comparisons.append(
                    evaluate_gcnn_episode(
                        instance,
                        seed,
                        config.environment,
                        method,
                        evaluation_learner,
                        _policy_seed(config.seed, seed, method, instance_index),
                    )
                )
    with (output / "evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=GraphEpisodeMetrics.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(item.__dict__ for item in comparisons)

    summary = {
        "run_name": config.run_name,
        "gradient_steps": learner.gradient_step,
        "episodes": episode,
        "replay_size": len(replay),
        "best_validation_nodes": best_validation,
        "wall_time": time.monotonic() - start_wall,
        "device": str(device),
        "torch_version": torch.__version__,
        "loss_mode": config.model.loss_mode,
        "evaluation": [item.__dict__ for item in comparisons],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
