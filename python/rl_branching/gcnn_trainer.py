import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import torch
import yaml

from .config import (
    BBMDPConfig,
    RewardMode,
    SELECTION_RULE,
    finalized_gap,
    hybrid_identity_holds,
    metric_mean,
    missing_to_inf,
    par2_time,
    selection_key,
    solved_flag,
)
from .candidate_features import AVIATION_VARIABLE_CATEGORIES
from .graph_features import GRAPH_VARIABLE_FEATURE_NAMES
from .environment import BBMDPBranchingEnv
from .gcnn_config import GCNNTrainingConfig, LARGE_INSTANCE_PREFIX
from .gcnn_dqn import GraphDoubleDQNLearner, GraphUpdateMetrics
from .gcnn_model import BipartiteGCNNQNetwork, export_gcnn_torchscript
from .graph_features import (
    AVIATION_CONSTRAINT_CATEGORIES,
    GraphState,
    RunningGraphNormalizer,
    training_graph_state,
)
from .graph_replay import DualPoolGraphReplay, DualPoolQuotaUnfillable, LOGICAL_BATCH_SIZE
from .observation import EDGE_FEATURE_NAMES, EXTENDED_ROW_FEATURE_NAMES, GLOBAL_FEATURE_NAMES
from .replay import NStepAccumulator, OneStepExperience


HISTORY_FIELDS = (
    "event",
    "episode",
    "gradient_step",
    "optimizer_steps",
    "stage",
    "stage_boundary_step",
    "instance_time_limit",
    "instance_node_limit",
    "loss",
    "td_error",
    "gradient_norm",
    "epsilon",
    "reward",
    "node_reward",
    "lp_reward",
    "total_reward",
    "reward_identity_ok",
    "episode_nodes",
    "episode_solving_time",
    "validation_nodes",
    "validation_time",
    "mean_pdi",
    "mean_final_gap",
    "mean_par2",
    "mean_solved",
    "mean_lp",
    "pdi",
    "final_gap",
    "par2",
    "first_solution_time",
    "solved",
    "lp_iterations",
    "bootstrap_target_used",
    "trainer_truncated",
    "replay_size",
    "replay_medium_count",
    "replay_large_count",
    "replay_medium_bytes",
    "replay_large_bytes",
    "replay_evictions_by_count",
    "replay_evictions_by_bytes",
    "medium_samples",
    "large_samples",
    "per_weight_min",
    "per_weight_max",
    "q_value_mean",
    "q_value_std",
    "selected_candidate_rank",
    "forward_time",
    "cpu_memory_mb",
    "gpu_memory_mb",
    "status",
    "instance",
)

SHARED_NORMALIZATION_INSTANCES = (
    "data/instances/transfer/real_01.cip",
    "data/instances/transfer/real_02.cip",
    "data/instances/transfer/real_03.cip",
    "data/instances/transfer/real_05.cip",
    "data/instances/train/real_06.cip",
    "data/instances/train/real_07.cip",
)
EXCLUDED_NORMALIZATION_STEMS = frozenset({"real_04"})


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
    pdi: float
    final_gap: float
    par2: float
    first_solution_time: float
    solved: float
    lp_iterations: int
    node_reward: float
    lp_reward: float
    total_reward: float
    reward_identity_ok: bool
    trainer_truncated: bool


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


def _ecole_action(action_set: np.ndarray, graph: GraphState, position: int) -> int:
    if position < 0 or position >= graph.candidate_count:
        raise ValueError("candidate position is out of range")
    if int(graph.candidate_count) != int(np.asarray(action_set).size):
        raise ValueError("two-hop candidates must stay aligned with the Ecole action set")
    return int(np.asarray(action_set)[position])


def _can_sample_logical_batch(replay: DualPoolGraphReplay) -> bool:
    if len(replay.large) >= replay.large_sample_quota:
        return len(replay.medium) >= replay.medium_sample_quota
    return len(replay.medium) >= replay.medium_sample_quota + replay.large_sample_quota


def _batch_mix_counts(batch) -> tuple[int, int]:
    medium = sum(1 for handle in batch.handles if handle.pool == "medium")
    large = sum(1 for handle in batch.handles if handle.pool == "large")
    return medium, large


def _select_mix_instance(
    guard: "InstanceQuotaGuard",
    *,
    require_mix_16_0: bool,
    require_mix_12_4: bool,
    mix_16_0: int,
    mix_12_4: int,
) -> str:
    remaining = guard.available()
    medium = [name for name in remaining if _pool_for_instance(name) == "medium"]
    large = [name for name in remaining if _pool_for_instance(name) == "large"]
    if require_mix_16_0 and mix_16_0 == 0:
        if not medium:
            raise DualPoolQuotaUnfillable(
                "quota_unfillable: no medium instances left for the 16+0 mix"
            )
        return medium[0]
    if require_mix_12_4 and mix_12_4 == 0:
        if not large:
            raise DualPoolQuotaUnfillable(
                "quota_unfillable: no real_02 instances left for the 12+4 mix"
            )
        return large[0]
    return guard.next_instance()


def _ready_for_update(
    replay: DualPoolGraphReplay,
    *,
    min_replay_size: int,
    require_mix_16_0: bool,
    require_mix_12_4: bool,
    mix_16_0: int,
    mix_12_4: int,
    stage: str | None = None,
) -> bool:
    if stage == "A":
        return len(replay.medium) >= LOGICAL_BATCH_SIZE and len(replay) >= int(min_replay_size)
    if stage == "B":
        return (
            len(replay.large) >= replay.large_sample_quota
            and len(replay.medium) >= replay.medium_sample_quota
            and len(replay) >= int(min_replay_size)
        )
    if not _can_sample_logical_batch(replay):
        return False
    if len(replay) < int(min_replay_size):
        return False
    large_ready = len(replay.large) >= replay.large_sample_quota
    if require_mix_16_0 and mix_16_0 == 0:
        return not large_ready
    if require_mix_12_4 and mix_12_4 == 0:
        return large_ready
    return True


def _training_should_stop(
    *,
    gradient_step: int,
    total_gradient_steps: int,
    elapsed: float,
    wall_time_limit: float,
    require_mix_16_0: bool,
    require_mix_12_4: bool,
    mix_16_0: int,
    mix_12_4: int,
    stage_a_steps: int = 0,
    stage_b_steps: int = 0,
) -> str:
    if wall_time_limit > 0.0 and elapsed >= wall_time_limit:
        return "wall_time"
    if stage_a_steps > 0:
        if gradient_step >= int(stage_a_steps) + int(stage_b_steps):
            return "gradient_steps"
        return ""
    mixes_done = (not require_mix_16_0 or mix_16_0 > 0) and (
        not require_mix_12_4 or mix_12_4 > 0
    )
    if mixes_done and (require_mix_16_0 or require_mix_12_4):
        return "mix_gates"
    if mixes_done and gradient_step >= total_gradient_steps:
        return "gradient_steps"
    return ""


def _replay_log_fields(replay: DualPoolGraphReplay) -> dict:
    snapshot = replay.snapshot()
    return {
        "replay_size": len(replay),
        "replay_medium_count": snapshot.medium_count,
        "replay_large_count": snapshot.large_count,
        "replay_medium_bytes": snapshot.medium_bytes,
        "replay_large_bytes": snapshot.large_bytes,
        "replay_evictions_by_count": snapshot.evictions_by_count,
        "replay_evictions_by_bytes": snapshot.evictions_by_bytes,
    }


def _environment_for_instance(
    config: GCNNTrainingConfig,
    instance: str,
    seed: int,
) -> BBMDPConfig:
    time_limit, node_limit = config.instance_budget(instance)
    return replace(
        config.environment,
        seed=int(seed),
        time_limit=float(time_limit),
        node_limit=int(node_limit),
    )


def _apply_live_truncation(env: BBMDPBranchingEnv, transition):
    if not env.config.bootstrap_on_truncation:
        return transition
    return env.mark_live_budget_truncation(transition)


def _optional_metric(value) -> float:
    if value is None:
        return float("inf")
    try:
        return missing_to_inf(float(value))
    except (TypeError, ValueError):
        return float("inf")


def _episode_identity_ok(
    *,
    reward_mode: RewardMode,
    rewards: list[float],
    n0: int,
    n_t: int,
    lp0: int,
    lp_t: int,
) -> bool:
    if reward_mode != RewardMode.HYBRID_NODE_LP or not rewards:
        return True
    if n_t < n0 or lp_t < lp0:
        return hybrid_identity_holds(rewards, n0, n_t, lp0, lp_t)
    ok = hybrid_identity_holds(rewards, n0, n_t, lp0, lp_t)
    if not ok:
        raise RuntimeError(
            "hybrid reward identity failed: "
            f"sum_r={sum(rewards)} n0={n0} nT={n_t} lp0={lp0} lpT={lp_t}"
        )
    return True


def _selection_from_episodes(
    episodes: list[GraphEpisodeMetrics],
    *,
    gradient_step: int,
    seed: int,
) -> tuple:
    return selection_key(
        solved_rate=float(np.mean([item.solved for item in episodes])) if episodes else 0.0,
        mean_pdi=metric_mean(item.pdi for item in episodes),
        mean_final_gap=metric_mean(item.final_gap for item in episodes),
        mean_par2=metric_mean(item.par2 for item in episodes),
        mean_lp=metric_mean(float(item.lp_iterations) for item in episodes),
        mean_nodes=metric_mean(float(item.nodes) for item in episodes),
        gradient_step=int(gradient_step),
        seed=int(seed),
    )


def _collect_normalization_states(
    instance: str,
    *,
    states_needed: int,
    seed: int,
    time_limit: float,
    node_limit: int,
) -> list[GraphState]:
    env = BBMDPBranchingEnv(
        BBMDPConfig(seed=seed, time_limit=time_limit, node_limit=node_limit)
    )
    collected: list[GraphState] = []
    try:
        state = env.reset(instance)
        while (
            len(collected) < states_needed
            and not state.terminated
            and not state.truncated
            and state.observation is not None
            and state.action_set.size > 0
        ):
            collected.append(training_graph_state(state.observation, state.action_set))
            action = int(state.action_set[0])
            transition = env.step(action)
            state = env.current_state
            if transition.next_observation is None or transition.next_action_set.size == 0:
                break
    finally:
        env.close()
    return collected


def build_shared_normalization(
    output_path: Path | str,
    *,
    instances: tuple[str, ...] = SHARED_NORMALIZATION_INSTANCES,
    states_per_instance: int = 2,
    seed: int = 0,
    default_time_limit: float = 180.0,
    large_time_limit: float = 300.0,
    node_limit: int = 20,
) -> dict:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stems = [Path(path).stem for path in instances]
    if any(stem in EXCLUDED_NORMALIZATION_STEMS for stem in stems):
        raise ValueError("shared normalization must exclude real_04")
    missing = [stem for stem in ("real_01", "real_02", "real_03", "real_05", "real_06", "real_07") if stem not in stems]
    if missing:
        raise ValueError(f"shared normalization is missing required instances: {missing}")
    if states_per_instance < 2:
        raise ValueError("shared normalization needs at least 2 states per instance")

    normalizer = RunningGraphNormalizer()
    per_instance: dict[str, dict] = {}
    for index, instance in enumerate(instances):
        path = Path(instance)
        if not path.is_file():
            raise FileNotFoundError(instance)
        if path.stem in EXCLUDED_NORMALIZATION_STEMS:
            raise ValueError(f"refusing excluded instance {path.stem}")
        time_limit = large_time_limit if path.stem.startswith("real_02") else default_time_limit
        graphs = _collect_normalization_states(
            str(path),
            states_needed=states_per_instance,
            seed=seed + index,
            time_limit=time_limit,
            node_limit=node_limit,
        )
        if len(graphs) < states_per_instance:
            raise RuntimeError(
                f"{path.stem} produced {len(graphs)} branching states, need {states_per_instance}"
            )
        for graph in graphs:
            normalizer.update(graph)
        per_instance[path.stem] = {
            "path": str(path),
            "states": len(graphs),
            "time_limit": time_limit,
            "node_limit": node_limit,
            "variable_count": int(graphs[0].variable_features.shape[0]),
            "row_count": int(graphs[0].row_features.shape[0]),
        }
    normalizer.freeze()
    payload = {
        "schema": "shared_graph_normalization_v1",
        "frozen": True,
        "excluded_instances": sorted(EXCLUDED_NORMALIZATION_STEMS),
        "source_instances": per_instance,
        "states_per_instance": states_per_instance,
        **normalizer.to_json(),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    sha256 = _file_sha256(output)
    manifest = {
        "normalization_path": str(output),
        "sha256": sha256,
        "excluded_instances": sorted(EXCLUDED_NORMALIZATION_STEMS),
        "source_instances": per_instance,
        "states_per_instance": states_per_instance,
        "readonly": True,
    }
    manifest_path = output.with_name(output.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def run_deploy_retest(
    model_path: Path | str,
    *,
    instance: str,
    seed_overlay: str,
    solve_node_limit: int,
    output_json: Path,
    output_log: Path,
    scip_tree: Path | str = "build/scip_tree",
    time_limit: float = 60.0,
    overhead_fraction: float = 0.10,
    rl_device: str = "cpu",
) -> dict:
    runner = Path(scip_tree)
    if not runner.is_file():
        raise FileNotFoundError(f"scip_tree is missing: {runner}")
    overlay = Path(seed_overlay)
    if not overlay.is_file():
        raise FileNotFoundError(f"seed overlay is missing: {overlay}")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(runner),
            "--instance",
            instance,
            "--branching",
            "rl-gcnn",
            "--rl-model",
            str(model_path),
            "--rl-device",
            str(rl_device),
            "--seed-overlay",
            str(overlay),
            "--scip-profile",
            "configs/scip/project-production-v1.set",
            "--time-limit",
            str(time_limit),
            "--node-limit",
            "-1",
            "--solve-node-limit",
            str(solve_node_limit),
            "--threads",
            "1",
            "--output-json",
            str(output_json),
            "--rl-log",
            str(output_log),
        ],
        check=True,
    )
    smoke = json.loads(output_json.read_text(encoding="utf-8"))
    extract_time = 0.0
    inference_time = 0.0
    if output_log.is_file():
        with output_log.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                extract_time += float(row.get("graph_extract_time_seconds") or 0.0)
                inference_time += float(row.get("inference_time_seconds") or 0.0)
    solving_time = float(smoke.get("solving_time") or 0.0)
    overhead = extract_time + inference_time
    overhead_ratio = overhead / solving_time if solving_time > 0.0 else float("inf")
    branch_decisions = int(smoke.get("branch_decisions", 0) or 0)
    fallback_count = int(smoke.get("fallback_count", -1) if smoke.get("fallback_count") is not None else -1)
    illegal_actions = int(
        smoke.get("custom_illegal_actions", -1)
        if smoke.get("custom_illegal_actions") is not None
        else -1
    )
    gate = {
        "instance": instance,
        "model": str(model_path),
        "output_json": str(output_json),
        "rl_device": str(rl_device),
        "branch_decisions": branch_decisions,
        "fallback_count": fallback_count,
        "custom_illegal_actions": illegal_actions,
        "solving_time": solving_time,
        "graph_extract_time": extract_time,
        "inference_time": inference_time,
        "gcnn_overhead": overhead,
        "gcnn_overhead_ratio": overhead_ratio,
        "passed": bool(
            branch_decisions >= 1
            and fallback_count == 0
            and illegal_actions == 0
            and math.isfinite(overhead_ratio)
            and overhead_ratio <= overhead_fraction
        ),
    }
    return gate


ZERO_DECISION_SKIP_N = 3


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_shared_normalizer(path: Path | str) -> tuple[RunningGraphNormalizer, str]:
    shared_norm = Path(path)
    if not shared_norm.is_file():
        raise FileNotFoundError(
            f"normalization_path is set but does not exist: {shared_norm}"
        )
    sha256 = _file_sha256(shared_norm)
    raw = json.loads(shared_norm.read_text(encoding="utf-8"))
    statistics = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in raw.items()
        if key.endswith("_mean") or key.endswith("_std")
    }
    return RunningGraphNormalizer.from_statistics(statistics), sha256


class InstanceQuotaGuard:
    def __init__(
        self,
        instances: tuple[str, ...] | list[str],
        skip_after: int = ZERO_DECISION_SKIP_N,
    ) -> None:
        if not instances:
            raise ValueError("train instances must be non-empty")
        self.instances = tuple(instances)
        self.skip_after = int(skip_after)
        self.zero_streak = {name: 0 for name in self.instances}
        self.unfillable: set[str] = set()
        self._cursor = 0

    def available(self) -> list[str]:
        return [name for name in self.instances if name not in self.unfillable]

    def next_instance(self) -> str:
        remaining = self.available()
        if not remaining:
            raise DualPoolQuotaUnfillable(
                "quota_unfillable: all train instances skipped after consecutive "
                "zero-decision episodes"
            )
        instance = remaining[self._cursor % len(remaining)]
        self._cursor += 1
        return instance

    def record_episode(self, instance: str, n_decisions: int) -> bool:
        if n_decisions > 0:
            self.zero_streak[instance] = 0
            return False
        self.zero_streak[instance] = self.zero_streak.get(instance, 0) + 1
        if self.zero_streak[instance] >= self.skip_after:
            self.unfillable.add(instance)
            return True
        return False


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
    n0 = int(state.info.get("node_count", 0))
    lp0 = int(state.info.get("lp_iterations", 0))
    rewards: list[float] = []
    node_reward = 0.0
    lp_reward = 0.0
    transitions = 0
    ranks = []
    inference = 0.0
    trainer_truncated = False
    while not (state.terminated or state.truncated):
        if graph is None:
            raise RuntimeError("live GCNN environment state has no graph")
        if method == "random":
            position = int(rng.integers(graph.candidate_count))
            action = _ecole_action(state.action_set, graph, position)
        else:
            if learner is None:
                raise ValueError(f"method {method} requires a learner")
            start = time.monotonic()
            _, position, rank, _ = learner.select_action(graph, 0.0, rng)
            if learner.device.type == "cuda":
                torch.cuda.synchronize(learner.device)
            inference += time.monotonic() - start
            ranks.append(rank)
            action = _ecole_action(state.action_set, graph, position)
        transition = _apply_live_truncation(env, env.step(action))
        rewards.append(float(transition.reward))
        node_reward += float(transition.info.get("node_reward", 0.0))
        lp_reward += float(transition.info.get("lp_reward", 0.0))
        trainer_truncated = trainer_truncated or bool(transition.info.get("trainer_truncated"))
        transitions += 1
        state = env.current_state
        graph = (
            None
            if transition.next_observation is None or transition.next_action_set.size == 0
            else training_graph_state(transition.next_observation, transition.next_action_set)
        )
    status = str(state.info.get("status", "unknown"))
    nodes = int(state.info.get("node_count", 0))
    lp_iterations = int(state.info.get("lp_iterations", 0))
    time_limit = float(environment_config.time_limit)
    identity_ok = _episode_identity_ok(
        reward_mode=environment_config.reward_mode,
        rewards=rewards,
        n0=n0,
        n_t=nodes,
        lp0=lp0,
        lp_t=lp_iterations,
    )
    result = GraphEpisodeMetrics(
        method=method,
        instance=str(instance),
        seed=int(seed),
        policy_seed=int(policy_seed),
        status=status,
        nodes=nodes,
        solving_time=float(state.info.get("solving_time", 0.0)),
        reward=float(sum(rewards)),
        transitions=transitions,
        mean_rank=float(np.mean(ranks)) if ranks else 0.0,
        inference_time=inference,
        pdi=_optional_metric(state.info.get("primal_dual_integral")),
        final_gap=finalized_gap(
            status,
            float(state.info.get("primal_bound", float("inf"))),
            float(state.info.get("dual_bound", float("-inf"))),
            float(state.info.get("gap", float("inf"))),
        ),
        par2=par2_time(status, float(state.info.get("solving_time", 0.0)), time_limit),
        first_solution_time=_optional_metric(state.info.get("first_solution_time")),
        solved=solved_flag(status),
        lp_iterations=lp_iterations,
        node_reward=float(node_reward),
        lp_reward=float(lp_reward),
        total_reward=float(sum(rewards)),
        reward_identity_ok=bool(identity_ok),
        trainer_truncated=bool(trainer_truncated),
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
        medium_count_limit=config.optimization.medium_count_limit,
        large_count_limit=config.optimization.large_count_limit,
        alpha=config.optimization.per_alpha,
        beta_start=config.optimization.per_beta_start,
        beta_steps=config.optimization.per_beta_steps,
        epsilon=config.optimization.per_epsilon,
    )
    normalization_sha256 = ""
    if config.normalization_path:
        normalizer, normalization_sha256 = load_shared_normalizer(config.normalization_path)
        learner.online.set_normalization(normalizer.statistics())
        learner.target.set_normalization(normalizer.statistics())
        normalizer_frozen = True
    else:
        normalizer = RunningGraphNormalizer()
        normalizer_frozen = False
    observed_states = 0
    history = HistoryWriter(output / "training_history.csv")
    history.write(
        event="normalization",
        status=normalization_sha256 or "local_warmup",
        instance=config.normalization_path,
    )
    latest: Optional[GraphUpdateMetrics] = None
    episode = 0
    stage_guards = {
        None: InstanceQuotaGuard(config.train_instances),
        "A": InstanceQuotaGuard(config.instances_for_stage("A"))
        if config.curriculum_enabled()
        else None,
        "B": InstanceQuotaGuard(config.instances_for_stage("B"))
        if config.curriculum_enabled()
        else None,
    }
    best_key = None
    best_selection = {}
    stale_validations = 0
    next_validation = config.evaluation.interval_steps
    start_wall = time.monotonic()
    mix_16_0 = 0
    mix_12_4 = 0
    stop_reason = ""
    finite_metrics = True
    last_medium_samples = 0
    last_large_samples = 0
    logged_stage_boundary = False
    bootstrap_target_updates = 0

    try:
        while not stop_reason:
            elapsed = time.monotonic() - start_wall
            stage = config.stage_for_step(learner.gradient_step)
            if (
                config.curriculum_enabled()
                and stage == "B"
                and not logged_stage_boundary
            ):
                history.write(
                    event="stage_boundary",
                    episode=episode,
                    gradient_step=learner.gradient_step,
                    optimizer_steps=learner.gradient_step,
                    stage="B",
                    stage_boundary_step=learner.gradient_step,
                    instance_time_limit=config.large_time_limit,
                    instance_node_limit=config.large_node_limit,
                )
                logged_stage_boundary = True
            stop_reason = _training_should_stop(
                gradient_step=learner.gradient_step,
                total_gradient_steps=config.optimization.total_gradient_steps,
                elapsed=elapsed,
                wall_time_limit=config.wall_time_limit,
                require_mix_16_0=config.require_mix_16_0,
                require_mix_12_4=config.require_mix_12_4,
                mix_16_0=mix_16_0,
                mix_12_4=mix_12_4,
                stage_a_steps=config.stage_a_steps,
                stage_b_steps=config.stage_b_steps,
            )
            if stop_reason:
                break
            instance_guard = stage_guards[stage]
            if stage is None:
                hunt_16 = config.require_mix_16_0
                hunt_12 = config.require_mix_12_4
                mix_16_for_select = mix_16_0
                mix_12_for_select = mix_12_4
            elif stage == "A":
                hunt_16 = False
                hunt_12 = False
                mix_16_for_select = 1
                mix_12_for_select = 1
            else:
                hunt_16 = False
                hunt_12 = mix_12_4 == 0
                mix_16_for_select = 1
                mix_12_for_select = mix_12_4
            instance = _select_mix_instance(
                instance_guard,
                require_mix_16_0=hunt_16,
                require_mix_12_4=hunt_12,
                mix_16_0=mix_16_for_select,
                mix_12_4=mix_12_for_select,
            )
            if stage == "A" and Path(instance).stem.startswith(LARGE_INSTANCE_PREFIX):
                raise RuntimeError(f"real_02 entered Stage A: {instance}")
            instance_time_limit, instance_node_limit = config.instance_budget(instance)
            env = BBMDPBranchingEnv(
                _environment_for_instance(config, instance, config.seed + episode)
            )
            state = env.reset(instance)
            graph = None if state.observation is None else training_graph_state(state.observation, state.action_set)
            accumulator = NStepAccumulator(
                config.optimization.n_step, config.optimization.gamma
            )
            n0 = int(state.info.get("node_count", 0))
            lp0 = int(state.info.get("lp_iterations", 0))
            step_rewards: list[float] = []
            node_reward_sum = 0.0
            lp_reward_sum = 0.0
            trainer_truncated = False
            ranks = []
            while not (state.terminated or state.truncated):
                if config.wall_time_limit > 0.0 and (
                    time.monotonic() - start_wall >= config.wall_time_limit
                ):
                    stop_reason = "wall_time"
                    break
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
                _, position, rank, _ = learner.select_action(
                    graph,
                    epsilon,
                    rng,
                    config.exploration.boltzmann_temperature,
                )
                ranks.append(rank)
                transition = _apply_live_truncation(
                    env, env.step(_ecole_action(state.action_set, graph, position))
                )
                trainer_truncated = trainer_truncated or bool(
                    transition.info.get("trainer_truncated")
                )
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
                step_rewards.append(float(transition.reward))
                node_reward_sum += float(transition.info.get("node_reward", 0.0))
                lp_reward_sum += float(transition.info.get("lp_reward", 0.0))
                graph = next_graph
                state = env.current_state

                hunting_mix = (not config.curriculum_enabled()) and (
                    (config.require_mix_16_0 and mix_16_0 == 0)
                    or (config.require_mix_12_4 and mix_12_4 == 0)
                )
                stage_cap = (
                    config.stage_a_steps
                    if stage == "A"
                    else config.optimization.total_gradient_steps
                )
                if normalizer_frozen and _ready_for_update(
                    replay,
                    min_replay_size=config.optimization.min_replay_size,
                    require_mix_16_0=config.require_mix_16_0,
                    require_mix_12_4=config.require_mix_12_4,
                    mix_16_0=mix_16_0,
                    mix_12_4=mix_12_4,
                    stage=stage,
                ):
                    update_count = (
                        1 if hunting_mix else config.optimization.updates_per_env_step
                    )
                    for _ in range(update_count):
                        if not hunting_mix and learner.gradient_step >= stage_cap:
                            if stage == "A":
                                break
                            stop_reason = "gradient_steps"
                            break
                        sample = replay.sample_logical_batch(
                            learner.gradient_step,
                            allow_large=stage != "A",
                        )
                        last_medium_samples, last_large_samples = _batch_mix_counts(sample)
                        if stage == "A" and last_large_samples != 0:
                            raise RuntimeError("Stage A sampled a large-pool transition")
                        forward_start = time.monotonic()
                        latest = learner.update(sample)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        forward_time = time.monotonic() - forward_start
                        replay.update_priorities(sample.handles, latest.priorities)
                        if latest.bootstrap_target_used:
                            bootstrap_target_updates += 1
                        metric_values = (
                            latest.loss,
                            latest.td_error,
                            latest.gradient_norm,
                            float(np.min(sample.weights)),
                            float(np.max(sample.weights)),
                        )
                        if not all(math.isfinite(float(value)) for value in metric_values):
                            finite_metrics = False
                            raise FloatingPointError(
                                "GCNN train update produced a non-finite metric"
                            )
                        if last_medium_samples == LOGICAL_BATCH_SIZE and last_large_samples == 0:
                            mix_16_0 += 1
                            mix_event = "mix_16_0"
                        elif last_medium_samples == 12 and last_large_samples == 4:
                            mix_12_4 += 1
                            mix_event = "mix_12_4"
                        else:
                            mix_event = ""
                        if (
                            mix_event
                            or learner.gradient_step % config.log_interval_steps == 0
                        ):
                            cpu_mb, gpu_mb = _memory(device)
                            history.write(
                                event=mix_event or "train_update",
                                episode=episode,
                                gradient_step=learner.gradient_step,
                                optimizer_steps=learner.gradient_step,
                                stage=stage or "",
                                instance_time_limit=instance_time_limit,
                                instance_node_limit=instance_node_limit,
                                loss=latest.loss,
                                td_error=latest.td_error,
                                gradient_norm=latest.gradient_norm,
                                epsilon=epsilon,
                                bootstrap_target_used=latest.bootstrap_target_used,
                                medium_samples=last_medium_samples,
                                large_samples=last_large_samples,
                                per_weight_min=float(np.min(sample.weights)),
                                per_weight_max=float(np.max(sample.weights)),
                                q_value_mean=latest.q_mean,
                                q_value_std=latest.q_std,
                                selected_candidate_rank=rank,
                                forward_time=forward_time,
                                cpu_memory_mb=cpu_mb,
                                gpu_memory_mb=gpu_mb,
                                instance=instance,
                                **_replay_log_fields(replay),
                            )
                        stop_reason = _training_should_stop(
                            gradient_step=learner.gradient_step,
                            total_gradient_steps=config.optimization.total_gradient_steps,
                            elapsed=time.monotonic() - start_wall,
                            wall_time_limit=config.wall_time_limit,
                            require_mix_16_0=config.require_mix_16_0,
                            require_mix_12_4=config.require_mix_12_4,
                            mix_16_0=mix_16_0,
                            mix_12_4=mix_12_4,
                            stage_a_steps=config.stage_a_steps,
                            stage_b_steps=config.stage_b_steps,
                        )
                        if stop_reason:
                            break
                        if stage == "A" and learner.gradient_step >= config.stage_a_steps:
                            break
                if (
                    not stop_reason
                    and not config.curriculum_enabled()
                    and config.require_mix_12_4
                    and mix_16_0 > 0
                    and mix_12_4 == 0
                    and _pool_for_instance(instance) == "medium"
                ):
                    break
                if stop_reason:
                    break
                if stage == "A" and learner.gradient_step >= config.stage_a_steps:
                    break
            for experience in accumulator.flush():
                replay.add(experience, _pool_for_instance(instance))
            final_info = state.info
            identity_ok = _episode_identity_ok(
                reward_mode=config.environment.reward_mode,
                rewards=step_rewards,
                n0=n0,
                n_t=int(final_info.get("node_count", 0)),
                lp0=lp0,
                lp_t=int(final_info.get("lp_iterations", 0)),
            )
            env.close()
            cpu_mb, gpu_mb = _memory(device)
            history.write(
                event="episode",
                episode=episode,
                gradient_step=learner.gradient_step,
                optimizer_steps=learner.gradient_step,
                stage=stage or "",
                instance_time_limit=instance_time_limit,
                instance_node_limit=instance_node_limit,
                loss="" if latest is None else latest.loss,
                td_error="" if latest is None else latest.td_error,
                gradient_norm="" if latest is None else latest.gradient_norm,
                epsilon=config.exploration.epsilon(learner.gradient_step),
                reward=float(sum(step_rewards)),
                node_reward=node_reward_sum,
                lp_reward=lp_reward_sum,
                total_reward=float(sum(step_rewards)),
                reward_identity_ok=identity_ok,
                episode_nodes=final_info.get("node_count", 0),
                episode_solving_time=final_info.get("solving_time", 0.0),
                pdi=_optional_metric(final_info.get("primal_dual_integral")),
                final_gap=finalized_gap(
                    str(final_info.get("status", "unknown")),
                    float(final_info.get("primal_bound", float("inf"))),
                    float(final_info.get("dual_bound", float("-inf"))),
                    float(final_info.get("gap", float("inf"))),
                ),
                par2=par2_time(
                    str(final_info.get("status", "unknown")),
                    float(final_info.get("solving_time", 0.0)),
                    float(instance_time_limit),
                ),
                first_solution_time=_optional_metric(final_info.get("first_solution_time")),
                solved=solved_flag(str(final_info.get("status", "unknown"))),
                lp_iterations=final_info.get("lp_iterations", 0),
                trainer_truncated=trainer_truncated,
                bootstrap_target_used="" if latest is None else latest.bootstrap_target_used,
                medium_samples=last_medium_samples,
                large_samples=last_large_samples,
                selected_candidate_rank=float(np.mean(ranks)) if ranks else 0.0,
                cpu_memory_mb=cpu_mb,
                gpu_memory_mb=gpu_mb,
                status=final_info.get("status", "unknown"),
                instance=instance,
                **_replay_log_fields(replay),
            )
            if instance_guard.record_episode(instance, len(ranks)):
                history.write(
                    event="quota_unfillable",
                    episode=episode,
                    gradient_step=learner.gradient_step,
                    optimizer_steps=learner.gradient_step,
                    stage=stage or "",
                    status="skipped_zero_decision",
                    instance=instance,
                )
            episode += 1
            if stop_reason:
                break

            reached_validation = (
                not config.skip_mid_validation
                and (
                    learner.gradient_step >= next_validation
                    or learner.gradient_step >= config.optimization.total_gradient_steps
                )
            )
            if reached_validation:
                validation = []
                for instance_index, validation_instance in enumerate(config.validation_instances):
                    validation_env = _environment_for_instance(
                        config, validation_instance, config.evaluation.seeds[0]
                    )
                    for seed in config.evaluation.seeds:
                        validation.append(
                            evaluate_gcnn_episode(
                                validation_instance,
                                seed,
                                validation_env,
                                "rl",
                                learner,
                                _policy_seed(config.seed, seed, "rl", instance_index),
                            )
                        )
                current_key = _selection_from_episodes(
                    validation,
                    gradient_step=learner.gradient_step,
                    seed=config.seed,
                )
                validation_nodes = float(np.mean([item.nodes for item in validation]))
                validation_time = float(np.mean([item.solving_time for item in validation]))
                history.write(
                    event="validation",
                    episode=episode,
                    gradient_step=learner.gradient_step,
                    optimizer_steps=learner.gradient_step,
                    stage=stage or "",
                    validation_nodes=validation_nodes,
                    validation_time=validation_time,
                    mean_solved=current_key[0] * -1.0,
                    mean_pdi=current_key[1],
                    mean_final_gap=current_key[2],
                    mean_par2=current_key[3],
                    mean_lp=current_key[4],
                    status=";".join(item.status for item in validation),
                    instance=";".join(config.validation_instances),
                    **_replay_log_fields(replay),
                )
                if best_key is None or current_key < best_key:
                    best_key = current_key
                    best_selection = {
                        "rule": SELECTION_RULE,
                        "gradient_step": learner.gradient_step,
                        "seed": config.seed,
                        "solved_rate": -current_key[0],
                        "mean_pdi": current_key[1],
                        "mean_final_gap": current_key[2],
                        "mean_par2": current_key[3],
                        "mean_lp": current_key[4],
                        "mean_nodes": current_key[5],
                    }
                    stale_validations = 0
                    _save_checkpoint(output / "best_model.pt", learner, config)
                    export_gcnn_torchscript(
                        learner.online, output / "best_model_scripted.pt"
                    )
                else:
                    stale_validations += 1
                next_validation += config.evaluation.interval_steps
                if stale_validations >= config.evaluation.early_stopping_patience:
                    stop_reason = "early_stop"
                    break
    finally:
        history.close()

    if not normalizer_frozen:
        statistics = normalizer.statistics()
        learner.online.set_normalization(statistics)
        learner.target.set_normalization(statistics)
    if config.normalization_path:
        shutil.copyfile(config.normalization_path, output / "normalization.json")
        copied_sha = _file_sha256(output / "normalization.json")
        if copied_sha != normalization_sha256:
            raise RuntimeError("copied shared normalization SHA256 does not match source")
    else:
        (output / "normalization.json").write_text(
            json.dumps(normalizer.to_json(), indent=2) + "\n", encoding="utf-8"
        )
    _save_checkpoint(output / "last_model.pt", learner, config)
    export_gcnn_torchscript(learner.online, output / "last_model_scripted.pt")
    if not (output / "best_model.pt").exists():
        _save_checkpoint(output / "best_model.pt", learner, config)
        export_gcnn_torchscript(learner.online, output / "best_model_scripted.pt")

    comparisons = []
    if not config.skip_final_comparison:
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

        for instance_index, instance in enumerate(config.validation_instances):
            eval_env = _environment_for_instance(
                config, instance, config.evaluation.compare_seeds[0]
            )
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
                            eval_env,
                            method,
                            evaluation_learner,
                            _policy_seed(config.seed, seed, method, instance_index),
                        )
                    )
    with (output / "evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=GraphEpisodeMetrics.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(item.__dict__ for item in comparisons)

    deploy_retest = {}
    if config.deploy_retest_instance:
        deploy_retest = run_deploy_retest(
            output / "last_model_scripted.pt",
            instance=config.deploy_retest_instance,
            seed_overlay=config.deploy_retest_seed_overlay,
            solve_node_limit=config.deploy_retest_solve_node_limit,
            output_json=output / "deploy_retest.json",
            output_log=output / "deploy_retest.branches.csv",
            rl_device="cpu",
            overhead_fraction=0.10,
        )

    snapshot = replay.snapshot()
    r1_gate = {
        "stop_reason": stop_reason,
        "mix_16_0": mix_16_0,
        "mix_12_4": mix_12_4,
        "optimizer_steps": learner.gradient_step,
        "finite_metrics": finite_metrics,
        "torchscript_exported": (output / "last_model_scripted.pt").is_file(),
        "replay_medium_count": snapshot.medium_count,
        "replay_large_count": snapshot.large_count,
        "replay_evictions_by_count": snapshot.evictions_by_count,
        "replay_evictions_by_bytes": snapshot.evictions_by_bytes,
        "bootstrap_target_updates": bootstrap_target_updates,
        "deploy_retest": deploy_retest,
    }
    mix_ok = (not config.require_mix_16_0 or mix_16_0 > 0) and (
        not config.require_mix_12_4 or mix_12_4 > 0
    )
    deploy_ok = (not config.deploy_retest_instance) or bool(deploy_retest.get("passed"))
    r1_gate["passed"] = bool(
        mix_ok
        and finite_metrics
        and learner.gradient_step >= 1
        and r1_gate["torchscript_exported"]
        and deploy_ok
    )

    summary = {
        "run_name": config.run_name,
        "gradient_steps": learner.gradient_step,
        "optimizer_steps": learner.gradient_step,
        "episodes": episode,
        "replay_size": len(replay),
        "replay_medium_count": snapshot.medium_count,
        "replay_large_count": snapshot.large_count,
        "replay_evictions_by_count": snapshot.evictions_by_count,
        "replay_evictions_by_bytes": snapshot.evictions_by_bytes,
        "mix_16_0": mix_16_0,
        "mix_12_4": mix_12_4,
        "stop_reason": stop_reason,
        "stage_a_steps": config.stage_a_steps,
        "stage_b_steps": config.stage_b_steps,
        "selection_rule": SELECTION_RULE,
        "best_selection": best_selection,
        "bootstrap_target_updates": bootstrap_target_updates,
        "wall_time": time.monotonic() - start_wall,
        "device": str(device),
        "torch_version": torch.__version__,
        "loss_mode": config.model.loss_mode,
        "normalization_path": config.normalization_path,
        "normalization_sha256": normalization_sha256,
        "normalization_source": "shared" if normalization_sha256 else "local_warmup",
        "evaluation": [item.__dict__ for item in comparisons],
        "deploy_retest": deploy_retest,
        "r1_gate": r1_gate,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output / "r1_gate.json").write_text(
        json.dumps(r1_gate, indent=2) + "\n", encoding="utf-8"
    )
    (output / "selection.json").write_text(
        json.dumps(
            {
                "rule": SELECTION_RULE,
                "missing_value": "inf",
                "tie_break": "earlier_gradient_step then lower_seed",
                "best": best_selection,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if (config.require_mix_16_0 or config.require_mix_12_4) and not r1_gate["passed"]:
        raise RuntimeError(f"R1 gate failed: {json.dumps(r1_gate)}")
    return summary
