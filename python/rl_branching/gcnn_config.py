from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import BBMDPConfig, RewardMode
from .graph_replay import LARGE_COUNT_LIMIT, LARGE_SAMPLE_QUOTA, LOGICAL_BATCH_SIZE, MEDIUM_COUNT_LIMIT


@dataclass(frozen=True)
class GCNNModelConfig:
    embedding_dim: int = 64
    hidden_dim: int = 128
    loss_mode: str = "scalar"
    distributional_bins: int = 18
    z_min: float = -1.0
    z_max: float = 12.0
    hl_gauss_sigma: float = 0.75
    use_aviation_categories: bool = True
    use_global_features: bool = True

    @property
    def output_bins(self) -> int:
        return 1 if self.loss_mode == "scalar" else self.distributional_bins

    def __post_init__(self) -> None:
        if self.loss_mode not in {"scalar", "hl_gauss"}:
            raise ValueError("loss_mode must be scalar or hl_gauss")
        if self.embedding_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("GCNN dimensions must be positive")
        if self.loss_mode == "hl_gauss" and self.distributional_bins <= 1:
            raise ValueError("HL-Gauss requires more than one bin")
        if self.z_min >= self.z_max or self.hl_gauss_sigma <= 0.0:
            raise ValueError("invalid HL-Gauss support")


@dataclass(frozen=True)
class GCNNOptimizationConfig:
    total_gradient_steps: int = 100
    batch_size: int = LOGICAL_BATCH_SIZE
    medium_count_limit: int = MEDIUM_COUNT_LIMIT
    large_count_limit: int = LARGE_COUNT_LIMIT
    replay_capacity: int = MEDIUM_COUNT_LIMIT + LARGE_COUNT_LIMIT
    min_replay_size: int = LOGICAL_BATCH_SIZE
    learning_rate: float = 0.0003
    updates_per_env_step: int = 4
    n_step: int = 3
    gamma: float = 1.0
    gradient_clip: float = 10.0
    target_tau: float = 0.01
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_steps: int = 1000
    per_epsilon: float = 1.0e-5

    def __post_init__(self) -> None:
        if self.batch_size != LOGICAL_BATCH_SIZE:
            raise ValueError(
                "GCNN DualPool always samples 16 graphs; set optimization.batch_size=16"
            )
        if self.medium_count_limit < LOGICAL_BATCH_SIZE:
            raise ValueError("medium_count_limit must be at least the DualPool logical batch of 16")
        if self.large_count_limit < LARGE_SAMPLE_QUOTA:
            raise ValueError("large_count_limit must be at least the DualPool large quota of 4")
        expected_capacity = self.medium_count_limit + self.large_count_limit
        if self.replay_capacity != expected_capacity:
            raise ValueError(
                "optimization.replay_capacity must equal medium_count_limit + "
                f"large_count_limit ({self.medium_count_limit}+{self.large_count_limit}"
                f"={expected_capacity}); DualPool does not use a flat 2048-slot buffer"
            )
        if self.min_replay_size < LOGICAL_BATCH_SIZE:
            raise ValueError("min_replay_size must be at least the DualPool logical batch of 16")
        if self.replay_capacity < self.min_replay_size:
            raise ValueError("GCNN replay capacity is too small")
        if self.n_step <= 0 or self.gamma != 1.0:
            raise ValueError("GCNN BBMDP training requires n_step>0 and gamma=1")
        if not 0.0 < self.target_tau <= 1.0:
            raise ValueError("target_tau must be in (0, 1]")


@dataclass(frozen=True)
class GCNNExplorationConfig:
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    decay_steps: int = 1000
    boltzmann_temperature: float = 0.0

    def epsilon(self, gradient_step: int) -> float:
        fraction = min(max(gradient_step, 0) / max(self.decay_steps, 1), 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)


@dataclass(frozen=True)
class GCNNEvaluationConfig:
    seeds: tuple[int, ...] = (100,)
    interval_steps: int = 50
    early_stopping_patience: int = 4
    compare_seeds: tuple[int, ...] = (100,)


@dataclass(frozen=True)
class GCNNTrainingConfig:
    run_name: str
    seed: int
    device: str
    output_dir: str
    train_instances: tuple[str, ...]
    validation_instances: tuple[str, ...]
    environment: BBMDPConfig
    model: GCNNModelConfig = field(default_factory=GCNNModelConfig)
    optimization: GCNNOptimizationConfig = field(default_factory=GCNNOptimizationConfig)
    exploration: GCNNExplorationConfig = field(default_factory=GCNNExplorationConfig)
    evaluation: GCNNEvaluationConfig = field(default_factory=GCNNEvaluationConfig)
    normalization_warmup_states: int = 2
    normalization_path: str = ""
    log_interval_steps: int = 10
    wall_time_limit: float = 0.0
    require_mix_16_0: bool = False
    require_mix_12_4: bool = False
    skip_mid_validation: bool = False
    skip_final_comparison: bool = False
    deploy_retest_instance: str = ""
    deploy_retest_seed_overlay: str = ""
    deploy_retest_solve_node_limit: int = 2

    def __post_init__(self) -> None:
        if not self.train_instances or not self.validation_instances:
            raise ValueError("GCNN train and validation instances must be non-empty")
        if self.environment.gamma != 1.0:
            raise ValueError("GCNN BBMDP environment requires gamma=1")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("GCNN device must be cpu or cuda")
        if self.normalization_warmup_states <= 0:
            raise ValueError("normalization_warmup_states must be positive")
        if self.wall_time_limit < 0.0:
            raise ValueError("wall_time_limit must be non-negative")
        if self.require_mix_12_4 and not any(
            Path(path).stem.startswith("real_02") for path in self.train_instances
        ):
            raise ValueError("require_mix_12_4 needs a real_02 train instance")
        if self.deploy_retest_solve_node_limit < 1:
            raise ValueError("deploy_retest_solve_node_limit must be at least 1")

    @classmethod
    def from_yaml(cls, path: Path | str) -> "GCNNTrainingConfig":
        with Path(path).open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        expected = {
            "run_name",
            "seed",
            "device",
            "output_dir",
            "train_instances",
            "validation_instances",
            "environment",
            "model",
            "optimization",
            "exploration",
            "evaluation",
            "normalization_warmup_states",
            "normalization_path",
            "log_interval_steps",
            "wall_time_limit",
            "require_mix_16_0",
            "require_mix_12_4",
            "skip_mid_validation",
            "skip_final_comparison",
            "deploy_retest_instance",
            "deploy_retest_seed_overlay",
            "deploy_retest_solve_node_limit",
        }
        unknown = set(raw) - expected
        if unknown:
            raise ValueError(f"unknown GCNN config keys: {sorted(unknown)}")
        environment = dict(raw["environment"])
        if "reward_mode" in environment:
            environment["reward_mode"] = RewardMode(environment["reward_mode"])
        evaluation = dict(raw.get("evaluation", {}))
        for key in ("seeds", "compare_seeds"):
            if key in evaluation:
                evaluation[key] = tuple(evaluation[key])
        return cls(
            run_name=str(raw["run_name"]),
            seed=int(raw["seed"]),
            device=str(raw["device"]),
            output_dir=str(raw["output_dir"]),
            train_instances=tuple(raw["train_instances"]),
            validation_instances=tuple(raw["validation_instances"]),
            environment=BBMDPConfig(**environment),
            model=GCNNModelConfig(**raw.get("model", {})),
            optimization=GCNNOptimizationConfig(**raw.get("optimization", {})),
            exploration=GCNNExplorationConfig(**raw.get("exploration", {})),
            evaluation=GCNNEvaluationConfig(**evaluation),
            normalization_warmup_states=int(raw.get("normalization_warmup_states", 2)),
            normalization_path=str(raw.get("normalization_path", "")),
            log_interval_steps=int(raw.get("log_interval_steps", 10)),
            wall_time_limit=float(raw.get("wall_time_limit", 0.0)),
            require_mix_16_0=bool(raw.get("require_mix_16_0", False)),
            require_mix_12_4=bool(raw.get("require_mix_12_4", False)),
            skip_mid_validation=bool(raw.get("skip_mid_validation", False)),
            skip_final_comparison=bool(raw.get("skip_final_comparison", False)),
            deploy_retest_instance=str(raw.get("deploy_retest_instance", "")),
            deploy_retest_seed_overlay=str(raw.get("deploy_retest_seed_overlay", "")),
            deploy_retest_solve_node_limit=int(raw.get("deploy_retest_solve_node_limit", 2)),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["environment"]["reward_mode"] = self.environment.reward_mode.value
        return result
