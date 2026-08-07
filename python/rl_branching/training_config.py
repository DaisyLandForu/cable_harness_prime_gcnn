from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import BBMDPConfig, RewardMode


@dataclass(frozen=True)
class ModelConfig:
    hidden_sizes: tuple[int, ...] = (128, 128)


@dataclass(frozen=True)
class OptimizationConfig:
    total_gradient_steps: int = 500
    batch_size: int = 16
    replay_capacity: int = 5000
    min_replay_size: int = 16
    learning_rate: float = 0.0003
    updates_per_env_step: int = 16
    n_step: int = 3
    gamma: float = 1.0
    gradient_clip: float = 10.0
    target_update_interval: int = 100


@dataclass(frozen=True)
class ExplorationConfig:
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    decay_steps: int = 400

    def epsilon(self, gradient_step: int) -> float:
        fraction = min(max(gradient_step, 0) / max(self.decay_steps, 1), 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)


@dataclass(frozen=True)
class EvaluationConfig:
    seeds: tuple[int, ...] = (100,)
    interval_steps: int = 250
    early_stopping_patience: int = 4
    compare_seeds: tuple[int, ...] = (100, 101, 102)


@dataclass(frozen=True)
class MLPTrainingConfig:
    run_name: str
    seed: int
    device: str
    output_dir: str
    train_instances: tuple[str, ...]
    validation_instances: tuple[str, ...]
    environment: BBMDPConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    normalization_warmup_states: int = 16
    log_interval_steps: int = 25

    def __post_init__(self) -> None:
        if not self.train_instances or not self.validation_instances:
            raise ValueError("train and validation instances must be non-empty")
        if self.environment.gamma != 1.0 or self.optimization.gamma != 1.0:
            raise ValueError("BBMDP Candidate MLP requires gamma=1")
        if self.optimization.n_step <= 0:
            raise ValueError("n_step must be positive")
        if self.optimization.batch_size > self.optimization.min_replay_size:
            raise ValueError("batch_size cannot exceed min_replay_size")
        if self.normalization_warmup_states <= 0:
            raise ValueError("normalization_warmup_states must be positive")

    @classmethod
    def from_yaml(cls, path: Path | str) -> "MLPTrainingConfig":
        with Path(path).open() as stream:
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
            "log_interval_steps",
        }
        unknown = set(raw) - expected
        if unknown:
            raise ValueError(f"unknown MLP config keys: {sorted(unknown)}")

        environment = dict(raw["environment"])
        if "reward_mode" in environment:
            environment["reward_mode"] = RewardMode(environment["reward_mode"])
        model = dict(raw.get("model", {}))
        if "hidden_sizes" in model:
            model["hidden_sizes"] = tuple(model["hidden_sizes"])
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
            model=ModelConfig(**model),
            optimization=OptimizationConfig(**raw.get("optimization", {})),
            exploration=ExplorationConfig(**raw.get("exploration", {})),
            evaluation=EvaluationConfig(**evaluation),
            normalization_warmup_states=int(raw.get("normalization_warmup_states", 16)),
            log_interval_steps=int(raw.get("log_interval_steps", 25)),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["environment"]["reward_mode"] = self.environment.reward_mode.value
        return result
