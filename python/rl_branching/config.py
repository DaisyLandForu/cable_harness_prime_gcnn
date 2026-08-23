from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict

import yaml

from .scip_profile import load_production_scip_params, resolve_scip_profile


class RewardMode(str, Enum):
    NEGATIVE_NODE_INCREMENT = "negative_node_increment"
    CONSTANT_MINUS_ONE = "constant_minus_one"


@dataclass(frozen=True)
class BBMDPConfig:
    seed: int = 0
    time_limit: float = 60.0
    node_limit: int = 1000
    gamma: float = 1.0
    reward_mode: RewardMode = RewardMode.NEGATIVE_NODE_INCREMENT
    bootstrap_on_truncation: bool = False
    cache_static_features: bool = False
    scip_profile: str = ""

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.time_limit <= 0:
            raise ValueError("time_limit must be positive")
        if self.node_limit == 0 or self.node_limit < -1:
            raise ValueError("node_limit must be -1 or a positive integer")
        if self.gamma != 1.0:
            raise ValueError("the BBMDP-faithful profile requires gamma=1")
        object.__setattr__(self, "scip_profile", str(resolve_scip_profile(self.scip_profile)))

    @classmethod
    def from_yaml(cls, path: Path | str) -> "BBMDPConfig":
        with Path(path).open() as stream:
            raw = yaml.safe_load(stream) or {}
        unknown = set(raw) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown BBMDP config keys: {sorted(unknown)}")
        if "reward_mode" in raw:
            raw["reward_mode"] = RewardMode(raw["reward_mode"])
        return cls(**raw)

    def scip_parameters(self) -> Dict[str, Any]:
        return load_production_scip_params(
            seed=self.seed,
            time_limit=self.time_limit,
            node_limit=self.node_limit,
            profile=self.scip_profile,
        )
