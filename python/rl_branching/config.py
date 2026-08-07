from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict

import yaml


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
    cache_static_features: bool = True

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.time_limit <= 0:
            raise ValueError("time_limit must be positive")
        if self.node_limit == 0 or self.node_limit < -1:
            raise ValueError("node_limit must be -1 or a positive integer")
        if self.gamma != 1.0:
            raise ValueError("the BBMDP-faithful profile requires gamma=1")

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
        parameters: Dict[str, Any] = {
            "nodeselection/dfs/stdpriority": 1_000_000,
            "nodeselection/dfs/memsavepriority": 1_000_000,
            "separating/maxrounds": 0,
            "estimation/restarts/restartpolicy": "n",
            "limits/restarts": 0,
            "presolving/maxrestarts": 0,
            "parallel/minnthreads": 1,
            "parallel/maxnthreads": 1,
            "lp/threads": 1,
            "randomization/randomseedshift": self.seed,
            "randomization/permutationseed": self.seed,
            "randomization/lpseed": self.seed,
            "limits/time": self.time_limit,
        }
        if self.node_limit >= 0:
            parameters["limits/nodes"] = self.node_limit
        return parameters
