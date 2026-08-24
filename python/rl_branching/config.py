from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable
import math

import yaml

from .scip_profile import load_production_scip_params, resolve_scip_profile


class RewardMode(str, Enum):
    NEGATIVE_NODE_INCREMENT = "negative_node_increment"
    CONSTANT_MINUS_ONE = "constant_minus_one"
    HYBRID_NODE_LP = "hybrid_node_lp"


HYBRID_LP_COEFF = 1.0e-4
MISSING_METRIC = float("inf")
SELECTION_RULE = (
    "neg_solved_rate, mean_pdi, mean_final_gap, mean_par2, mean_lp, mean_nodes, "
    "earlier_gradient_step, lower_seed"
)


def hybrid_rewards(delta_nodes: int, delta_lp: int) -> tuple[float, float, float]:
    node_reward = -float(max(0, int(delta_nodes)))
    lp_reward = -HYBRID_LP_COEFF * float(max(0, int(delta_lp)))
    return node_reward, lp_reward, node_reward + lp_reward


def hybrid_identity_holds(
    rewards: Iterable[float],
    n0: int,
    n_t: int,
    lp0: int,
    lp_t: int,
    *,
    atol: float = 1.0e-8,
) -> bool:
    expected = -float(n_t - n0) - HYBRID_LP_COEFF * float(lp_t - lp0)
    return abs(float(sum(rewards)) - expected) <= atol * max(1.0, abs(expected))


def finalized_gap(status: str, primal: float, dual: float, scip_gap: float) -> float:
    if str(status).lower() == "optimal":
        return 0.0
    if not math.isfinite(float(primal)) or not math.isfinite(float(dual)):
        return MISSING_METRIC
    if math.isfinite(float(scip_gap)) and float(scip_gap) >= 0.0:
        return float(scip_gap)
    denom = min(abs(float(primal)), abs(float(dual)))
    if denom < 1.0e-12:
        return 0.0 if abs(float(primal) - float(dual)) <= 1.0e-9 else MISSING_METRIC
    return abs(float(primal) - float(dual)) / denom


def par2_time(status: str, solving_time: float, time_limit: float) -> float:
    if str(status).lower() == "optimal":
        return float(solving_time)
    return 2.0 * float(time_limit)


def solved_flag(status: str) -> float:
    return 1.0 if str(status).lower() == "optimal" else 0.0


def missing_to_inf(value: float) -> float:
    number = float(value)
    return number if math.isfinite(number) else MISSING_METRIC


def metric_mean(values: Iterable[float]) -> float:
    converted = [missing_to_inf(value) for value in values]
    if not converted or any(math.isinf(value) for value in converted):
        return MISSING_METRIC
    return float(sum(converted) / len(converted))


def selection_key(
    *,
    solved_rate: float,
    mean_pdi: float,
    mean_final_gap: float,
    mean_par2: float,
    mean_lp: float,
    mean_nodes: float,
    gradient_step: int,
    seed: int,
) -> tuple:
    return (
        -float(solved_rate),
        float(mean_pdi),
        float(mean_final_gap),
        float(mean_par2),
        float(mean_lp),
        float(mean_nodes),
        int(gradient_step),
        int(seed),
    )


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

    def scip_search_limits(self) -> tuple[float, int]:
        """SCIP hard caps. Live-budget truncation uses the public time/node limits."""
        if not self.bootstrap_on_truncation:
            return float(self.time_limit), int(self.node_limit)
        node_limit = int(self.node_limit)
        if node_limit > 0:
            node_limit += 1
        return float(self.time_limit) + 5.0, node_limit

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
        time_limit, node_limit = self.scip_search_limits()
        return load_production_scip_params(
            seed=self.seed,
            time_limit=time_limit,
            node_limit=node_limit,
            profile=self.scip_profile,
        )
