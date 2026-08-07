from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from .observation import BipartiteObservation, GLOBAL_FEATURE_NAMES


ECOLE_VARIABLE_FEATURE_NAMES = (
    "objective",
    "is_type_binary",
    "is_type_integer",
    "is_type_implicit_integer",
    "is_type_continuous",
    "has_lower_bound",
    "has_upper_bound",
    "normed_reduced_cost",
    "solution_value",
    "solution_frac",
    "is_solution_at_lower_bound",
    "is_solution_at_upper_bound",
    "scaled_age",
    "incumbent_value",
    "average_incumbent_value",
    "is_basis_lower",
    "is_basis_basic",
    "is_basis_upper",
    "is_basis_zero",
)

AVIATION_VARIABLE_CATEGORIES = ("m", "z", "y", "absf", "f", "other")


def aviation_variable_category(variable_name: str) -> int:
    name = str(variable_name)
    while name.startswith("t_"):
        name = name[2:]
    for index, category in enumerate(AVIATION_VARIABLE_CATEGORIES[:-1]):
        if name == category or name.startswith(f"{category}_"):
            return index
    return len(AVIATION_VARIABLE_CATEGORIES) - 1


def category_one_hot(variable_names: Iterable[str]) -> np.ndarray:
    names = tuple(variable_names)
    encoded = np.zeros((len(names), len(AVIATION_VARIABLE_CATEGORIES)), dtype=np.float32)
    if names:
        encoded[np.arange(len(names)), [aviation_variable_category(name) for name in names]] = 1.0
    return encoded


def _immutable(values, dtype=np.float32) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    np.nan_to_num(array, copy=False, nan=0.0, posinf=1e20, neginf=-1e20)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CandidateState:
    variable_features: np.ndarray
    global_features: np.ndarray
    category_features: np.ndarray
    actions: np.ndarray
    variable_names: Tuple[str, ...]

    @property
    def candidate_count(self) -> int:
        return int(self.actions.size)

    def validate(self) -> None:
        count = self.candidate_count
        if self.variable_features.shape != (count, len(ECOLE_VARIABLE_FEATURE_NAMES)):
            raise ValueError("candidate variable features have the wrong shape")
        if self.global_features.shape != (len(GLOBAL_FEATURE_NAMES),):
            raise ValueError("candidate global features have the wrong shape")
        if self.category_features.shape != (count, len(AVIATION_VARIABLE_CATEGORIES)):
            raise ValueError("candidate category features have the wrong shape")
        if len(self.variable_names) != count:
            raise ValueError("candidate names have the wrong length")
        if count == 0:
            raise ValueError("a CandidateState must contain at least one candidate")
        for values in (self.variable_features, self.global_features, self.category_features, self.actions):
            if values.flags.writeable:
                raise ValueError("candidate state arrays must be immutable")
            if not np.isfinite(values).all():
                raise ValueError("candidate state contains NaN or Inf")


def extract_candidate_state(
    observation: BipartiteObservation,
    action_set: np.ndarray,
) -> CandidateState:
    actions = _immutable(action_set, np.int64)
    if actions.ndim != 1 or actions.size == 0:
        raise ValueError("action_set must be a non-empty vector")
    if actions.min() < 0 or actions.max() >= observation.variable_features.shape[0]:
        raise ValueError("action_set contains an out-of-range variable index")

    names = tuple(observation.variable_names[int(action)] for action in actions)
    state = CandidateState(
        variable_features=_immutable(observation.variable_features[actions]),
        global_features=_immutable(observation.global_features),
        category_features=_immutable(category_one_hot(names)),
        actions=actions,
        variable_names=names,
    )
    state.validate()
    return state


class RunningFeatureNormalizer:
    def __init__(self) -> None:
        self.variable_count = 0
        self.variable_mean = np.zeros(len(ECOLE_VARIABLE_FEATURE_NAMES), dtype=np.float64)
        self.variable_m2 = np.zeros_like(self.variable_mean)
        self.global_count = 0
        self.global_mean = np.zeros(len(GLOBAL_FEATURE_NAMES), dtype=np.float64)
        self.global_m2 = np.zeros_like(self.global_mean)

    @staticmethod
    def _update_batch(count, mean, m2, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        batch_count = values.shape[0]
        if batch_count == 0:
            return count, mean, m2
        batch_mean = values.mean(axis=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(axis=0)
        if count == 0:
            return batch_count, batch_mean, batch_m2
        delta = batch_mean - mean
        total = count + batch_count
        merged_mean = mean + delta * batch_count / total
        merged_m2 = m2 + batch_m2 + delta**2 * count * batch_count / total
        return total, merged_mean, merged_m2

    def update(self, state: CandidateState) -> None:
        self.variable_count, self.variable_mean, self.variable_m2 = self._update_batch(
            self.variable_count,
            self.variable_mean,
            self.variable_m2,
            state.variable_features,
        )
        self.global_count, self.global_mean, self.global_m2 = self._update_batch(
            self.global_count,
            self.global_mean,
            self.global_m2,
            state.global_features,
        )

    @staticmethod
    def _std(count: int, m2: np.ndarray) -> np.ndarray:
        if count < 2:
            return np.ones_like(m2, dtype=np.float32)
        std = np.sqrt(m2 / (count - 1))
        std[std < 1e-6] = 1.0
        return std.astype(np.float32)

    def statistics(self) -> dict[str, np.ndarray]:
        return {
            "variable_mean": self.variable_mean.astype(np.float32),
            "variable_std": self._std(self.variable_count, self.variable_m2),
            "global_mean": self.global_mean.astype(np.float32),
            "global_std": self._std(self.global_count, self.global_m2),
        }

    def to_json(self) -> dict:
        stats = self.statistics()
        return {
            "variable_count": self.variable_count,
            "global_count": self.global_count,
            **{key: value.tolist() for key, value in stats.items()},
        }
