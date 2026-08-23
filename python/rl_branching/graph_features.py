from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from .candidate_features import (
    AVIATION_VARIABLE_CATEGORIES,
    ECOLE_VARIABLE_FEATURE_NAMES,
    category_one_hot,
)
from .observation import (
    EDGE_FEATURE_NAMES,
    EXTENDED_ROW_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    BipartiteObservation,
)
from .prim_bias import PRIM_VARIABLE_FEATURE_NAMES, prim_variable_feature_matrix

# Phase B graph variable width = Ecole(19) + Prim neighborhood(6).
GRAPH_VARIABLE_FEATURE_NAMES = ECOLE_VARIABLE_FEATURE_NAMES + PRIM_VARIABLE_FEATURE_NAMES


AVIATION_CONSTRAINT_CATEGORIES = (
    "flow",
    "absolute",
    "topology",
    "selection",
    "imbalance",
    "other",
)


def aviation_constraint_category(row_name: str) -> int:
    name = str(row_name)
    while name.startswith("t_"):
        name = name[2:]
    if name.startswith("flow_") or name.startswith("fforbid"):
        return 0
    if name.startswith("abs"):
        return 1
    if name.startswith("topo_") or name.startswith("only_father"):
        return 2
    if name.startswith("zlower") or name.startswith("onlym"):
        return 3
    if name.startswith("imbalance"):
        return 4
    return len(AVIATION_CONSTRAINT_CATEGORIES) - 1


def constraint_category_one_hot(row_names: Iterable[str]) -> np.ndarray:
    names = tuple(row_names)
    encoded = np.zeros(
        (len(names), len(AVIATION_CONSTRAINT_CATEGORIES)), dtype=np.float32
    )
    if names:
        encoded[
            np.arange(len(names)),
            [aviation_constraint_category(name) for name in names],
        ] = 1.0
    return encoded


def _immutable(values, dtype=np.float32) -> np.ndarray:
    array = np.asarray(values)
    target_dtype = np.dtype(dtype)
    if array.dtype == target_dtype and not array.flags.writeable:
        return array
    array = np.asarray(values, dtype=target_dtype).copy()
    if np.issubdtype(array.dtype, np.floating):
        np.nan_to_num(array, copy=False, nan=0.0, posinf=1e20, neginf=-1e20)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class GraphState:
    row_features: np.ndarray
    variable_features: np.ndarray
    edge_indices: np.ndarray
    edge_features: np.ndarray
    global_features: np.ndarray
    variable_categories: np.ndarray
    row_categories: np.ndarray
    actions: np.ndarray
    candidate_names: Tuple[str, ...]
    variable_names: Tuple[str, ...] = ()

    @property
    def candidate_count(self) -> int:
        return int(self.actions.size)

    def validate(self) -> None:
        n_rows = self.row_features.shape[0]
        n_variables = self.variable_features.shape[0]
        if self.row_features.shape != (n_rows, len(EXTENDED_ROW_FEATURE_NAMES)):
            raise ValueError("graph row features have the wrong shape")
        if self.variable_features.shape != (
            n_variables,
            len(GRAPH_VARIABLE_FEATURE_NAMES),
        ):
            raise ValueError("graph variable features have the wrong shape")
        if self.edge_indices.ndim != 2 or self.edge_indices.shape[0] != 2:
            raise ValueError("graph edge indices must have shape [2, n_edges]")
        if self.edge_features.shape != (
            self.edge_indices.shape[1],
            len(EDGE_FEATURE_NAMES),
        ):
            raise ValueError("graph edge features have the wrong shape")
        if self.global_features.shape != (len(GLOBAL_FEATURE_NAMES),):
            raise ValueError("graph global features have the wrong shape")
        if self.variable_categories.shape != (
            n_variables,
            len(AVIATION_VARIABLE_CATEGORIES),
        ):
            raise ValueError("graph variable categories have the wrong shape")
        if self.row_categories.shape != (
            n_rows,
            len(AVIATION_CONSTRAINT_CATEGORIES),
        ):
            raise ValueError("graph row categories have the wrong shape")
        if self.actions.ndim != 1 or self.actions.size == 0:
            raise ValueError("graph action set must be a non-empty vector")
        if self.actions.min() < 0 or self.actions.max() >= n_variables:
            raise ValueError("graph action set contains an invalid variable index")
        if len(self.candidate_names) != self.candidate_count:
            raise ValueError("candidate names must align with graph actions")
        if self.variable_names and len(self.variable_names) != n_variables:
            raise ValueError("variable names must align with variable features")
        if self.edge_indices.size:
            if self.edge_indices[0].min() < 0 or self.edge_indices[0].max() >= n_rows:
                raise ValueError("graph edge contains an invalid row index")
            if self.edge_indices[1].min() < 0 or self.edge_indices[1].max() >= n_variables:
                raise ValueError("graph edge contains an invalid variable index")
        for values in (
            self.row_features,
            self.variable_features,
            self.edge_indices,
            self.edge_features,
            self.global_features,
            self.variable_categories,
            self.row_categories,
            self.actions,
        ):
            if values.flags.writeable:
                raise ValueError("graph state arrays must be immutable")
            if not np.isfinite(values).all():
                raise ValueError("graph state contains NaN or Inf")


def extract_graph_state(
    observation: BipartiteObservation,
    action_set: np.ndarray,
) -> GraphState:
    if observation.extended_row_features is None or observation.edge_features is None:
        raise ValueError("observation does not contain phase-7 graph features")
    if not observation.row_names:
        raise ValueError("observation does not contain row names")
    actions = _immutable(action_set, np.int64)
    ecole_features = np.asarray(observation.variable_features, dtype=np.float32)
    if observation.local_lower_bounds is None:
        raise ValueError("observation is missing transformed local lower bounds")
    prim_features = prim_variable_feature_matrix(
        observation.variable_names,
        lower_bounds=observation.local_lower_bounds,
    )
    if ecole_features.ndim != 2 or ecole_features.shape[1] != len(ECOLE_VARIABLE_FEATURE_NAMES):
        raise ValueError("observation variable features must have shape [n, 19]")
    if prim_features.shape != (ecole_features.shape[0], len(PRIM_VARIABLE_FEATURE_NAMES)):
        raise ValueError("prim variable features have the wrong shape")
    variable_features = np.concatenate([ecole_features, prim_features], axis=1)
    state = GraphState(
        row_features=_immutable(observation.extended_row_features),
        variable_features=_immutable(variable_features),
        edge_indices=_immutable(observation.edge_indices, np.int64),
        edge_features=_immutable(observation.edge_features),
        global_features=_immutable(observation.global_features),
        variable_categories=_immutable(category_one_hot(observation.variable_names)),
        row_categories=_immutable(constraint_category_one_hot(observation.row_names)),
        actions=actions,
        candidate_names=tuple(observation.variable_names[int(action)] for action in actions),
        variable_names=tuple(observation.variable_names),
    )
    state.validate()
    return state


def graph_state_storage_bytes(state: GraphState) -> int:
    return int(
        state.row_features.nbytes
        + state.variable_features.nbytes
        + state.edge_indices.nbytes
        + state.edge_features.nbytes
        + state.global_features.nbytes
        + state.variable_categories.nbytes
        + state.row_categories.nbytes
        + state.actions.nbytes
    )


def transition_storage_bytes(
    state: GraphState,
    next_state: GraphState | None,
    *,
    container_overhead: int = 64,
) -> int:
    total = graph_state_storage_bytes(state) + int(container_overhead)
    if next_state is not None:
        total += graph_state_storage_bytes(next_state)
    return int(total)


def candidate_twohop_state(state: GraphState) -> GraphState:
    n_variables = state.variable_features.shape[0]
    n_rows = state.row_features.shape[0]
    candidates = np.asarray(state.actions, dtype=np.int64)
    if candidates.size == 0:
        raise ValueError("candidate two-hop requires a non-empty action set")
    row_idx = np.asarray(state.edge_indices[0], dtype=np.int64)
    var_idx = np.asarray(state.edge_indices[1], dtype=np.int64)
    candidate_mask = np.zeros(n_variables, dtype=bool)
    candidate_mask[candidates] = True
    keep_row_mask = np.zeros(n_rows, dtype=bool)
    if row_idx.size:
        keep_row_mask[np.unique(row_idx[candidate_mask[var_idx]])] = True
    keep_edge = keep_row_mask[row_idx] if row_idx.size else np.zeros(0, dtype=bool)
    keep_var_mask = candidate_mask.copy()
    if keep_edge.size:
        keep_var_mask[var_idx[keep_edge]] = True
    keep_rows = np.flatnonzero(keep_row_mask)
    keep_vars = np.flatnonzero(keep_var_mask)
    row_map = np.full(n_rows, -1, dtype=np.int64)
    var_map = np.full(n_variables, -1, dtype=np.int64)
    row_map[keep_rows] = np.arange(keep_rows.size, dtype=np.int64)
    var_map[keep_vars] = np.arange(keep_vars.size, dtype=np.int64)
    if keep_edge.size:
        new_edges = np.vstack(
            [row_map[row_idx[keep_edge]], var_map[var_idx[keep_edge]]]
        )
        new_edge_features = np.asarray(state.edge_features)[keep_edge]
    else:
        new_edges = np.zeros((2, 0), dtype=np.int64)
        new_edge_features = np.zeros((0, state.edge_features.shape[1]), dtype=np.float32)
    new_actions = var_map[candidates]
    if np.any(new_actions < 0):
        raise ValueError("candidate missing from the union two-hop variable set")
    names = state.variable_names
    restricted = GraphState(
        row_features=_immutable(np.asarray(state.row_features)[keep_rows]),
        variable_features=_immutable(np.asarray(state.variable_features)[keep_vars]),
        edge_indices=_immutable(new_edges, np.int64),
        edge_features=_immutable(new_edge_features),
        global_features=_immutable(state.global_features),
        variable_categories=_immutable(np.asarray(state.variable_categories)[keep_vars]),
        row_categories=_immutable(np.asarray(state.row_categories)[keep_rows]),
        actions=_immutable(new_actions, np.int64),
        candidate_names=state.candidate_names,
        variable_names=tuple(names[int(index)] for index in keep_vars) if names else (),
    )
    restricted.validate()
    return restricted


class RunningGraphNormalizer:
    FEATURE_GROUPS = {
        "variable": len(GRAPH_VARIABLE_FEATURE_NAMES),
        "row": len(EXTENDED_ROW_FEATURE_NAMES),
        "edge": len(EDGE_FEATURE_NAMES),
        "global": len(GLOBAL_FEATURE_NAMES),
    }

    def __init__(self) -> None:
        self.counts = {name: 0 for name in self.FEATURE_GROUPS}
        self.means = {
            name: np.zeros(width, dtype=np.float64)
            for name, width in self.FEATURE_GROUPS.items()
        }
        self.m2 = {name: np.zeros_like(mean) for name, mean in self.means.items()}

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
        return (
            total,
            mean + delta * batch_count / total,
            m2 + batch_m2 + delta**2 * count * batch_count / total,
        )

    def update(self, state: GraphState) -> None:
        values = {
            "variable": state.variable_features,
            "row": state.row_features,
            "edge": state.edge_features,
            "global": state.global_features,
        }
        for name, group_values in values.items():
            self.counts[name], self.means[name], self.m2[name] = self._update_batch(
                self.counts[name], self.means[name], self.m2[name], group_values
            )

    def statistics(self) -> dict[str, np.ndarray]:
        result = {}
        for name in self.FEATURE_GROUPS:
            count = self.counts[name]
            if count < 2:
                std = np.ones_like(self.m2[name])
            else:
                std = np.sqrt(self.m2[name] / (count - 1))
                std[std < 1.0e-6] = 1.0
            result[f"{name}_mean"] = self.means[name].astype(np.float32)
            result[f"{name}_std"] = std.astype(np.float32)
        return result

    def to_json(self) -> dict:
        return {
            "counts": self.counts,
            **{key: value.tolist() for key, value in self.statistics().items()},
        }
