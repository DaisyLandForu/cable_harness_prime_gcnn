"""Exact candidate closures for the one-round B0 bipartite network."""

from __future__ import annotations

import numpy as np

from .bipartite_observation import MilpBipartiteState, make_bipartite_state


def candidate_exact_closure(state: MilpBipartiteState) -> MilpBipartiteState:
    """Return the exact receptive field of every candidate after one B0 round.

    The closure is the union of candidate variables, their incident constraints,
    and every variable incident to those constraints. Edge order and global IDs
    remain stable so full/closure parity is directly auditable.
    """
    candidates = np.asarray(state.candidate_indices, dtype=np.int64)
    if candidates.size == 0:
        return make_bipartite_state(
            constraint_features=np.zeros((0, state.constraint_features.shape[1]), dtype=np.float32),
            variable_features=np.zeros((0, state.variable_features.shape[1]), dtype=np.float32),
            edge_indices=np.zeros((2, 0), dtype=np.int64),
            edge_features=np.zeros((0, state.edge_features.shape[1]), dtype=np.float32),
            variable_names=(),
            variable_global_ids=np.zeros(0, dtype=np.int64),
            constraint_global_ids=np.zeros(0, dtype=np.int64),
            candidate_indices=np.zeros(0, dtype=np.int64),
            candidate_edge_ids=np.zeros(0, dtype=np.int64),
        )

    n_constraints = state.constraint_features.shape[0]
    n_variables = state.variable_features.shape[0]
    row_indices = np.asarray(state.edge_indices[0], dtype=np.int64)
    variable_indices = np.asarray(state.edge_indices[1], dtype=np.int64)

    candidate_mask = np.zeros(n_variables, dtype=bool)
    candidate_mask[candidates] = True
    keep_constraint_mask = np.zeros(n_constraints, dtype=bool)
    if row_indices.size:
        keep_constraint_mask[np.unique(row_indices[candidate_mask[variable_indices]])] = True
    keep_edge_mask = (
        keep_constraint_mask[row_indices]
        if row_indices.size
        else np.zeros(0, dtype=bool)
    )
    keep_variable_mask = candidate_mask.copy()
    if keep_edge_mask.size:
        keep_variable_mask[variable_indices[keep_edge_mask]] = True

    kept_constraints = np.flatnonzero(keep_constraint_mask)
    kept_variables = np.flatnonzero(keep_variable_mask)
    constraint_map = np.full(n_constraints, -1, dtype=np.int64)
    variable_map = np.full(n_variables, -1, dtype=np.int64)
    constraint_map[kept_constraints] = np.arange(kept_constraints.size, dtype=np.int64)
    variable_map[kept_variables] = np.arange(kept_variables.size, dtype=np.int64)

    if keep_edge_mask.size:
        restricted_edges = np.vstack(
            (
                constraint_map[row_indices[keep_edge_mask]],
                variable_map[variable_indices[keep_edge_mask]],
            )
        )
        restricted_edge_features = state.edge_features[keep_edge_mask]
    else:
        restricted_edges = np.zeros((2, 0), dtype=np.int64)
        restricted_edge_features = np.zeros((0, state.edge_features.shape[1]), dtype=np.float32)

    restricted_candidates = variable_map[candidates]
    if np.any(restricted_candidates < 0):
        raise RuntimeError("candidate was lost while constructing its exact closure")
    return make_bipartite_state(
        constraint_features=state.constraint_features[kept_constraints],
        variable_features=state.variable_features[kept_variables],
        edge_indices=restricted_edges,
        edge_features=restricted_edge_features,
        variable_names=tuple(state.variable_names[int(index)] for index in kept_variables),
        variable_global_ids=state.variable_global_ids[kept_variables],
        constraint_global_ids=state.constraint_global_ids[kept_constraints],
        candidate_indices=restricted_candidates,
        candidate_edge_ids=state.candidate_edge_ids,
    )
