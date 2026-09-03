"""Aviation-independent Ecole-style 19/5/1 MILP bipartite observations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable

import ecole
import numpy as np

from ..contracts import GraphSchema, ProblemMetadata
from ..milp.naming import edge_id_from_scip_variable_name, original_variable_name


VARIABLE_FEATURE_NAMES = (
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

CONSTRAINT_FEATURE_NAMES = (
    "normalized_bias",
    "objective_cosine_similarity",
    "is_tight",
    "normalized_dual_solution_value",
    "scaled_age",
)

EDGE_FEATURE_NAMES = ("normalized_coefficient",)

# Ecole deliberately emits NaN for these two features until SCIP has an
# incumbent. B0 has no extra missingness channel, so v1 uses a documented zero
# sentinel for exactly these undefined values and rejects every other NaN/Inf.
UNDEFINED_INCUMBENT_FEATURE_INDICES = (13, 14)

MILP_BIPARTITE_V1 = GraphSchema(
    schema_id="milp_bipartite_v1",
    version=1,
    variable_features=VARIABLE_FEATURE_NAMES,
    constraint_features=CONSTRAINT_FEATURE_NAMES,
    edge_features=EDGE_FEATURE_NAMES,
    candidate_entity_kinds=("EDGE",),
)


def _immutable_copy(values: Any, dtype: np.dtype[Any] | type) -> np.ndarray:
    copied = np.array(values, dtype=dtype, copy=True, order="C")
    copied.setflags(write=False)
    return copied


def _update_array_hash(digest: Any, label: str, values: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(values)
    digest.update(label.encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))


@dataclass(frozen=True)
class MilpBipartiteState:
    """Immutable B0 state with local indices and stable full-graph identities."""

    constraint_features: np.ndarray
    variable_features: np.ndarray
    edge_indices: np.ndarray
    edge_features: np.ndarray
    variable_names: tuple[str, ...]
    variable_global_ids: np.ndarray
    constraint_global_ids: np.ndarray
    candidate_indices: np.ndarray
    candidate_edge_ids: np.ndarray
    schema_id: str = MILP_BIPARTITE_V1.schema_id

    def __post_init__(self) -> None:
        self.validate()

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_indices.size)

    @property
    def candidate_names(self) -> tuple[str, ...]:
        return tuple(self.variable_names[int(index)] for index in self.candidate_indices)

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema_id.encode("ascii"))
        for label, values in (
            ("constraints", self.constraint_features),
            ("variables", self.variable_features),
            ("edge_indices", self.edge_indices),
            ("edge_features", self.edge_features),
            ("variable_global_ids", self.variable_global_ids),
            ("constraint_global_ids", self.constraint_global_ids),
            ("candidate_indices", self.candidate_indices),
            ("candidate_edge_ids", self.candidate_edge_ids),
        ):
            _update_array_hash(digest, label, values)
        for name in self.variable_names:
            encoded = name.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.hexdigest()

    def validate(self) -> None:
        if self.schema_id != MILP_BIPARTITE_V1.schema_id:
            raise ValueError(f"unsupported bipartite schema: {self.schema_id}")
        n_constraints = int(self.constraint_features.shape[0])
        n_variables = int(self.variable_features.shape[0])
        if self.constraint_features.shape != (n_constraints, len(CONSTRAINT_FEATURE_NAMES)):
            raise ValueError("constraint features must have shape [n_constraints, 5]")
        if self.variable_features.shape != (n_variables, len(VARIABLE_FEATURE_NAMES)):
            raise ValueError("variable features must have shape [n_variables, 19]")
        if self.edge_indices.ndim != 2 or self.edge_indices.shape[0] != 2:
            raise ValueError("edge indices must have shape [2, n_edges]")
        if self.edge_features.shape != (self.edge_indices.shape[1], len(EDGE_FEATURE_NAMES)):
            raise ValueError("edge features must have shape [n_edges, 1]")
        if len(self.variable_names) != n_variables or len(set(self.variable_names)) != n_variables:
            raise ValueError("variable names must be unique and align with variable features")
        if self.variable_global_ids.shape != (n_variables,):
            raise ValueError("variable_global_ids must align with variables")
        if self.constraint_global_ids.shape != (n_constraints,):
            raise ValueError("constraint_global_ids must align with constraints")
        if len(set(int(value) for value in self.variable_global_ids)) != n_variables:
            raise ValueError("variable_global_ids must be unique")
        if len(set(int(value) for value in self.constraint_global_ids)) != n_constraints:
            raise ValueError("constraint_global_ids must be unique")
        if self.candidate_indices.ndim != 1:
            raise ValueError("candidate_indices must be one-dimensional")
        if self.candidate_edge_ids.shape != self.candidate_indices.shape:
            raise ValueError("candidate edge IDs must align with candidate indices")
        if len(set(int(value) for value in self.candidate_indices)) != self.candidate_count:
            raise ValueError("candidate indices must be unique")
        if len(set(int(value) for value in self.candidate_edge_ids)) != self.candidate_count:
            raise ValueError("candidate edge IDs must be unique")
        if self.candidate_count and (
            int(self.candidate_indices.min()) < 0
            or int(self.candidate_indices.max()) >= n_variables
            or int(self.candidate_edge_ids.min()) < 0
        ):
            raise ValueError("candidate index or edge ID is out of range")
        if self.edge_indices.size:
            if int(self.edge_indices[0].min()) < 0 or int(self.edge_indices[0].max()) >= n_constraints:
                raise ValueError("constraint edge index is out of range")
            if int(self.edge_indices[1].min()) < 0 or int(self.edge_indices[1].max()) >= n_variables:
                raise ValueError("variable edge index is out of range")
        for values in (
            self.constraint_features,
            self.variable_features,
            self.edge_indices,
            self.edge_features,
            self.variable_global_ids,
            self.constraint_global_ids,
            self.candidate_indices,
            self.candidate_edge_ids,
        ):
            if values.flags.writeable:
                raise ValueError("bipartite state arrays must be immutable copies")
            if not np.isfinite(values).all():
                raise ValueError("bipartite state contains NaN or Inf")


def make_bipartite_state(
    *,
    constraint_features: Any,
    variable_features: Any,
    edge_indices: Any,
    edge_features: Any,
    variable_names: Iterable[str],
    variable_global_ids: Any | None = None,
    constraint_global_ids: Any | None = None,
    candidate_indices: Any = (),
    candidate_edge_ids: Any = (),
) -> MilpBipartiteState:
    constraints = _immutable_copy(constraint_features, np.float32)
    variables = _immutable_copy(variable_features, np.float32)
    indices = _immutable_copy(edge_indices, np.int64)
    edge_values = np.asarray(edge_features)
    if edge_values.ndim == 1:
        edge_values = edge_values.reshape(-1, 1)
    edges = _immutable_copy(edge_values, np.float32)
    names = tuple(str(name) for name in variable_names)
    if variable_global_ids is None:
        variable_global_ids = np.arange(variables.shape[0], dtype=np.int64)
    if constraint_global_ids is None:
        constraint_global_ids = np.arange(constraints.shape[0], dtype=np.int64)
    return MilpBipartiteState(
        constraint_features=constraints,
        variable_features=variables,
        edge_indices=indices,
        edge_features=edges,
        variable_names=names,
        variable_global_ids=_immutable_copy(variable_global_ids, np.int64),
        constraint_global_ids=_immutable_copy(constraint_global_ids, np.int64),
        candidate_indices=_immutable_copy(candidate_indices, np.int64),
        candidate_edge_ids=_immutable_copy(candidate_edge_ids, np.int64),
    )


def copy_node_bipartite(raw: Any, variable_names: Iterable[str]) -> MilpBipartiteState:
    """Copy Ecole arrays and apply the v1 undefined-incumbent convention."""
    variables = np.array(raw.variable_features, dtype=np.float32, copy=True, order="C")
    nonfinite = ~np.isfinite(variables)
    if nonfinite.any():
        allowed = np.zeros_like(nonfinite, dtype=bool)
        allowed[:, UNDEFINED_INCUMBENT_FEATURE_INDICES] = np.isnan(
            variables[:, UNDEFINED_INCUMBENT_FEATURE_INDICES]
        )
        if np.any(nonfinite & ~allowed):
            raise ValueError("Ecole variable features contain an unsupported NaN or Inf")
        variables[:, UNDEFINED_INCUMBENT_FEATURE_INDICES] = np.nan_to_num(
            variables[:, UNDEFINED_INCUMBENT_FEATURE_INDICES], nan=0.0
        )
    return make_bipartite_state(
        constraint_features=raw.row_features,
        variable_features=variables,
        edge_indices=raw.edge_features.indices,
        edge_features=raw.edge_features.values,
        variable_names=variable_names,
    )


def with_legal_edge_actions(
    state: MilpBipartiteState,
    action_set: Any,
    metadata: ProblemMetadata,
    *,
    fraction_tolerance: float = 1.0e-9,
) -> MilpBipartiteState:
    """Attach the exact legal SPG edge actions or fail closed on protocol drift."""
    actions = np.array(action_set if action_set is not None else (), dtype=np.int64, copy=True)
    if actions.ndim != 1:
        raise ValueError("action_set must be one-dimensional")
    if len(set(int(value) for value in actions)) != int(actions.size):
        raise ValueError("action_set contains duplicate variable indices")
    if actions.size and (int(actions.min()) < 0 or int(actions.max()) >= state.variable_features.shape[0]):
        raise ValueError("action_set contains an out-of-range variable index")
    expected = {item.edge_id: item for item in metadata.edge_variables}
    edge_ids: list[int] = []
    for action in actions:
        index = int(action)
        features = state.variable_features[index]
        if not np.isclose(features[1], 1.0) or not np.isclose(features[1:5].sum(), 1.0):
            raise ValueError(f"SCIP action {index} is not a binary variable")
        fractionality = float(features[9])
        if not float(fraction_tolerance) < fractionality <= 0.5 + float(fraction_tolerance):
            raise ValueError(f"SCIP action {index} is not fractional")
        name = state.variable_names[index]
        edge_id = edge_id_from_scip_variable_name(name)
        item = expected.get(edge_id)
        if item is None or item.variable_name != original_variable_name(name):
            raise ValueError(f"SCIP action {name!r} has no exact metadata edge mapping")
        edge_ids.append(edge_id)
    return make_bipartite_state(
        constraint_features=state.constraint_features,
        variable_features=state.variable_features,
        edge_indices=state.edge_indices,
        edge_features=state.edge_features,
        variable_names=state.variable_names,
        variable_global_ids=state.variable_global_ids,
        constraint_global_ids=state.constraint_global_ids,
        candidate_indices=actions,
        candidate_edge_ids=edge_ids,
    )


class SteinerNodeBipartite(ecole.observation.NodeBipartite):
    """Ecole extractor that copies standard features and transformed names."""

    def __init__(self) -> None:
        super().__init__(cache=False)

    def extract(self, model: Any, done: bool) -> MilpBipartiteState | None:
        raw = super().extract(model, done)
        if raw is None:
            return None
        variables = model.as_pyscipopt().getVars(transformed=True)
        names = tuple(str(variable.name) for variable in variables)
        return copy_node_bipartite(raw, names)
