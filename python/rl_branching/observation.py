from dataclasses import dataclass
from typing import Tuple

import ecole
import numpy as np


GLOBAL_FEATURE_NAMES = (
    "depth",
    "processed_nodes",
    "total_nodes",
    "open_nodes",
    "feasible_leaves",
    "infeasible_leaves",
    "lp_iterations",
    "primal_bound",
    "primal_bound_finite",
    "dual_bound",
    "dual_bound_finite",
    "relative_gap",
    "relative_gap_finite",
    "incumbent_count",
)

EXTENDED_ROW_FEATURE_NAMES = (
    "normalized_bias",
    "objective_cosine_similarity",
    "is_tight",
    "normalized_dual_solution_value",
    "scaled_age",
    "lhs",
    "lhs_finite",
    "rhs",
    "rhs_finite",
    "lp_activity",
    "side_slack",
    "is_equality",
    "is_lhs_side",
    "is_rhs_side",
)

EDGE_FEATURE_NAMES = (
    "coefficient",
    "normalized_coefficient",
    "coefficient_sign",
)


@dataclass(frozen=True)
class BipartiteObservation:
    row_features: np.ndarray
    variable_features: np.ndarray
    edge_indices: np.ndarray
    edge_values: np.ndarray
    global_features: np.ndarray
    variable_names: Tuple[str, ...]
    extended_row_features: np.ndarray | None = None
    edge_features: np.ndarray | None = None
    row_names: Tuple[str, ...] = ()
    local_lower_bounds: np.ndarray | None = None

    def validate(self) -> None:
        if self.row_features.ndim != 2 or self.row_features.shape[1] != 5:
            raise ValueError("row features must have shape [n_rows, 5]")
        if self.variable_features.ndim != 2 or self.variable_features.shape[1] != 19:
            raise ValueError("variable features must have shape [n_variables, 19]")
        if self.edge_indices.ndim != 2 or self.edge_indices.shape[0] != 2:
            raise ValueError("edge indices must have shape [2, n_edges]")
        if self.edge_values.ndim != 1 or self.edge_values.shape[0] != self.edge_indices.shape[1]:
            raise ValueError("edge values must align with edge indices")
        if self.global_features.shape != (len(GLOBAL_FEATURE_NAMES),):
            raise ValueError("global feature vector has the wrong shape")
        if len(self.variable_names) != self.variable_features.shape[0]:
            raise ValueError("variable names must align with variable features")
        if self.extended_row_features is not None:
            if self.extended_row_features.shape != (
                self.row_features.shape[0],
                len(EXTENDED_ROW_FEATURE_NAMES),
            ):
                raise ValueError("extended row features have the wrong shape")
            if self.extended_row_features.flags.writeable:
                raise ValueError("extended row features must be immutable")
            if not np.isfinite(self.extended_row_features).all():
                raise ValueError("extended row features contain NaN or Inf")
        if self.edge_features is not None:
            if self.edge_features.shape != (
                self.edge_indices.shape[1],
                len(EDGE_FEATURE_NAMES),
            ):
                raise ValueError("extended edge features have the wrong shape")
            if self.edge_features.flags.writeable:
                raise ValueError("extended edge features must be immutable")
            if not np.isfinite(self.edge_features).all():
                raise ValueError("extended edge features contain NaN or Inf")
        if self.row_names and len(self.row_names) != self.row_features.shape[0]:
            raise ValueError("row names must align with row features")
        if self.local_lower_bounds is not None:
            if self.local_lower_bounds.shape != (self.variable_features.shape[0],):
                raise ValueError("local lower bounds must align with variables")
            if self.local_lower_bounds.flags.writeable:
                raise ValueError("local lower bounds must be immutable")
        if self.edge_indices.size:
            if self.edge_indices[0].min() < 0 or self.edge_indices[0].max() >= self.row_features.shape[0]:
                raise ValueError("constraint edge index is out of range")
            if self.edge_indices[1].min() < 0 or self.edge_indices[1].max() >= self.variable_features.shape[0]:
                raise ValueError("variable edge index is out of range")
        for array in (
            self.row_features,
            self.variable_features,
            self.edge_indices,
            self.edge_values,
            self.global_features,
        ):
            if array.flags.writeable:
                raise ValueError("observation arrays must be immutable copies")
        if not np.isfinite(self.row_features).all() or not np.isfinite(self.variable_features).all():
            raise ValueError("node features contain NaN or Inf")
        if not np.isfinite(self.edge_values).all() or not np.isfinite(self.global_features).all():
            raise ValueError("edge or global features contain NaN or Inf")


def _finite_value(value: float) -> tuple[float, float]:
    value = float(value)
    if np.isfinite(value) and abs(value) < 1e20:
        return value, 1.0
    return 0.0, 0.0


def _immutable_copy(values, dtype) -> np.ndarray:
    copied = np.array(values, dtype=dtype, copy=True)
    if np.issubdtype(copied.dtype, np.floating):
        np.nan_to_num(copied, copy=False, nan=0.0, posinf=1e20, neginf=-1e20)
    copied.setflags(write=False)
    return copied


class CopiedNodeBipartite(ecole.observation.NodeBipartite):
    def __init__(self, cache: bool = False):
        super().__init__(cache=False)
        self._variable_names: Tuple[str, ...] = ()
        self._variable_cache_key: tuple | None = None
        self._edge_indices: np.ndarray | None = None
        self._edge_features: np.ndarray | None = None
        self._row_norms: np.ndarray | None = None
        self._row_lhs: np.ndarray | None = None
        self._row_rhs: np.ndarray | None = None
        self._row_lhs_finite: np.ndarray | None = None
        self._row_rhs_finite: np.ndarray | None = None
        self._row_side_sign: np.ndarray | None = None
        self._row_equality: np.ndarray | None = None
        self._row_names: Tuple[str, ...] = ()

    def before_reset(self, model) -> None:
        super().before_reset(model)
        self._variable_names = ()
        self._variable_cache_key = None
        self._edge_indices = None
        self._edge_features = None
        self._row_norms = None
        self._row_lhs = None
        self._row_rhs = None
        self._row_lhs_finite = None
        self._row_rhs_finite = None
        self._row_side_sign = None
        self._row_equality = None
        self._row_names = ()

    def _build_static_graph_metadata(self, pyscip_model, raw) -> None:
        variable_index = {
            str(variable.name): index
            for index, variable in enumerate(pyscip_model.getVars(transformed=True))
        }
        row_indices: list[int] = []
        variable_indices: list[int] = []
        edge_features: list[tuple[float, float, float]] = []
        row_norms: list[float] = []
        row_lhs: list[float] = []
        row_rhs: list[float] = []
        row_lhs_finite: list[float] = []
        row_rhs_finite: list[float] = []
        row_side_sign: list[float] = []
        row_equality: list[float] = []
        row_names: list[str] = []

        output_row = 0
        for row in pyscip_model.getLPRowsData():
            constant = float(row.getConstant())
            original_lhs = float(row.getLhs())
            original_rhs = float(row.getRhs())
            _, lhs_finite = _finite_value(original_lhs)
            _, rhs_finite = _finite_value(original_rhs)
            lhs, _ = _finite_value(original_lhs - constant)
            rhs, _ = _finite_value(original_rhs - constant)
            norm = max(float(row.getNorm()), 1.0e-20)
            equality = float(
                lhs_finite == 1.0
                and rhs_finite == 1.0
                and np.isclose(lhs, rhs, rtol=1.0e-9, atol=1.0e-9)
            )
            columns = row.getCols()
            coefficients = row.getVals()
            for side_sign, side_finite in ((-1.0, lhs_finite), (1.0, rhs_finite)):
                if side_finite == 0.0:
                    continue
                for column, coefficient in zip(columns, coefficients):
                    coefficient = side_sign * float(coefficient)
                    variable_name = str(column.getVar().name)
                    if variable_name not in variable_index:
                        raise RuntimeError(f"LP row references unknown variable {variable_name}")
                    row_indices.append(output_row)
                    variable_indices.append(variable_index[variable_name])
                    edge_features.append(
                        (coefficient, coefficient / norm, float(np.sign(coefficient)))
                    )
                row_norms.append(norm)
                row_lhs.append(lhs)
                row_rhs.append(rhs)
                row_lhs_finite.append(lhs_finite)
                row_rhs_finite.append(rhs_finite)
                row_side_sign.append(side_sign)
                row_equality.append(equality)
                row_names.append(str(row.name))
                output_row += 1

        reconstructed_indices = np.asarray(
            (row_indices, variable_indices), dtype=np.int64
        )
        expected_indices = np.asarray(raw.edge_features.indices, dtype=np.int64)
        if not np.array_equal(reconstructed_indices, expected_indices):
            raise RuntimeError("PySCIPOpt row order does not match Ecole edge order")
        features = np.asarray(edge_features, dtype=np.float32)
        expected_normalized = np.asarray(raw.edge_features.values, dtype=np.float32)
        if not np.allclose(features[:, 1], expected_normalized, rtol=1.0e-5, atol=1.0e-7):
            raise RuntimeError("reconstructed normalized edge coefficients disagree with Ecole")
        if output_row != raw.row_features.shape[0]:
            raise RuntimeError("PySCIPOpt row expansion does not match Ecole row count")

        self._edge_indices = _immutable_copy(reconstructed_indices, np.int64)
        self._edge_features = _immutable_copy(features, np.float32)
        self._row_norms = _immutable_copy(row_norms, np.float32)
        self._row_lhs = _immutable_copy(row_lhs, np.float32)
        self._row_rhs = _immutable_copy(row_rhs, np.float32)
        self._row_lhs_finite = _immutable_copy(row_lhs_finite, np.float32)
        self._row_rhs_finite = _immutable_copy(row_rhs_finite, np.float32)
        self._row_side_sign = _immutable_copy(row_side_sign, np.float32)
        self._row_equality = _immutable_copy(row_equality, np.float32)
        self._row_names = tuple(row_names)

    def _extended_rows(self, raw) -> np.ndarray:
        normalized_edges = self._edge_features[:, 1]
        row_indices = self._edge_indices[0]
        lp_solution = np.asarray(raw.variable_features[:, 8], dtype=np.float32)
        normalized_activity = np.bincount(
            row_indices,
            weights=normalized_edges * lp_solution[self._edge_indices[1]],
            minlength=raw.row_features.shape[0],
        ).astype(np.float32)
        activity = normalized_activity * self._row_norms * self._row_side_sign
        normalized_slack = np.asarray(raw.row_features[:, 0], dtype=np.float32) - normalized_activity
        slack = normalized_slack * self._row_norms
        is_lhs = (self._row_side_sign < 0.0).astype(np.float32)
        is_rhs = (self._row_side_sign > 0.0).astype(np.float32)
        return _immutable_copy(
            np.column_stack(
                (
                    raw.row_features,
                    self._row_lhs,
                    self._row_lhs_finite,
                    self._row_rhs,
                    self._row_rhs_finite,
                    activity,
                    slack,
                    self._row_equality,
                    is_lhs,
                    is_rhs,
                )
            ),
            np.float32,
        )

    def extract(self, model, done):
        raw = super().extract(model, done)
        if raw is None:
            return None

        pyscip_model = model.as_pyscipopt()
        variables = pyscip_model.getVars(transformed=True)
        names = tuple(str(variable.name) for variable in variables)
        stage = str(pyscip_model.getStage())
        n_runs = int(getattr(pyscip_model, "getNRuns", lambda: 0)())
        cache_key = (stage, n_runs, len(names), names)
        if self._variable_cache_key != cache_key:
            self._variable_names = names
            self._variable_cache_key = cache_key
        elif names != self._variable_names:
            raise RuntimeError("transformed variable identity changed without cache invalidation")

        variable_features = _immutable_copy(raw.variable_features, np.float32)
        if len(self._variable_names) != variable_features.shape[0]:
            raise RuntimeError("Ecole/PySCIPOpt variable order has inconsistent length")
        local_lower_bounds = _immutable_copy(
            [float(variable.getLbLocal()) for variable in variables],
            np.float32,
        )

        primal_bound, primal_finite = _finite_value(pyscip_model.getPrimalbound())
        dual_bound, dual_finite = _finite_value(pyscip_model.getDualbound())
        relative_gap, gap_finite = _finite_value(pyscip_model.getGap())
        global_features = _immutable_copy(
            (
                pyscip_model.getDepth(),
                pyscip_model.getNNodes(),
                pyscip_model.getNTotalNodes(),
                pyscip_model.getNLeaves(),
                pyscip_model.getNFeasibleLeaves(),
                pyscip_model.getNInfeasibleLeaves(),
                pyscip_model.getNLPIterations(),
                primal_bound,
                primal_finite,
                dual_bound,
                dual_finite,
                relative_gap,
                gap_finite,
                pyscip_model.getNSols(),
            ),
            np.float64,
        )

        # Cuts/restarts can keep the same edge_indices object while changing LP rows.
        self._build_static_graph_metadata(pyscip_model, raw)

        observation = BipartiteObservation(
            row_features=_immutable_copy(raw.row_features, np.float32),
            variable_features=variable_features,
            edge_indices=_immutable_copy(raw.edge_features.indices, np.int64),
            edge_values=_immutable_copy(raw.edge_features.values, np.float32),
            global_features=global_features,
            variable_names=self._variable_names,
            extended_row_features=self._extended_rows(raw),
            edge_features=self._edge_features,
            row_names=self._row_names,
            local_lower_bounds=local_lower_bounds,
        )
        observation.validate()
        return observation
