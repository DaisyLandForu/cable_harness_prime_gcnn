"""Prim-style neighborhood scores/features for aviation harness branching."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

import numpy as np


_Z_RE = re.compile(r"^(?:t_)*z_(\d+)_(\d+)_(\d+)$")
_M_RE = re.compile(r"^(?:t_)*m_(\d+)_(\d+)$")
_Y_RE = re.compile(r"^(?:t_)*y_(\d+)_(\d+)$")

# Phase B: appended after ECOLE's 19 variable features.
PRIM_VARIABLE_FEATURE_NAMES = (
    "prim_is_cut",
    "prim_both_in",
    "prim_both_out",
    "prim_grown_empty",
    "prim_m_on_grown",
    "prim_y_on_grown",
)


@dataclass(frozen=True)
class ParsedZ:
    src: int
    dst: int
    prime: int


@dataclass(frozen=True)
class ParsedM:
    node: int
    prime: int


@dataclass(frozen=True)
class ParsedY:
    node: int
    prime: int


def parse_z(name: str) -> Optional[ParsedZ]:
    match = _Z_RE.match(str(name))
    if not match:
        return None
    return ParsedZ(int(match.group(1)), int(match.group(2)), int(match.group(3)))


def parse_m(name: str) -> Optional[ParsedM]:
    match = _M_RE.match(str(name))
    if not match:
        return None
    return ParsedM(int(match.group(1)), int(match.group(2)))


def parse_y(name: str) -> Optional[ParsedY]:
    match = _Y_RE.match(str(name))
    if not match:
        return None
    return ParsedY(int(match.group(1)), int(match.group(2)))


def build_grown_sets(
    variable_names: Sequence[str],
    solution_values: Sequence[float],
    *,
    active_threshold: float = 0.5,
    lower_bounds: Optional[Sequence[float]] = None,
) -> Dict[int, Set[int]]:
    """Build per-prime node sets already covered by nearly-integral tree edges.

    Matches C++ semantics when lower_bounds is provided:
      active iff lb > threshold OR (lp > threshold).
    Without lower_bounds, falls back to lp-only (legacy training path).
    """
    grown: Dict[int, Set[int]] = {}
    if len(variable_names) != len(solution_values):
        raise ValueError("variable_names and solution_values must align")
    if lower_bounds is not None and len(lower_bounds) != len(variable_names):
        raise ValueError("lower_bounds must align with variable_names")
    for index, (name, value) in enumerate(zip(variable_names, solution_values)):
        parsed = parse_z(name)
        if parsed is None:
            continue
        lp_active = float(value) > active_threshold
        lb_active = (
            lower_bounds is not None and float(lower_bounds[index]) > active_threshold
        )
        if not (lp_active or lb_active):
            continue
        grown.setdefault(parsed.prime, set()).update((parsed.src, parsed.dst))
    return grown


def prim_score_for_name(
    name: str,
    grown: Dict[int, Set[int]],
    *,
    empty_s_z_prior: bool = True,
) -> float:
    """
    Structural prior:
      z cut-edge (exactly one endpoint in S): +1.0
      z both outside S: +0.25
      z both inside S (cycle risk): -0.5
      m/y on a node already in S: +0.3 / +0.15
      other: 0.0
    Empty S: treat all z as weak expanders (+0.5) if empty_s_z_prior else 0.
    """
    z_var = parse_z(name)
    if z_var is not None:
        s = grown.get(z_var.prime, set())
        if not s:
            return 0.5 if empty_s_z_prior else 0.0
        src_in = z_var.src in s
        dst_in = z_var.dst in s
        if src_in ^ dst_in:
            return 1.0
        if src_in and dst_in:
            return -0.5
        return 0.25

    m_var = parse_m(name)
    if m_var is not None:
        s = grown.get(m_var.prime, set())
        return 0.3 if m_var.node in s else 0.0

    y_var = parse_y(name)
    if y_var is not None:
        s = grown.get(y_var.prime, set())
        return 0.15 if y_var.node in s else 0.0

    return 0.0


def bias_score_for_name(
    name: str,
    grown: Dict[int, Set[int]],
    *,
    mode: str = "prim",
    depth: int = 0,
) -> float:
    """C0 bias modes: none|z|root_z|prim|topology."""
    if mode in ("", "none"):
        return 0.0
    if mode == "z" or (mode == "root_z" and depth == 0):
        return 1.0 if parse_z(name) is not None else 0.0
    if mode == "root_z":
        return 0.0
    if mode == "prim":
        return prim_score_for_name(name, grown, empty_s_z_prior=True)
    if mode == "topology":
        return prim_score_for_name(name, grown, empty_s_z_prior=False)
    raise ValueError(f"unsupported bias mode: {mode}")


def candidate_prim_scores(
    candidate_names: Sequence[str],
    grown: Dict[int, Set[int]],
) -> np.ndarray:
    return np.asarray(
        [prim_score_for_name(name, grown) for name in candidate_names],
        dtype=np.float64,
    )


def candidate_bias_scores(
    candidate_names: Sequence[str],
    grown: Dict[int, Set[int]],
    *,
    mode: str = "prim",
    depth: int = 0,
) -> np.ndarray:
    return np.asarray(
        [
            bias_score_for_name(name, grown, mode=mode, depth=depth)
            for name in candidate_names
        ],
        dtype=np.float64,
    )


def prim_feature_vector_for_name(
    name: str,
    grown: Dict[int, Set[int]],
) -> Tuple[float, float, float, float, float, float]:
    """Binary neighborhood indicators aligned with PrimScore semantics."""
    z_var = parse_z(name)
    if z_var is not None:
        s = grown.get(z_var.prime, set())
        if not s:
            return (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        src_in = z_var.src in s
        dst_in = z_var.dst in s
        if src_in ^ dst_in:
            return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if src_in and dst_in:
            return (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        return (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

    m_var = parse_m(name)
    if m_var is not None:
        s = grown.get(m_var.prime, set())
        return (0.0, 0.0, 0.0, 0.0, 1.0 if m_var.node in s else 0.0, 0.0)

    y_var = parse_y(name)
    if y_var is not None:
        s = grown.get(y_var.prime, set())
        return (0.0, 0.0, 0.0, 0.0, 0.0, 1.0 if y_var.node in s else 0.0)

    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def prim_variable_feature_matrix(
    variable_names: Sequence[str],
    *,
    solution_values: Optional[Sequence[float]] = None,
    lower_bounds: Optional[Sequence[float]] = None,
    grown: Optional[Dict[int, Set[int]]] = None,
    active_threshold: float = 0.5,
) -> np.ndarray:
    """Return [N, 6] Prim neighborhood features for all variables."""
    if grown is None:
        if solution_values is None:
            grown = {}
        else:
            grown = build_grown_sets(
                variable_names,
                solution_values,
                active_threshold=active_threshold,
                lower_bounds=lower_bounds,
            )
    return np.asarray(
        [prim_feature_vector_for_name(name, grown) for name in variable_names],
        dtype=np.float32,
    )


def apply_prim_bias(
    q_values: np.ndarray,
    candidate_names: Sequence[str],
    *,
    variable_names: Optional[Sequence[str]] = None,
    solution_values: Optional[Sequence[float]] = None,
    lambda_prim: float = 0.0,
    grown: Optional[Dict[int, Set[int]]] = None,
    depth: int = 0,
    prim_min_depth: int = 0,
    prim_require_grown: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (biased_scores, prim_scores).

    When gating disables the bias, returns unmodified Q and zero PrimScores.
    """
    q = np.asarray(q_values, dtype=np.float64)
    zeros = np.zeros_like(q)
    if lambda_prim == 0.0 or depth < prim_min_depth:
        return q.copy(), zeros
    if grown is None:
        if variable_names is None or solution_values is None:
            grown = {}
        else:
            grown = build_grown_sets(variable_names, solution_values)
    if prim_require_grown and not any(grown.values()):
        return q.copy(), zeros
    prim = candidate_prim_scores(candidate_names, grown)
    if prim.shape != q.shape:
        raise ValueError("prim scores must align with Q values")
    return q + float(lambda_prim) * prim, prim


def stable_argmax_with_scores(
    scores: np.ndarray,
    candidate_names: Sequence[str],
    actions: Sequence[int],
    *,
    tolerance: float = 1.0e-7,
) -> int:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite non-empty vector")
    best = float(values.max())
    tied = np.flatnonzero(values >= best - tolerance)
    return min(
        (int(position) for position in tied),
        key=lambda position: (candidate_names[position], int(actions[position])),
    )
