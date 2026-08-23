"""Prim-style neighborhood scores/features for aviation harness branching."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

import numpy as np


_Z_RE = re.compile(r"^(?:t_)*z_(\d+)_(\d+)_(\d+)$")
_M_RE = re.compile(r"^(?:t_)*m_(\d+)_(\d+)$")
_Y_RE = re.compile(r"^(?:t_)*y_(\d+)_(\d+)$")

# Official DSU-Prime features appended after ECOLE's 19 variable features.
PRIM_VARIABLE_FEATURE_NAMES = (
    "prim_frontier",
    "prim_merge",
    "prim_cycle",
    "prim_unseen",
    "prim_src_component_ratio",
    "prim_dst_component_ratio",
)
DSU_VARIABLE_FEATURE_NAMES = PRIM_VARIABLE_FEATURE_NAMES


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


class LayerDSU:
    """Union-find over one layer's nodes that are already fixed to 1."""

    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}
        self.size: Dict[int, int] = {}

    def add(self, node: int) -> None:
        if node not in self.parent:
            self.parent[node] = node
            self.size[node] = 1

    def find(self, node: int) -> int:
        if node not in self.parent:
            raise KeyError(node)
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:
            nxt = self.parent[node]
            self.parent[node] = root
            node = nxt
        return root

    def union(self, left: int, right: int) -> None:
        self.add(left)
        self.add(right)
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]

    def contains(self, node: int) -> bool:
        return node in self.parent

    def component_size(self, node: int) -> int:
        return 0 if node not in self.parent else self.size[self.find(node)]

    def grown_node_count(self) -> int:
        return len(self.parent)


def build_dsu_layers(
    variable_names: Sequence[str],
    local_lower_bounds: Sequence[float],
    *,
    active_threshold: float = 0.5,
) -> Dict[int, LayerDSU]:
    if len(variable_names) != len(local_lower_bounds):
        raise ValueError("variable_names and local_lower_bounds must align")
    layers: Dict[int, LayerDSU] = {}
    for name, lower_bound in zip(variable_names, local_lower_bounds):
        parsed = parse_z(name)
        if parsed is None or float(lower_bound) <= active_threshold:
            continue
        layers.setdefault(parsed.prime, LayerDSU()).union(parsed.src, parsed.dst)
    return layers


def build_grown_sets(
    variable_names: Sequence[str],
    solution_values: Sequence[float],
    *,
    active_threshold: float = 0.5,
    lower_bounds: Optional[Sequence[float]] = None,
) -> Dict[int, Set[int]]:
    """Return DSU node sets. Official path uses local lower bounds only."""
    del solution_values
    if lower_bounds is None:
        return {}
    return {
        prime: set(dsu.parent)
        for prime, dsu in build_dsu_layers(
            variable_names, lower_bounds, active_threshold=active_threshold
        ).items()
    }


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


def dsu_feature_vector_for_name(
    name: str,
    layers: Dict[int, LayerDSU],
) -> Tuple[float, float, float, float, float, float]:
    z_var = parse_z(name)
    if z_var is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    dsu = layers.get(z_var.prime)
    if dsu is None or dsu.grown_node_count() == 0:
        return (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    src_in = dsu.contains(z_var.src)
    dst_in = dsu.contains(z_var.dst)
    grown = float(dsu.grown_node_count())
    src_ratio = dsu.component_size(z_var.src) / grown if src_in else 0.0
    dst_ratio = dsu.component_size(z_var.dst) / grown if dst_in else 0.0
    if src_in and dst_in:
        if dsu.find(z_var.src) == dsu.find(z_var.dst):
            return (0.0, 0.0, 1.0, 0.0, src_ratio, dst_ratio)
        return (0.0, 1.0, 0.0, 0.0, src_ratio, dst_ratio)
    if src_in ^ dst_in:
        return (1.0, 0.0, 0.0, 0.0, src_ratio, dst_ratio)
    return (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def prim_feature_vector_for_name(
    name: str,
    grown: Dict[int, Set[int]],
) -> Tuple[float, float, float, float, float, float]:
    """Compatibility wrapper around set-only grown maps (no component sizes)."""
    layers = {prime: LayerDSU() for prime in grown}
    for prime, nodes in grown.items():
        nodes = list(nodes)
        if not nodes:
            continue
        first = nodes[0]
        layers[prime].add(first)
        for node in nodes[1:]:
            layers[prime].union(first, node)
    return dsu_feature_vector_for_name(name, layers)


def prim_variable_feature_matrix(
    variable_names: Sequence[str],
    *,
    solution_values: Optional[Sequence[float]] = None,
    lower_bounds: Optional[Sequence[float]] = None,
    grown: Optional[Dict[int, Set[int]]] = None,
    active_threshold: float = 0.5,
) -> np.ndarray:
    """Return [N, 6] DSU-Prime features. Official path requires local bounds."""
    del solution_values, grown
    if lower_bounds is None:
        layers: Dict[int, LayerDSU] = {}
    else:
        layers = build_dsu_layers(
            variable_names, lower_bounds, active_threshold=active_threshold
        )
    return np.asarray(
        [dsu_feature_vector_for_name(name, layers) for name in variable_names],
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
