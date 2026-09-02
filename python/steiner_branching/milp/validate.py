"""Independent selected-subgraph validation and small-graph enumeration."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Iterable

from ..contracts import SteinerGraph
from .mcf import McfBuild


@dataclass(frozen=True)
class SolutionCheck:
    feasible: bool
    objective: float
    selected_edge_ids: tuple[int, ...]
    reached_terminals: tuple[int, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class ExactSolution:
    objective: float
    selected_edge_ids: tuple[int, ...]


@dataclass(frozen=True)
class ScipSolutionResult:
    status: str
    objective: float | None
    selected_edge_ids: tuple[int, ...]
    check: SolutionCheck | None


def check_selected_edges(
    graph: SteinerGraph,
    selected_edge_ids: Iterable[int],
    *,
    claimed_objective: float | None = None,
    tolerance: float = 1.0e-7,
) -> SolutionCheck:
    supplied = tuple(int(edge_id) for edge_id in selected_edge_ids)
    selected = tuple(sorted(set(supplied)))
    if len(selected) != len(supplied):
        raise ValueError("selected_edge_ids contains duplicates")
    valid_ids = set(range(len(graph.edges)))
    unknown = sorted(set(selected) - valid_ids)
    if unknown:
        raise ValueError(f"selected_edge_ids contains unknown IDs: {unknown}")
    adjacency = {node: set() for node in graph.nodes}
    for edge_id in selected:
        edge = graph.edges[edge_id]
        adjacency[edge.u].add(edge.v)
        adjacency[edge.v].add(edge.u)
    reached = {graph.root}
    frontier = [graph.root]
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node]:
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    reached_terminals = tuple(terminal for terminal in graph.terminals if terminal in reached)
    objective = math.fsum(graph.edges[edge_id].cost for edge_id in selected)
    errors: list[str] = []
    missing = sorted(set(graph.terminals) - reached)
    if missing:
        errors.append(f"disconnected terminals: {missing}")
    if claimed_objective is not None and not math.isclose(
        objective, float(claimed_objective), rel_tol=tolerance, abs_tol=tolerance
    ):
        errors.append(
            f"objective mismatch: selected={objective:.17g}, claimed={float(claimed_objective):.17g}"
        )
    return SolutionCheck(
        feasible=not errors,
        objective=objective,
        selected_edge_ids=selected,
        reached_terminals=reached_terminals,
        errors=tuple(errors),
    )


def brute_force_optimum(graph: SteinerGraph, *, max_edges: int = 24) -> ExactSolution:
    if len(graph.edges) > max_edges:
        raise ValueError(f"brute force is limited to {max_edges} edges")
    best: ExactSolution | None = None
    edge_ids = tuple(range(len(graph.edges)))
    for size in range(len(edge_ids) + 1):
        for selected in itertools.combinations(edge_ids, size):
            objective = math.fsum(graph.edges[edge_id].cost for edge_id in selected)
            if best is not None and objective > best.objective + 1.0e-12:
                continue
            check = check_selected_edges(graph, selected)
            if not check.feasible:
                continue
            candidate = ExactSolution(objective=objective, selected_edge_ids=selected)
            if best is None or (candidate.objective, candidate.selected_edge_ids) < (
                best.objective,
                best.selected_edge_ids,
            ):
                best = candidate
    if best is None:
        raise ValueError("no feasible Steiner subgraph found")
    return best


def selected_edges_from_scip(build: McfBuild, *, tolerance: float = 0.5) -> tuple[int, ...]:
    return tuple(
        edge_id
        for edge_id, variable in sorted(build.edge_variables.items())
        if float(build.model.getVal(variable)) > tolerance
    )


def solve_and_validate(build: McfBuild) -> ScipSolutionResult:
    build.model.optimize()
    status = str(build.model.getStatus()).lower()
    if status != "optimal":
        return ScipSolutionResult(status=status, objective=None, selected_edge_ids=(), check=None)
    objective = float(build.model.getObjVal())
    selected = selected_edges_from_scip(build)
    check = check_selected_edges(build.graph, selected, claimed_objective=objective)
    return ScipSolutionResult(
        status=status,
        objective=objective,
        selected_edge_ids=selected,
        check=check,
    )
