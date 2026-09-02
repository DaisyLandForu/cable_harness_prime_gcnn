from pathlib import Path

import pytest

from steiner_branching.data.steinlib import parse_steinlib
from steiner_branching.milp.validate import brute_force_optimum, check_selected_edges


FIXTURES = Path("tests/steiner/fixtures")


def test_checker_accepts_feasible_subgraph_and_rejects_disconnected_or_mismatch():
    graph = parse_steinlib(FIXTURES / "triangle.stp")
    optimum = brute_force_optimum(graph)
    assert optimum.objective == 2.0
    assert check_selected_edges(graph, optimum.selected_edge_ids, claimed_objective=2.0).feasible
    disconnected = check_selected_edges(graph, (0,))
    assert not disconnected.feasible
    assert "disconnected" in disconnected.errors[0]
    mismatch = check_selected_edges(graph, optimum.selected_edge_ids, claimed_objective=3.0)
    assert not mismatch.feasible
    assert "objective mismatch" in mismatch.errors[0]
    with pytest.raises(ValueError, match="unknown"):
        check_selected_edges(graph, (99,))
    with pytest.raises(ValueError, match="duplicates"):
        check_selected_edges(graph, (0, 0))


def test_positive_cost_cycle_is_feasible_but_not_better_than_optimum():
    graph = parse_steinlib(FIXTURES / "triangle.stp")
    cycle = check_selected_edges(graph, (0, 1, 2))
    assert cycle.feasible
    assert cycle.objective > brute_force_optimum(graph).objective
