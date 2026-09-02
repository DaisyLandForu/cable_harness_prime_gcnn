from pathlib import Path

import pytest

pytest.importorskip("pyscipopt")

from steiner_branching.data.steinlib import parse_steinlib
from steiner_branching.data.generate import GeneratorConfig, SYNTHETIC_FAMILIES, generate_graph
from steiner_branching.milp.mcf import build_mcf
from steiner_branching.milp.naming import edge_id_from_variable_name
from steiner_branching.milp.validate import brute_force_optimum, solve_and_validate


FIXTURES = Path("tests/steiner/fixtures")


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [("path.stp", 6.0), ("triangle.stp", 2.0), ("star.stp", 3.0), ("parallel.stp", 2.0), ("high_cost.stp", 4.0)],
)
def test_mcf_matches_bruteforce_and_independent_checker(fixture: str, expected: float):
    graph = parse_steinlib(FIXTURES / fixture)
    exact = brute_force_optimum(graph)
    build = build_mcf(graph)
    result = solve_and_validate(build)
    assert exact.objective == expected
    assert result.status == "optimal"
    assert result.objective == pytest.approx(expected)
    assert result.check is not None and result.check.feasible
    assert result.check.objective == pytest.approx(result.objective)


def test_mcf_variable_counts_types_names_and_metadata_are_deterministic():
    graph = parse_steinlib(FIXTURES / "star.stp")
    first = build_mcf(graph)
    second = build_mcf(graph)
    commodities = len(graph.terminals) - 1
    assert first.counts.binary_edge_variables == len(graph.edges)
    assert first.counts.continuous_flow_variables == 2 * len(graph.edges) * commodities
    assert first.counts.flow_balance_constraints == len(graph.nodes) * commodities
    assert first.counts.linking_constraints == len(graph.edges) * commodities
    assert first.model.getNVars() == (
        first.counts.binary_edge_variables + first.counts.continuous_flow_variables
    )
    assert first.model.getNConss() == (
        first.counts.flow_balance_constraints + first.counts.linking_constraints
    )
    for edge_id, variable in first.edge_variables.items():
        assert variable.vtype() == "BINARY"
        assert edge_id_from_variable_name(variable.name) == edge_id
    assert all(variable.vtype() == "CONTINUOUS" for variable in first.flow_variables.values())
    assert first.metadata == second.metadata
    assert first.metadata.sha256 == second.metadata.sha256


@pytest.mark.parametrize("family", SYNTHETIC_FAMILIES)
def test_random_small_family_mcf_matches_bruteforce(family: str):
    graph = generate_graph(
        GeneratorConfig(family=family, n_nodes=6, n_terminals=3, seed=100000)
    )
    exact = brute_force_optimum(graph)
    result = solve_and_validate(build_mcf(graph))
    assert result.status == "optimal"
    assert result.objective == pytest.approx(exact.objective)
    assert result.check is not None and result.check.feasible
