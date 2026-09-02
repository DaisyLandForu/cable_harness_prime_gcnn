from dataclasses import replace

from steiner_branching.data.canonical import (
    RawEdge,
    RawSteinerInstance,
    canonicalize_raw,
)
from steiner_branching.milp.validate import brute_force_optimum


def raw_instance(edges, *, labels=(10, 20, 40)):
    return RawSteinerInstance(
        name="determinism",
        node_ids=labels,
        edges=tuple(edges),
        terminals=(labels[0], labels[-1]),
        source="unit-test",
        source_sha256="a" * 64,
    )


def test_edge_order_does_not_change_hash_or_mapping():
    edges = [RawEdge(10, 20, 1), RawEdge(20, 40, 2), RawEdge(10, 40, 9)]
    left = canonicalize_raw(raw_instance(edges))
    right = canonicalize_raw(raw_instance(reversed(edges)))
    assert left.graph_sha256 == right.graph_sha256
    assert left.edges == right.edges


def test_node_relabeling_preserves_optimum():
    left = canonicalize_raw(
        raw_instance([RawEdge(10, 20, 1), RawEdge(20, 40, 2), RawEdge(10, 40, 9)])
    )
    right = canonicalize_raw(
        RawSteinerInstance(
            name="relabel",
            node_ids=(101, 7, 22),
            edges=(RawEdge(101, 7, 1), RawEdge(7, 22, 2), RawEdge(101, 22, 9)),
            terminals=(101, 22),
            source="unit-test",
            source_sha256="b" * 64,
        )
    )
    assert brute_force_optimum(left).objective == brute_force_optimum(right).objective == 3.0


def test_increasing_an_edge_cost_cannot_lower_optimum():
    base = canonicalize_raw(
        raw_instance([RawEdge(10, 20, 1), RawEdge(20, 40, 2), RawEdge(10, 40, 4)])
    )
    increased = canonicalize_raw(
        raw_instance([RawEdge(10, 20, 10), RawEdge(20, 40, 2), RawEdge(10, 40, 4)])
    )
    assert brute_force_optimum(increased).objective >= brute_force_optimum(base).objective
