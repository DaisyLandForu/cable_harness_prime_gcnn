import hashlib
from pathlib import Path

import pytest

from steiner_branching.data.canonical import SteinerDataError
from steiner_branching.data.pace import parse_pace, parse_pace_text
from steiner_branching.data.steinlib import (
    UnsupportedSteinerFormat,
    parse_steinlib,
    parse_steinlib_text,
)


FIXTURES = Path("tests/steiner/fixtures")


def test_pace_and_steinlib_parse_to_the_same_canonical_graph():
    pace = parse_pace(FIXTURES / "path.gr")
    stp = parse_steinlib(FIXTURES / "path.stp")
    assert pace.nodes == stp.nodes == (0, 1, 2, 3)
    assert pace.terminals == stp.terminals == (0, 3)
    assert pace.root == stp.root == 0
    assert pace.graph_sha256 == stp.graph_sha256
    assert [edge.cost for edge in pace.edges] == [1.0, 2.0, 3.0]


def test_parallel_edges_receive_distinct_deterministic_ids():
    graph = parse_steinlib(FIXTURES / "parallel.stp")
    parallel = [edge for edge in graph.edges if (edge.u, edge.v) == (0, 1)]
    assert [(edge.edge_id, edge.cost) for edge in parallel] == [(0, 1.0), (1, 5.0)]
    assert len({edge.edge_id for edge in graph.edges}) == len(graph.edges)


@pytest.mark.parametrize(
    ("text", "error"),
    [
        (
            "SECTION Graph\nNodes 2\nEdges 1\nA 1 2 1\nEND\n"
            "SECTION Terminals\nTerminals 2\nT 1\nT 2\nEND\nEOF\n",
            UnsupportedSteinerFormat,
        ),
        (
            "SECTION Graph\nNodes 2\nEdges 1\nE 1 2 0\nEND\n"
            "SECTION Terminals\nTerminals 2\nT 1\nT 2\nEND\nEOF\n",
            SteinerDataError,
        ),
        (
            "SECTION Graph\nNodes 3\nEdges 1\nE 1 2 1\nEND\n"
            "SECTION Terminals\nTerminals 2\nT 1\nT 2\nEND\nEOF\n",
            SteinerDataError,
        ),
        (
            "SECTION Graph\nNodes 2\nEdges 2\nE 1 2 1\nEND\n"
            "SECTION Terminals\nTerminals 2\nT 1\nT 2\nEND\nEOF\n",
            SteinerDataError,
        ),
        (
            "SECTION Graph\nNodes 2\nEdges 1\nE 1 2 1\nEND\n"
            "SECTION Prizes\nP 1 1\nEND\n"
            "SECTION Terminals\nTerminals 2\nT 1\nT 2\nEND\nEOF\n",
            UnsupportedSteinerFormat,
        ),
        (
            "SECTION Graph\nNodes 2\nEdges 1\nE 1 2 1\nEND\n"
            "SECTION Terminals\nTerminals 2\nT 1\nT 2\nEND\n",
            SteinerDataError,
        ),
        (
            "SECTION Graph\nNodes 2\nNodes 2\nEdges 1\nE 1 2 1\nEND\n"
            "SECTION Terminals\nTerminals 2\nT 1\nT 2\nEND\nEOF\n",
            SteinerDataError,
        ),
        (
            "SECTION Graph\nNodes 2\nEdges 1\nE 1 2 1\nEND\n"
            "SECTION Terminals\nTerminals 2\nT 1\nT 2\nEND\nEOF\n"
            "SECTION Comment\nEND\n",
            SteinerDataError,
        ),
    ],
)
def test_invalid_or_unsupported_pace_inputs_fail_loudly(text, error):
    with pytest.raises(error):
        parse_pace_text(text)


def test_steinlib_requires_header_and_rejects_rooted_terminal_entry():
    pace_text = (FIXTURES / "path.gr").read_text(encoding="utf-8")
    with pytest.raises(SteinerDataError, match="header"):
        parse_steinlib_text(pace_text)
    rooted = (FIXTURES / "path.stp").read_text(encoding="utf-8").replace(
        "T 1\n", "Root 1\n", 1
    )
    with pytest.raises(UnsupportedSteinerFormat, match="variant"):
        parse_steinlib_text(rooted)


def test_path_parser_source_hash_preserves_exact_crlf_bytes(tmp_path: Path):
    raw = (FIXTURES / "path.stp").read_bytes().replace(b"\n", b"\r\n")
    instance = tmp_path / "crlf.stp"
    instance.write_bytes(raw)
    graph = parse_steinlib(instance)
    assert graph.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert graph.graph_sha256 == parse_steinlib(FIXTURES / "path.stp").graph_sha256
