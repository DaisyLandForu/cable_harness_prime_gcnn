"""Deterministic small-instance writers."""

from __future__ import annotations

from pathlib import Path

from ..contracts import SteinerGraph


def graph_to_stp(graph: SteinerGraph) -> str:
    lines = [
        "33D32945 STP File, STP Format Version 1.0",
        "SECTION Comment",
        f'Name "{graph.name}"',
        'Creator "steiner_branching"',
        "END",
        "SECTION Graph",
        f"Nodes {len(graph.nodes)}",
        f"Edges {len(graph.edges)}",
    ]
    for edge in graph.edges:
        cost = format(edge.cost, ".17g")
        lines.append(f"E {edge.u + 1} {edge.v + 1} {cost}")
    lines.extend(
        ["END", "SECTION Terminals", f"Terminals {len(graph.terminals)}"]
    )
    lines.extend(f"T {terminal + 1}" for terminal in graph.terminals)
    lines.extend(["END", "EOF"])
    return "\n".join(lines) + "\n"


def write_stp(graph: SteinerGraph, path: Path | str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(graph_to_stp(graph), encoding="utf-8")
    return output
