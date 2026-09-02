"""Canonical graph construction and content hashing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

from ..contracts import SteinerEdge, SteinerGraph, content_sha256


class SteinerDataError(ValueError):
    """Raised for invalid or unsupported Steiner instance data."""


@dataclass(frozen=True)
class RawEdge:
    u: int
    v: int
    cost: float


@dataclass(frozen=True)
class RawSteinerInstance:
    name: str
    node_ids: tuple[int, ...]
    edges: tuple[RawEdge, ...]
    terminals: tuple[int, ...]
    source: str
    source_sha256: str


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_text(content: str) -> str:
    return sha256_bytes(content.encode("utf-8"))


def _connected(nodes: tuple[int, ...], edges: Iterable[SteinerEdge]) -> bool:
    adjacency = {node: set() for node in nodes}
    for edge in edges:
        adjacency[edge.u].add(edge.v)
        adjacency[edge.v].add(edge.u)
    reached = {nodes[0]}
    frontier = [nodes[0]]
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node]:
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    return len(reached) == len(nodes)


def canonicalize_raw(raw: RawSteinerInstance) -> SteinerGraph:
    if not raw.node_ids or len(set(raw.node_ids)) != len(raw.node_ids):
        raise SteinerDataError("node IDs must be non-empty and unique")
    original_ids = tuple(sorted(raw.node_ids))
    node_map = {original: canonical for canonical, original in enumerate(original_ids)}
    if len(raw.terminals) < 2 or len(set(raw.terminals)) != len(raw.terminals):
        raise SteinerDataError("terminals must contain at least two unique nodes")
    if any(terminal not in node_map for terminal in raw.terminals):
        raise SteinerDataError("terminal references an unknown node")
    sortable_edges: list[tuple[int, int, float]] = []
    for raw_edge in raw.edges:
        if raw_edge.u not in node_map or raw_edge.v not in node_map:
            raise SteinerDataError("edge references an unknown node")
        u = node_map[raw_edge.u]
        v = node_map[raw_edge.v]
        if u == v:
            raise SteinerDataError("self-loops are not supported")
        cost = float(raw_edge.cost)
        if not math.isfinite(cost) or cost <= 0.0:
            raise SteinerDataError("edge costs must be finite and strictly positive")
        sortable_edges.append((min(u, v), max(u, v), cost))
    if not sortable_edges:
        raise SteinerDataError("graph must contain at least one edge")
    sortable_edges.sort()
    edges = tuple(
        SteinerEdge(edge_id=edge_id, u=u, v=v, cost=cost)
        for edge_id, (u, v, cost) in enumerate(sortable_edges)
    )
    nodes = tuple(range(len(original_ids)))
    if not _connected(nodes, edges):
        raise SteinerDataError("input graph is disconnected")
    terminals = tuple(sorted(node_map[terminal] for terminal in raw.terminals))
    root = min(terminals)
    graph_core = {
        "nodes": nodes,
        "edges": tuple((edge.edge_id, edge.u, edge.v, edge.cost) for edge in edges),
        "terminals": terminals,
        "root": root,
    }
    return SteinerGraph(
        name=raw.name,
        nodes=nodes,
        edges=edges,
        terminals=terminals,
        root=root,
        source=raw.source,
        source_sha256=raw.source_sha256,
        graph_sha256=content_sha256(graph_core),
        original_node_ids=tuple(str(node) for node in original_ids),
    )
