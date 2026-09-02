"""Strict parser for the classic undirected SPG subset of SteinLib STP."""

from __future__ import annotations

from pathlib import Path
import re

from .canonical import (
    RawEdge,
    RawSteinerInstance,
    SteinerDataError,
    canonicalize_raw,
    sha256_bytes,
)
from ..contracts import SteinerGraph


class UnsupportedSteinerFormat(SteinerDataError):
    """Raised when a file describes a variant outside classic undirected SPG."""


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not name:
        raise SteinerDataError("instance name is empty after canonicalization")
    return name


def _parse_classic_text(
    text: str, *, name: str, source: str, require_stp_header: bool
) -> SteinerGraph:
    current: str | None = None
    seen_sections: set[str] = set()
    declared_nodes: int | None = None
    declared_edges: int | None = None
    declared_terminals: int | None = None
    edges: list[RawEdge] = []
    terminals: list[int] = []
    saw_header = False
    saw_eof = False
    for line_number, original_line in enumerate(text.splitlines(), start=1):
        line = original_line.strip()
        if not line:
            continue
        upper = line.upper()
        if saw_eof:
            raise SteinerDataError(f"content after EOF at line {line_number}")
        if current == "COMMENT":
            if upper == "END":
                current = None
            continue
        if upper.startswith("33D32945"):
            if current is not None or saw_header or seen_sections:
                raise SteinerDataError(f"invalid STP header at line {line_number}")
            saw_header = True
            continue
        if upper.startswith("SECTION "):
            if current is not None:
                raise SteinerDataError(f"nested section at line {line_number}")
            section = line.split(maxsplit=1)[1].upper()
            if section not in {"COMMENT", "GRAPH", "TERMINALS"}:
                raise UnsupportedSteinerFormat(
                    f"unsupported section {section!r} at line {line_number}"
                )
            if section in seen_sections:
                raise SteinerDataError(f"duplicate section {section!r}")
            seen_sections.add(section)
            current = section
            continue
        if upper == "END":
            if current is None:
                raise SteinerDataError(f"END outside a section at line {line_number}")
            current = None
            continue
        if upper == "EOF":
            if current is not None:
                raise SteinerDataError("EOF encountered before END")
            saw_eof = True
            continue
        fields = line.split()
        if current == "GRAPH":
            key = fields[0].upper()
            if key == "NODES" and len(fields) == 2:
                if declared_nodes is not None:
                    raise SteinerDataError("duplicate Nodes declaration")
                try:
                    declared_nodes = int(fields[1])
                except ValueError as error:
                    raise SteinerDataError(f"invalid Nodes declaration at line {line_number}") from error
            elif key == "EDGES" and len(fields) == 2:
                if declared_edges is not None:
                    raise SteinerDataError("duplicate Edges declaration")
                try:
                    declared_edges = int(fields[1])
                except ValueError as error:
                    raise SteinerDataError(f"invalid Edges declaration at line {line_number}") from error
            elif key == "E" and len(fields) == 4:
                try:
                    edges.append(RawEdge(int(fields[1]), int(fields[2]), float(fields[3])))
                except ValueError as error:
                    raise SteinerDataError(f"invalid edge at line {line_number}") from error
            elif key in {"A", "AA", "D"}:
                raise UnsupportedSteinerFormat("directed arcs are not classic undirected SPG")
            else:
                raise UnsupportedSteinerFormat(
                    f"unsupported Graph entry {fields[0]!r} at line {line_number}"
                )
        elif current == "TERMINALS":
            key = fields[0].upper()
            if key == "TERMINALS" and len(fields) == 2:
                if declared_terminals is not None:
                    raise SteinerDataError("duplicate Terminals declaration")
                try:
                    declared_terminals = int(fields[1])
                except ValueError as error:
                    raise SteinerDataError(
                        f"invalid Terminals declaration at line {line_number}"
                    ) from error
            elif key == "T" and len(fields) == 2:
                try:
                    terminals.append(int(fields[1]))
                except ValueError as error:
                    raise SteinerDataError(f"invalid terminal at line {line_number}") from error
            elif key in {"ROOT", "TP", "TF"}:
                raise UnsupportedSteinerFormat(
                    f"terminal entry {key!r} belongs to an unsupported variant"
                )
            else:
                raise UnsupportedSteinerFormat(
                    f"unsupported Terminals entry {fields[0]!r} at line {line_number}"
                )
        else:
            raise SteinerDataError(f"content outside a section at line {line_number}")
    if current is not None:
        raise SteinerDataError(f"unterminated section {current!r}")
    if not saw_eof:
        raise SteinerDataError("missing EOF marker")
    if require_stp_header and not saw_header:
        raise SteinerDataError("missing SteinLib STP header")
    if declared_nodes is None or declared_nodes < 1:
        raise SteinerDataError("missing or invalid Nodes declaration")
    if declared_edges is None or declared_edges != len(edges):
        raise SteinerDataError(
            f"Edges declaration mismatch: declared={declared_edges}, parsed={len(edges)}"
        )
    if declared_terminals is None or declared_terminals != len(terminals):
        raise SteinerDataError(
            "Terminals declaration mismatch: "
            f"declared={declared_terminals}, parsed={len(terminals)}"
        )
    node_ids = tuple(range(1, declared_nodes + 1))
    return canonicalize_raw(
        RawSteinerInstance(
            name=_safe_name(name),
            node_ids=node_ids,
            edges=tuple(edges),
            terminals=tuple(terminals),
            source=source,
            source_sha256=sha256_bytes(text.encode("utf-8")),
        )
    )


def parse_steinlib_text(text: str, *, name: str = "steinlib-instance", source: str = "memory") -> SteinerGraph:
    return _parse_classic_text(text, name=name, source=source, require_stp_header=True)


def parse_steinlib(path: Path | str) -> SteinerGraph:
    instance_path = Path(path)
    text = instance_path.read_bytes().decode("utf-8")
    return parse_steinlib_text(text, name=instance_path.stem, source=str(instance_path))
