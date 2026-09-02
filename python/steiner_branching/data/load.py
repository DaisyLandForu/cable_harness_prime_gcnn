"""Strict extension-based dispatch for supported Steiner instance formats."""

from __future__ import annotations

from pathlib import Path

from ..contracts import SteinerGraph
from .pace import parse_pace
from .steinlib import UnsupportedSteinerFormat, parse_steinlib


def load_graph(path: Path | str) -> SteinerGraph:
    instance_path = Path(path)
    suffix = instance_path.suffix.lower()
    if suffix == ".gr":
        return parse_pace(instance_path)
    if suffix == ".stp":
        return parse_steinlib(instance_path)
    raise UnsupportedSteinerFormat(
        f"unsupported instance extension {instance_path.suffix!r}; expected .gr or .stp"
    )
