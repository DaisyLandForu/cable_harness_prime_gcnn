"""Parser for PACE 2018 classic Steiner `.gr` instances."""

from __future__ import annotations

from pathlib import Path

from .steinlib import _parse_classic_text
from ..contracts import SteinerGraph


def parse_pace_text(text: str, *, name: str = "pace-instance", source: str = "memory") -> SteinerGraph:
    return _parse_classic_text(text, name=name, source=source, require_stp_header=False)


def parse_pace(path: Path | str) -> SteinerGraph:
    instance_path = Path(path)
    text = instance_path.read_bytes().decode("utf-8")
    return parse_pace_text(text, name=instance_path.stem, source=str(instance_path))
