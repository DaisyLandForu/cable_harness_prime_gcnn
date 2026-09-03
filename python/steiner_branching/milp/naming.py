"""Frozen, collision-free MCF variable naming."""

from __future__ import annotations

import re


_X_RE = re.compile(r"^stp_x_e([0-9]{8})$")


def edge_variable_name(edge_id: int) -> str:
    if edge_id < 0 or edge_id > 99_999_999:
        raise ValueError("edge_id is outside the 8-digit naming range")
    return f"stp_x_e{edge_id:08d}"


def flow_variable_name(terminal: int, arc_id: int) -> str:
    if terminal < 0 or terminal > 9999:
        raise ValueError("terminal is outside the 4-digit naming range")
    if arc_id < 0 or arc_id > 99_999_999:
        raise ValueError("arc_id is outside the 8-digit naming range")
    return f"stp_f_t{terminal:04d}_a{arc_id:08d}"


def original_variable_name(name: str) -> str:
    """Return the original SCIP name after deterministic transform prefixes."""
    value = str(name)
    while value.startswith("t_"):
        value = value[2:]
    return value


def edge_id_from_variable_name(name: str) -> int:
    match = _X_RE.fullmatch(str(name))
    if match is None:
        raise ValueError(f"not a canonical SPG edge variable: {name!r}")
    return int(match.group(1))


def edge_id_from_scip_variable_name(name: str) -> int:
    """Map an original or transformed SCIP edge-variable name to its edge ID."""
    return edge_id_from_variable_name(original_variable_name(name))
