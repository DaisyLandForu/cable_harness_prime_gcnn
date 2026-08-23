"""Load the frozen project-production-v1 SCIP profile for Python/Ecole."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

FORBIDDEN_PRODUCTION_KEYS = (
    "nodeselection/dfs/stdpriority",
    "nodeselection/dfs/memsavepriority",
    "separating/maxrounds",
    "estimation/restarts/restartpolicy",
    "limits/restarts",
    "presolving/maxrestarts",
)

DEFAULT_PROFILE_RELATIVE = Path("configs/scip/project-production-v1.set")
_ENTRY_RE = re.compile(r"^([A-Za-z0-9_/#.-]+)\s*=\s*(.+?)\s*$")


def default_scip_profile_path() -> Path:
    return Path(__file__).resolve().parents[2] / DEFAULT_PROFILE_RELATIVE


def resolve_scip_profile(path: str | Path | None = None) -> Path:
    if path is None or str(path).strip() == "":
        return default_scip_profile_path()
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    from_root = Path(__file__).resolve().parents[2] / candidate
    if from_root.is_file():
        return from_root
    raise FileNotFoundError(f"SCIP profile not found: {path}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_scip_set(path: str | Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ENTRY_RE.match(line)
        if match is None:
            raise ValueError(f"invalid SCIP set line: {raw_line}")
        name = match.group(1)
        value = match.group(2).strip()
        if name in seen:
            raise ValueError(f"duplicate SCIP parameter {name}")
        seen.add(name)
        entries.append((name, value))
    return entries


def parse_scip_value(raw: str) -> Any:
    if raw in {"TRUE", "FALSE"}:
        return raw == "TRUE"
    if len(raw) >= 2 and raw[0] == raw[-1] == "'":
        if len(raw) != 3:
            raise ValueError(f"unsupported SCIP char value: {raw}")
        return raw[1]
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    try:
        if raw.startswith(("+", "-")) or raw.isdigit():
            return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(f"unsupported SCIP value: {raw}") from error


def canonicalize_profile_value(raw: str) -> str:
    parsed = parse_scip_value(raw)
    if isinstance(parsed, bool):
        return "TRUE" if parsed else "FALSE"
    if isinstance(parsed, int) and not isinstance(parsed, bool):
        return str(parsed)
    if isinstance(parsed, float) and parsed.is_integer():
        return str(int(parsed))
    if isinstance(parsed, float):
        return format(parsed, ".15g")
    if isinstance(parsed, str) and len(parsed) == 1:
        return f"'{parsed}'"
    return str(parsed)


def profile_dump(entries: list[tuple[str, str]]) -> str:
    lines = [
        f"{name} = {canonicalize_profile_value(value)}"
        for name, value in sorted(entries, key=lambda item: item[0])
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def ecole_params_from_profile(path: str | Path | None = None) -> dict[str, Any]:
    profile = resolve_scip_profile(path)
    entries = parse_scip_set(profile)
    params = {name: parse_scip_value(value) for name, value in entries}
    overlap = set(params) & set(FORBIDDEN_PRODUCTION_KEYS)
    if overlap:
        raise ValueError(
            "project-production-v1 must not contain training overrides: "
            + ", ".join(sorted(overlap))
        )
    return params


def load_production_scip_params(
    *,
    seed: int,
    time_limit: float,
    node_limit: int = -1,
    profile: str | Path | None = None,
) -> dict[str, Any]:
    params = ecole_params_from_profile(profile)
    params["randomization/randomseedshift"] = int(seed)
    params["randomization/permutationseed"] = int(seed)
    params["randomization/lpseed"] = int(seed)
    params["limits/time"] = float(time_limit)
    if int(node_limit) >= 0:
        params["limits/nodes"] = int(node_limit)
    overlap = set(params) & set(FORBIDDEN_PRODUCTION_KEYS)
    if overlap:
        raise ValueError(
            "production SCIP params leaked forbidden keys: "
            + ", ".join(sorted(overlap))
        )
    return params
