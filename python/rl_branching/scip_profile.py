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

# Live SCIP keys hashed as effective_search_params_sha256 on Python/Ecole and C++.
EFFECTIVE_SEARCH_PARAM_NAMES = (
    "branching/preferbinary",
    "estimation/restarts/restartpolicy",
    "heuristics/alns/freq",
    "heuristics/alns/priority",
    "heuristics/rens/freq",
    "heuristics/rens/priority",
    "limits/gap",
    "limits/nodes",
    "limits/restarts",
    "limits/time",
    "lp/threads",
    "nodeselection/dfs/stdpriority",
    "nodeselection/estimate/stdpriority",
    "parallel/maxnthreads",
    "parallel/minnthreads",
    "presolving/maxrestarts",
    "randomization/lpseed",
    "randomization/permutationseed",
    "randomization/randomseedshift",
    "separating/maxrounds",
)

EFFECTIVE_SEED_PARAM_NAMES = (
    "randomization/lpseed",
    "randomization/permutationseed",
    "randomization/randomseedshift",
)

PARITY_INSTANCE_RELATIVE = Path("data/instances/train/syn_medium_s101.cip")
LIFECYCLE_DRIFT_INSTANCE_RELATIVE = Path("data/instances/test/real_09.cip")

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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonicalize_live_param(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str) and len(value) == 1:
        return f"'{value}'"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return format(value, ".15g")
    if isinstance(value, bytes) and len(value) == 1:
        return f"'{value.decode('ascii')}'"
    return str(value)


def dump_effective_search_params(getter, *, include_seeds: bool = True) -> str:
    names = EFFECTIVE_SEARCH_PARAM_NAMES
    if not include_seeds:
        names = tuple(name for name in names if name not in EFFECTIVE_SEED_PARAM_NAMES)
    lines = []
    for name in sorted(names):
        value = getter(name)
        lines.append(f"{name} = {canonicalize_live_param(value)}")
    return "\n".join(lines) + ("\n" if lines else "")


def effective_search_params_sha256(getter, *, include_seeds: bool = True) -> str:
    return sha256_text(dump_effective_search_params(getter, include_seeds=include_seeds))


def assert_production_live_invariants(getter) -> None:
    if int(getter("parallel/minnthreads")) != 1:
        raise ValueError("project-production-v1 requires minnthreads=1")
    if int(getter("parallel/maxnthreads")) != 1:
        raise ValueError("project-production-v1 requires maxnthreads=1")
    if int(getter("lp/threads")) != 1:
        raise ValueError("project-production-v1 requires lp/threads=1")
    if int(getter("separating/maxrounds")) == 0:
        raise ValueError("project-production-v1 must not disable cuts")
    if int(getter("nodeselection/dfs/stdpriority")) >= int(
        getter("nodeselection/estimate/stdpriority")
    ):
        raise ValueError("project-production-v1 must keep estimate above DFS")
    if getter("branching/preferbinary") not in {True, "TRUE", 1}:
        raise ValueError("project-production-v1 requires branching/preferbinary")
    policy = getter("estimation/restarts/restartpolicy")
    policy_char = policy.decode("ascii") if isinstance(policy, bytes) else str(policy)
    if policy_char.replace("'", "") == "n":
        raise ValueError("project-production-v1 must not disable restarts")
    if int(getter("limits/restarts")) == 0:
        raise ValueError("project-production-v1 must not set limits/restarts=0")


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
