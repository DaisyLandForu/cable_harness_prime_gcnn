"""Frozen synthetic split assignment."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any

import yaml


DEFAULT_POLICY = Path("configs/steiner/splits/split_policy_v1.yml")


def load_seed_ranges(path: Path | str = DEFAULT_POLICY) -> dict[str, tuple[int, int]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("unsupported split policy")
    ranges_raw = raw.get("synthetic_seed_ranges")
    if not isinstance(ranges_raw, Mapping):
        raise ValueError("split policy is missing synthetic_seed_ranges")
    ranges: dict[str, tuple[int, int]] = {}
    for name, values in ranges_raw.items():
        if not isinstance(values, Mapping):
            raise ValueError(f"invalid range for {name}")
        ranges[str(name)] = (
            int(values["start_inclusive"]),
            int(values["end_inclusive"]),
        )
    return ranges


def split_for_synthetic_seed(
    seed: int, *, policy_path: Path | str = DEFAULT_POLICY
) -> str:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    matches = [
        name
        for name, (start, end) in load_seed_ranges(policy_path).items()
        if start <= seed <= end
    ]
    if len(matches) != 1:
        raise ValueError(f"seed {seed} belongs to {len(matches)} frozen split ranges")
    return matches[0]
