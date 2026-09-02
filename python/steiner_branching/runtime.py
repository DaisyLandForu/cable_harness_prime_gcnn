"""Deterministic runtime, logging, and artifact path helpers."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import random
import re
import sys
import time
from typing import TextIO


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MANAGED_HANDLER = "_steiner_branching_managed"


def validate_identifier(value: str, *, label: str) -> str:
    text = str(value)
    if not _SAFE_ID.fullmatch(text) or text in {".", ".."}:
        raise ValueError(f"{label} must use only safe identifier characters")
    return text


@dataclass(frozen=True)
class ArtifactLayout:
    root: Path
    stage: str
    run_id: str

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser()
        if str(root).strip() in {"", "."}:
            raise ValueError("artifact root must be an explicit directory")
        stage = validate_identifier(self.stage, label="stage")
        if not re.fullmatch(r"S(?:0[0-9]|1[0-3])", stage):
            raise ValueError("artifact stage must be S00 through S13")
        validate_identifier(self.run_id, label="run_id")
        object.__setattr__(self, "root", root)

    @property
    def run_dir(self) -> Path:
        candidate = self.root / self.stage.lower() / self.run_id
        root_resolved = self.root.resolve(strict=False)
        candidate_resolved = candidate.resolve(strict=False)
        if candidate_resolved != root_resolved and root_resolved not in candidate_resolved.parents:
            raise ValueError("artifact path escapes its root")
        return candidate

    def file(self, kind: str, *, suffix: str) -> Path:
        safe_kind = validate_identifier(kind, label="artifact kind")
        if not suffix.startswith(".") or "/" in suffix or "\\" in suffix:
            raise ValueError("artifact suffix must be a simple dot suffix")
        return self.run_dir / f"{safe_kind}{suffix}"

    def create(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir


class _UTCFormatter(logging.Formatter):
    converter = time.gmtime


def configure_logging(
    *, level: str = "INFO", stream: TextIO | None = None
) -> logging.Logger:
    logger = logging.getLogger("steiner_branching")
    numeric_level = getattr(logging, str(level).upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"invalid log level: {level}")
    for handler in tuple(logger.handlers):
        if getattr(handler, _MANAGED_HANDLER, False):
            logger.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    setattr(handler, _MANAGED_HANDLER, True)
    handler.setFormatter(
        _UTCFormatter(
            fmt="%(asctime)sZ %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(numeric_level)
    logger.propagate = False
    return logger


def seed_everything(seed: int) -> dict[str, bool | int]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    seeded: dict[str, bool | int] = {"seed": seed, "python": True, "numpy": False, "torch": False}
    try:
        import numpy as np

        np.random.seed(seed)
        seeded["numpy"] = True
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        seeded["torch"] = True
    except ImportError:
        pass
    return seeded
