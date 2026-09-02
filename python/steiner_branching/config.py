"""Strict, small configuration primitives shared by Steiner CLIs."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml


class StrictConfigError(ValueError):
    """Raised when a versioned config is missing fields or contains surprises."""


ConfigT = TypeVar("ConfigT")


def load_yaml_mapping(path: Path | str) -> dict[str, Any]:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise StrictConfigError(f"invalid YAML in {config_path}: {error}") from error
    if not isinstance(raw, dict):
        raise StrictConfigError(f"{config_path} must contain one YAML mapping")
    if not all(isinstance(key, str) for key in raw):
        raise StrictConfigError(f"{config_path} keys must all be strings")
    return dict(raw)


def dataclass_from_mapping(cls: type[ConfigT], raw: Mapping[str, Any]) -> ConfigT:
    definitions = {field.name: field for field in fields(cls)}
    unknown = sorted(set(raw) - set(definitions))
    if unknown:
        raise StrictConfigError(f"unknown {cls.__name__} fields: {unknown}")
    required = {
        name
        for name, field in definitions.items()
        if field.default is MISSING and field.default_factory is MISSING
    }
    missing = sorted(required - set(raw))
    if missing:
        raise StrictConfigError(f"missing {cls.__name__} fields: {missing}")
    try:
        return cls(**dict(raw))
    except (TypeError, ValueError) as error:
        raise StrictConfigError(f"invalid {cls.__name__}: {error}") from error


def load_dataclass_yaml(cls: type[ConfigT], path: Path | str) -> ConfigT:
    return dataclass_from_mapping(cls, load_yaml_mapping(path))


@dataclass(frozen=True)
class ScaffoldConfig:
    schema_version: int
    stage: str
    seed: int
    artifact_root: str
    run_id: str
    log_level: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("ScaffoldConfig schema_version must be 1")
        if self.stage != "S01":
            raise ValueError("scaffold config is only valid for S01")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not str(self.artifact_root).strip():
            raise ValueError("artifact_root must not be empty")
        from .runtime import validate_identifier

        validate_identifier(self.run_id, label="run_id")
        if self.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("log_level must be DEBUG, INFO, WARNING, or ERROR")

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ScaffoldConfig":
        return load_dataclass_yaml(cls, path)
