"""Canonical dataset manifests with instance-level checksums."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

from ..contracts import canonical_json, content_sha256


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SPLITS = {"train", "validation_iid", "test_iid", "development_ood", "test_ood"}


@dataclass(frozen=True, order=True)
class InstanceRecord:
    instance_id: str
    base_lineage: str
    split: str
    source: str
    source_sha256: str
    graph_sha256: str
    relative_path: str

    def __post_init__(self) -> None:
        if not self.instance_id or not self.base_lineage or not self.source:
            raise ValueError("instance_id, base_lineage, and source must not be empty")
        if self.split not in _SPLITS:
            raise ValueError(f"unsupported split: {self.split}")
        for label, value in (
            ("source_sha256", self.source_sha256),
            ("graph_sha256", self.graph_sha256),
        ):
            if not _SHA256_RE.fullmatch(value):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("relative_path must be a safe relative path")


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: int
    manifest_id: str
    records: tuple[InstanceRecord, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.manifest_id:
            raise ValueError("unsupported or unnamed dataset manifest")
        ordered = tuple(sorted(self.records, key=lambda item: item.instance_id))
        if ordered != self.records:
            raise ValueError("manifest records must be sorted by instance_id")
        ids = [record.instance_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate instance_id in manifest")
        validate_no_lineage_leakage(self.records)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_json(self) -> str:
        return canonical_json(self.to_dict()) + "\n"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DatasetManifest":
        expected = {"schema_version", "manifest_id", "records"}
        if set(raw) != expected:
            raise ValueError("DatasetManifest fields do not match schema_version 1")
        return cls(
            schema_version=int(raw["schema_version"]),
            manifest_id=str(raw["manifest_id"]),
            records=tuple(InstanceRecord(**record) for record in raw["records"]),
        )


def validate_no_lineage_leakage(records: tuple[InstanceRecord, ...] | list[InstanceRecord]) -> None:
    assignments: dict[str, str] = {}
    for record in records:
        previous = assignments.setdefault(record.base_lineage, record.split)
        if previous != record.split:
            raise ValueError(
                f"base lineage {record.base_lineage!r} crosses splits: {previous}, {record.split}"
            )


def write_manifest(manifest: DatasetManifest, path: Path | str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(manifest.to_json(), encoding="utf-8")
    temporary.replace(output)
    return output


def read_manifest(path: Path | str) -> DatasetManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("manifest JSON must contain one object")
    return DatasetManifest.from_dict(raw)
