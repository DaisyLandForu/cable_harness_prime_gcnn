"""Versioned immutable contracts used across the independent Steiner stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import re
from typing import Any, Mapping


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _require_sha256(value: str, label: str) -> None:
    if not _SHA256_RE.fullmatch(str(value)):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _require_identifier(value: str, label: str) -> None:
    if not _ID_RE.fullmatch(str(value)):
        raise ValueError(f"{label} contains unsafe characters")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, order=True)
class SteinerEdge:
    edge_id: int
    u: int
    v: int
    cost: float

    def __post_init__(self) -> None:
        if isinstance(self.edge_id, bool) or not isinstance(self.edge_id, int) or self.edge_id < 0:
            raise ValueError("edge_id must be a non-negative integer")
        if not isinstance(self.u, int) or not isinstance(self.v, int) or self.u == self.v:
            raise ValueError("edge endpoints must be distinct integers")
        if not math.isfinite(float(self.cost)) or float(self.cost) <= 0.0:
            raise ValueError("edge cost must be finite and strictly positive")


@dataclass(frozen=True)
class SteinerGraph:
    name: str
    nodes: tuple[int, ...]
    edges: tuple[SteinerEdge, ...]
    terminals: tuple[int, ...]
    root: int
    source: str
    source_sha256: str
    graph_sha256: str
    original_node_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.name, "graph name")
        _require_sha256(self.source_sha256, "source_sha256")
        _require_sha256(self.graph_sha256, "graph_sha256")
        if not str(self.source).strip():
            raise ValueError("source must not be empty")
        if self.nodes != tuple(range(len(self.nodes))) or not self.nodes:
            raise ValueError("nodes must be the non-empty canonical range 0..n-1")
        if tuple(edge.edge_id for edge in self.edges) != tuple(range(len(self.edges))):
            raise ValueError("edge IDs must be contiguous and match edge order")
        node_set = set(self.nodes)
        if any(edge.u not in node_set or edge.v not in node_set for edge in self.edges):
            raise ValueError("edge endpoint is outside the graph")
        if len(self.terminals) < 2 or tuple(sorted(set(self.terminals))) != self.terminals:
            raise ValueError("terminals must be a sorted unique tuple with at least two nodes")
        if any(terminal not in node_set for terminal in self.terminals):
            raise ValueError("terminal is outside the graph")
        if self.root != min(self.terminals):
            raise ValueError("root must be the minimum canonical terminal")
        if self.original_node_ids:
            if len(self.original_node_ids) != len(self.nodes):
                raise ValueError("original_node_ids must align with nodes")
            if len(set(self.original_node_ids)) != len(self.original_node_ids):
                raise ValueError("original_node_ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SteinerGraph":
        expected = {
            "name", "nodes", "edges", "terminals", "root", "source",
            "source_sha256", "graph_sha256", "original_node_ids",
        }
        unknown = set(raw) - expected
        missing = expected - set(raw)
        if unknown or missing:
            raise ValueError(f"SteinerGraph fields mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}")
        return cls(
            name=str(raw["name"]),
            nodes=tuple(int(node) for node in raw["nodes"]),
            edges=tuple(SteinerEdge(**edge) for edge in raw["edges"]),
            terminals=tuple(int(node) for node in raw["terminals"]),
            root=int(raw["root"]),
            source=str(raw["source"]),
            source_sha256=str(raw["source_sha256"]),
            graph_sha256=str(raw["graph_sha256"]),
            original_node_ids=tuple(str(node) for node in raw["original_node_ids"]),
        )


@dataclass(frozen=True, order=True)
class EdgeVariableMetadata:
    edge_id: int
    u: int
    v: int
    variable_name: str

    def __post_init__(self) -> None:
        if self.edge_id < 0 or self.u < 0 or self.v < 0 or self.u == self.v:
            raise ValueError("invalid edge-variable mapping")
        if self.variable_name != f"stp_x_e{self.edge_id:08d}":
            raise ValueError("edge variable name does not follow the frozen convention")


@dataclass(frozen=True)
class ProblemMetadata:
    schema_version: int
    problem: str
    formulation_id: str
    formulation_version: int
    instance_name: str
    source_sha256: str
    graph_sha256: str
    root: int
    terminals: tuple[int, ...]
    edge_variables: tuple[EdgeVariableMetadata, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.formulation_version < 1:
            raise ValueError("unsupported ProblemMetadata version")
        if self.problem != "SPG" or self.formulation_id != "rooted_mcf_v1":
            raise ValueError("ProblemMetadata must describe SPG rooted_mcf_v1")
        _require_identifier(self.instance_name, "instance_name")
        _require_sha256(self.source_sha256, "source_sha256")
        _require_sha256(self.graph_sha256, "graph_sha256")
        if not self.terminals or self.root != min(self.terminals):
            raise ValueError("metadata root/terminals violate the root contract")
        if tuple(item.edge_id for item in self.edge_variables) != tuple(range(len(self.edge_variables))):
            raise ValueError("edge-variable mappings must be contiguous and ordered")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class GraphSchema:
    schema_id: str
    version: int
    variable_features: tuple[str, ...]
    constraint_features: tuple[str, ...]
    edge_features: tuple[str, ...]
    candidate_entity_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.schema_id, "schema_id")
        if self.version < 1:
            raise ValueError("schema version must be positive")
        for label, values in (
            ("variable_features", self.variable_features),
            ("constraint_features", self.constraint_features),
            ("edge_features", self.edge_features),
            ("candidate_entity_kinds", self.candidate_entity_kinds),
        ):
            if len(values) != len(set(values)) or any(not str(value) for value in values):
                raise ValueError(f"{label} must contain unique non-empty names")

    @property
    def sha256(self) -> str:
        return content_sha256(asdict(self))


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    run_id: str
    stage: str
    created_at_utc: str
    git_commit: str
    config_sha256: str
    split_manifest_sha256: str
    profile_id: str
    solver_seed: int
    training_seed: int | None
    artifact_root: str
    status: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("RunManifest schema_version must be 1")
        _require_identifier(self.run_id, "run_id")
        if not re.fullmatch(r"S(?:0[0-9]|1[0-3])", self.stage):
            raise ValueError("stage must be S00 through S13")
        try:
            parsed = datetime.fromisoformat(self.created_at_utc.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("created_at_utc must be ISO-8601") from error
        if not self.created_at_utc.endswith("Z") or parsed.utcoffset() is None:
            raise ValueError("created_at_utc must be UTC and end with Z")
        if not re.fullmatch(r"[0-9a-f]{40}", self.git_commit):
            raise ValueError("git_commit must be a full lowercase SHA-1")
        _require_sha256(self.config_sha256, "config_sha256")
        _require_sha256(self.split_manifest_sha256, "split_manifest_sha256")
        _require_identifier(self.profile_id, "profile_id")
        if self.solver_seed < 0 or (self.training_seed is not None and self.training_seed < 0):
            raise ValueError("seeds must be non-negative")
        if not str(self.artifact_root).strip():
            raise ValueError("artifact_root must not be empty")
        if self.status not in {"planned", "running", "completed", "failed", "skipped"}:
            raise ValueError("unsupported run status")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())
