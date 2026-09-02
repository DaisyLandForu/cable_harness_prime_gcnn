"""Deterministic, connected synthetic SPG generators for the frozen families."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any, Mapping

from ..config import dataclass_from_mapping, load_yaml_mapping
from .canonical import RawEdge, RawSteinerInstance, canonicalize_raw, sha256_text
from ..contracts import SteinerGraph, canonical_json
from ..runtime import validate_identifier


SYNTHETIC_FAMILIES = (
    "sparse_erdos_renyi",
    "random_geometric",
    "grid_with_holes",
    "community_block",
    "bridge_bottleneck",
)


@dataclass(frozen=True)
class GeneratorConfig:
    family: str
    n_nodes: int
    n_terminals: int
    seed: int

    def __post_init__(self) -> None:
        for label, value in (
            ("n_nodes", self.n_nodes),
            ("n_terminals", self.n_terminals),
            ("seed", self.seed),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{label} must be an integer")
        if self.family not in SYNTHETIC_FAMILIES:
            raise ValueError(f"unsupported synthetic family: {self.family}")
        if self.n_nodes < 4:
            raise ValueError("n_nodes must be at least 4")
        if not 2 <= self.n_terminals <= self.n_nodes:
            raise ValueError("n_terminals must be between 2 and n_nodes")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class SyntheticInstanceConfig:
    instance_id: str
    family: str
    n_nodes: int
    n_terminals: int
    seed: int

    def __post_init__(self) -> None:
        validate_identifier(self.instance_id, label="synthetic instance_id")
        GeneratorConfig(
            family=self.family,
            n_nodes=self.n_nodes,
            n_terminals=self.n_terminals,
            seed=self.seed,
        )

    def generator_config(self) -> GeneratorConfig:
        return GeneratorConfig(
            family=self.family,
            n_nodes=self.n_nodes,
            n_terminals=self.n_terminals,
            seed=self.seed,
        )


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    schema_version: int
    dataset_id: str
    instances: tuple[SyntheticInstanceConfig, ...]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("SyntheticDatasetConfig schema_version must be 1")
        validate_identifier(self.dataset_id, label="synthetic dataset_id")
        if not self.instances:
            raise ValueError("synthetic dataset must contain at least one instance")
        ids = tuple(item.instance_id for item in self.instances)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("synthetic instances must have unique, sorted instance_id values")

    @classmethod
    def from_yaml(cls, path: Path | str) -> "SyntheticDatasetConfig":
        raw = load_yaml_mapping(path)
        entries = raw.get("instances")
        if not isinstance(entries, list) or not all(isinstance(item, Mapping) for item in entries):
            raise ValueError("instances must be a YAML list of mappings")
        converted: dict[str, Any] = dict(raw)
        converted["instances"] = tuple(
            dataclass_from_mapping(SyntheticInstanceConfig, item) for item in entries
        )
        return dataclass_from_mapping(cls, converted)


def _random_cost(rng: random.Random) -> float:
    return float(rng.randint(1, 100))


def generate_graph(config: GeneratorConfig) -> SteinerGraph:
    rng = random.Random(config.seed)
    edge_costs: dict[tuple[int, int], float] = {}

    def add(u: int, v: int, cost: float | None = None) -> None:
        if u == v:
            return
        key = (min(u, v), max(u, v))
        edge_costs.setdefault(key, _random_cost(rng) if cost is None else float(cost))

    n = config.n_nodes
    if config.family == "sparse_erdos_renyi":
        for node in range(1, n):
            add(node, rng.randrange(node))
        probability = min(0.35, 3.0 / n)
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < probability:
                    add(u, v)
    elif config.family == "random_geometric":
        points = [(rng.random(), rng.random()) for _ in range(n)]
        order = sorted(range(n), key=lambda node: points[node])
        for left, right in zip(order, order[1:]):
            distance = math.dist(points[left], points[right])
            add(left, right, max(1.0, round(1000.0 * distance, 6)))
        radius = min(0.75, 1.75 * math.sqrt(math.log(n) / n))
        for u in range(n):
            for v in range(u + 1, n):
                distance = math.dist(points[u], points[v])
                if distance <= radius:
                    add(u, v, max(1.0, round(1000.0 * distance, 6)))
    elif config.family == "grid_with_holes":
        width = math.ceil(math.sqrt(n))
        for node in range(n):
            row, column = divmod(node, width)
            right = node + 1
            down = node + width
            if right < n and right // width == row and rng.random() > 0.15:
                add(node, right)
            if down < n and rng.random() > 0.15:
                add(node, down)
        for node in range(1, n):
            add(node - 1, node)
    elif config.family == "community_block":
        split = n // 2
        for start, stop in ((0, split), (split, n)):
            for node in range(start + 1, stop):
                add(node - 1, node)
            for u in range(start, stop):
                for v in range(u + 1, stop):
                    if rng.random() < 0.35:
                        add(u, v)
        add(split - 1, split, float(rng.randint(40, 100)))
    else:
        split = n // 2
        for node in range(1, split):
            add(node - 1, node)
        for node in range(split + 1, n):
            add(node - 1, node)
        add(split - 1, split, float(rng.randint(75, 125)))
        for start, stop in ((0, split), (split, n)):
            for node in range(start, stop - 2):
                if rng.random() < 0.5:
                    add(node, node + 2)
    terminals = tuple(sorted(rng.sample(range(n), config.n_terminals)))
    config_payload = {
        "family": config.family,
        "n_nodes": config.n_nodes,
        "n_terminals": config.n_terminals,
        "seed": config.seed,
    }
    source = f"synthetic:{canonical_json(config_payload)}"
    raw = RawSteinerInstance(
        name=f"synthetic-{config.family}-s{config.seed}",
        node_ids=tuple(range(n)),
        edges=tuple(RawEdge(u, v, cost) for (u, v), cost in edge_costs.items()),
        terminals=terminals,
        source=source,
        source_sha256=sha256_text(source),
    )
    return canonicalize_raw(raw)
