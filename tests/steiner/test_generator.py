from pathlib import Path

import pytest

from steiner_branching.config import StrictConfigError
from steiner_branching.data.generate import (
    GeneratorConfig,
    SYNTHETIC_FAMILIES,
    SyntheticDatasetConfig,
    generate_graph,
)


def test_all_frozen_families_generate_valid_deterministic_graphs():
    hashes = set()
    for index, family in enumerate(SYNTHETIC_FAMILIES):
        config = GeneratorConfig(family=family, n_nodes=16, n_terminals=4, seed=100000 + index)
        first = generate_graph(config)
        second = generate_graph(config)
        assert first == second
        assert first.root == min(first.terminals)
        assert first.nodes == tuple(range(16))
        assert len(first.edges) >= 15
        hashes.add(first.graph_sha256)
    assert len(hashes) == len(SYNTHETIC_FAMILIES)


def test_frozen_synthetic_config_is_strict_and_covers_all_families(tmp_path: Path):
    config_path = Path("configs/steiner/data/synthetic_v1.yml")
    config = SyntheticDatasetConfig.from_yaml(config_path)
    assert {instance.family for instance in config.instances} == set(SYNTHETIC_FAMILIES)
    invalid = tmp_path / "invalid.yml"
    invalid.write_text(config_path.read_text(encoding="utf-8") + "unknown: true\n", encoding="utf-8")
    with pytest.raises(StrictConfigError, match="unknown"):
        SyntheticDatasetConfig.from_yaml(invalid)
