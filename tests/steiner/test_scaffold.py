from __future__ import annotations

from dataclasses import replace
import io
import logging
from pathlib import Path
import random

import numpy as np
import pytest
import yaml

import steiner_branching
from steiner_branching.config import ScaffoldConfig, StrictConfigError
from steiner_branching.contracts import (
    EdgeVariableMetadata,
    GraphSchema,
    ProblemMetadata,
    RunManifest,
    SteinerEdge,
    SteinerGraph,
)
from steiner_branching.runtime import ArtifactLayout, configure_logging, seed_everything


SHA_A = "a" * 64
SHA_B = "b" * 64


def example_graph() -> SteinerGraph:
    return SteinerGraph(
        name="toy-path",
        nodes=(0, 1, 2),
        edges=(SteinerEdge(0, 0, 1, 1.0), SteinerEdge(1, 1, 2, 2.0)),
        terminals=(0, 2),
        root=0,
        source="unit-test",
        source_sha256=SHA_A,
        graph_sha256=SHA_B,
        original_node_ids=("10", "20", "30"),
    )


def test_package_and_subpackages_import_without_solver_dependencies():
    assert steiner_branching.__version__ == "0.1.0"
    for name in ("data", "milp", "solver", "models", "learning", "evaluation"):
        module = __import__(f"steiner_branching.{name}", fromlist=[name])
        assert module.__name__.endswith(name)


def test_minimal_config_loads_strictly(tmp_path: Path):
    config = ScaffoldConfig.from_yaml("configs/steiner/scaffold_smoke.yml")
    assert config.seed == 0
    raw = yaml.safe_load(Path("configs/steiner/scaffold_smoke.yml").read_text())
    raw["unknown"] = True
    invalid = tmp_path / "unknown.yml"
    invalid.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(StrictConfigError, match="unknown"):
        ScaffoldConfig.from_yaml(invalid)
    raw.pop("unknown")
    raw.pop("seed")
    invalid.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(StrictConfigError, match="missing"):
        ScaffoldConfig.from_yaml(invalid)
    raw["seed"] = 0
    raw["schema_version"] = 2
    invalid.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(StrictConfigError, match="schema_version"):
        ScaffoldConfig.from_yaml(invalid)


def test_contracts_validate_and_round_trip():
    graph = example_graph()
    assert SteinerGraph.from_dict(graph.to_dict()) == graph
    metadata = ProblemMetadata(
        schema_version=1,
        problem="SPG",
        formulation_id="rooted_mcf_v1",
        formulation_version=1,
        instance_name=graph.name,
        source_sha256=graph.source_sha256,
        graph_sha256=graph.graph_sha256,
        root=graph.root,
        terminals=graph.terminals,
        edge_variables=(
            EdgeVariableMetadata(0, 0, 1, "stp_x_e00000000"),
            EdgeVariableMetadata(1, 1, 2, "stp_x_e00000001"),
        ),
    )
    schema = GraphSchema(
        schema_id="milp_bipartite_v1",
        version=1,
        variable_features=("objective",),
        constraint_features=("bias",),
        edge_features=("coefficient",),
        candidate_entity_kinds=("EDGE",),
    )
    manifest = RunManifest(
        schema_version=1,
        run_id="s01-test",
        stage="S01",
        created_at_utc="2026-09-02T00:00:00Z",
        git_commit="a" * 40,
        config_sha256=SHA_A,
        split_manifest_sha256=SHA_B,
        profile_id="correctness-v1",
        solver_seed=0,
        training_seed=None,
        artifact_root="artifacts/steiner",
        status="planned",
    )
    assert len(metadata.sha256) == len(schema.sha256) == len(manifest.sha256) == 64
    with pytest.raises(ValueError, match="root"):
        replace(graph, root=2)
    with pytest.raises(ValueError, match="edge IDs"):
        replace(graph, edges=(SteinerEdge(1, 0, 1, 1.0),))
    with pytest.raises(ValueError, match="SHA-256"):
        replace(graph, source_sha256="bad")
    with pytest.raises(ValueError, match="unique"):
        replace(schema, variable_features=("x", "x"))


def test_seed_everything_is_deterministic_without_cuda():
    first_status = seed_everything(17)
    first = (random.random(), np.random.random())
    second_status = seed_everything(17)
    second = (random.random(), np.random.random())
    assert first_status == second_status
    assert first == second
    assert first_status["python"] and first_status["numpy"]


def test_artifact_layout_cannot_escape_root(tmp_path: Path):
    layout = ArtifactLayout(tmp_path / "artifacts", "S01", "safe-run")
    assert layout.file("manifest", suffix=".json") == (
        tmp_path / "artifacts/s01/safe-run/manifest.json"
    )
    assert layout.create().is_dir()
    with pytest.raises(ValueError):
        ArtifactLayout(tmp_path / "artifacts", "S01", "../escape")
    with pytest.raises(ValueError):
        layout.file("../escape", suffix=".json")
    with pytest.raises(ValueError):
        layout.file("manifest", suffix="/bad")


def test_logging_configuration_is_stable_and_idempotent():
    stream = io.StringIO()
    logger = configure_logging(level="INFO", stream=stream)
    logger = configure_logging(level="INFO", stream=stream)
    managed = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_steiner_branching_managed", False)
    ]
    assert len(managed) == 1
    logger.info("scaffold-ready")
    text = stream.getvalue()
    assert text.count("scaffold-ready") == 1
    assert "Z INFO steiner_branching scaffold-ready" in text
    assert logger.propagate is False
    for handler in tuple(logger.handlers):
        if getattr(handler, "_steiner_branching_managed", False):
            logger.removeHandler(handler)
            handler.close()
    logging.shutdown()
