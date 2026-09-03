from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import ecole
import numpy as np
import pytest
import torch
import yaml

from steiner_branching.config import StrictConfigError
from steiner_branching.contracts import EdgeVariableMetadata, ProblemMetadata
from steiner_branching.data.canonical import (
    RawEdge,
    RawSteinerInstance,
    canonicalize_raw,
    sha256_text,
)
from steiner_branching.data.generate import GeneratorConfig, generate_graph
from steiner_branching.milp.mcf import build_mcf
from steiner_branching.milp.naming import (
    edge_id_from_scip_variable_name,
    original_variable_name,
)
from steiner_branching.models.milp_gcnn import (
    MilpBipartiteGCNN,
    load_b0_config,
    model_state_sha256,
    parameter_count,
    score_state,
)
from steiner_branching.solver.bipartite_observation import (
    MILP_BIPARTITE_V1,
    SteinerNodeBipartite,
    copy_node_bipartite,
    make_bipartite_state,
    with_legal_edge_actions,
)
from steiner_branching.solver.graph_state import candidate_exact_closure
from steiner_branching.solver.branchability import configure_p1, load_s03_config
from steiner_branching.solver.scip_identity import (
    ScipIdentityError,
    _probindex_symbol,
    scip_variable_probindex,
    transformed_variable_names_by_probindex,
    variable_names_by_probindex,
)


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/steiner/models/b0_milp_gcnn_v1.yml"
RUNNER = REPO / "scripts/steiner/run_s04_b0_snapshot.py"


def _metadata() -> ProblemMetadata:
    return ProblemMetadata(
        schema_version=1,
        problem="SPG",
        formulation_id="rooted_mcf_v1",
        formulation_version=1,
        instance_name="parallel-actions",
        source_sha256="0" * 64,
        graph_sha256="1" * 64,
        root=0,
        terminals=(0, 2),
        edge_variables=(
            EdgeVariableMetadata(0, 0, 1, "stp_x_e00000000"),
            EdgeVariableMetadata(1, 0, 1, "stp_x_e00000001"),
        ),
    )


def _full_state():
    constraints = np.arange(15, dtype=np.float32).reshape(3, 5) / 17.0
    variables = np.arange(95, dtype=np.float32).reshape(5, 19) / 101.0
    variables[:, 1:5] = 0.0
    variables[:2, 1] = 1.0
    variables[2:, 4] = 1.0
    variables[:2, 9] = (0.25, 0.4)
    variables[2:, 9] = 0.0
    edge_indices = np.asarray(
        ((0, 0, 1, 1, 2, 2), (0, 1, 0, 2, 3, 4)), dtype=np.int64
    )
    edge_features = np.asarray((1.0, -0.5, 0.75, 0.25, 2.0, -1.0), dtype=np.float32)
    return make_bipartite_state(
        constraint_features=constraints,
        variable_features=variables,
        edge_indices=edge_indices,
        edge_features=edge_features,
        variable_names=(
            "t_stp_x_e00000000",
            "t_t_stp_x_e00000001",
            "t_stp_f_t0002_a00000000",
            "aux_a",
            "aux_b",
        ),
    )


def test_schema_is_exactly_19_5_1_and_has_no_aviation_features():
    assert len(MILP_BIPARTITE_V1.variable_features) == 19
    assert len(MILP_BIPARTITE_V1.constraint_features) == 5
    assert len(MILP_BIPARTITE_V1.edge_features) == 1
    assert MILP_BIPARTITE_V1.candidate_entity_kinds == ("EDGE",)
    names = " ".join(
        MILP_BIPARTITE_V1.variable_features
        + MILP_BIPARTITE_V1.constraint_features
        + MILP_BIPARTITE_V1.edge_features
    ).lower()
    assert "prim" not in names and "aviation" not in names and "dsu" not in names


def test_state_copies_arrays_rejects_nonfinite_values_and_preserves_empty_actions():
    original = np.zeros((2, 19), dtype=np.float32)
    state = make_bipartite_state(
        constraint_features=np.zeros((1, 5), dtype=np.float32),
        variable_features=original,
        edge_indices=np.asarray(((0,), (0,)), dtype=np.int64),
        edge_features=np.asarray((1.0,), dtype=np.float32),
        variable_names=("a", "b"),
    )
    original[0, 0] = 99.0
    assert state.variable_features[0, 0] == 0.0
    assert not state.variable_features.flags.writeable
    assert state.candidate_count == 0
    empty = candidate_exact_closure(state)
    assert empty.constraint_features.shape == (0, 5)
    assert empty.variable_features.shape == (0, 19)
    assert empty.candidate_count == 0
    torch.set_num_threads(1)
    torch.manual_seed(404)
    empty_logits = score_state(MilpBipartiteGCNN(embedding_dim=8, hidden_dim=8), empty)
    assert empty_logits.shape == (0,)

    bad = np.zeros((2, 19), dtype=np.float32)
    bad[0, 3] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        make_bipartite_state(
            constraint_features=np.zeros((1, 5), dtype=np.float32),
            variable_features=bad,
            edge_indices=np.zeros((2, 0), dtype=np.int64),
            edge_features=np.zeros((0, 1), dtype=np.float32),
            variable_names=("a", "b"),
        )


def test_only_ecole_undefined_incumbent_nans_receive_the_versioned_zero_sentinel():
    raw = SimpleNamespace(
        row_features=np.zeros((1, 5), dtype=np.float32),
        variable_features=np.zeros((2, 19), dtype=np.float32),
        edge_features=SimpleNamespace(
            indices=np.asarray(((0,), (0,)), dtype=np.int64),
            values=np.asarray((1.0,), dtype=np.float32),
        ),
    )
    raw.variable_features[:, 13:15] = np.nan
    state = copy_node_bipartite(raw, ("a", "b"))
    assert state.variable_features[:, 13:15].tolist() == [[0.0, 0.0], [0.0, 0.0]]
    raw.variable_features[0, 7] = np.nan
    with pytest.raises(ValueError, match="unsupported NaN or Inf"):
        copy_node_bipartite(raw, ("a", "b"))


def test_probindex_is_canonical_identity_under_permutation_and_fails_closed():
    variables = (
        SimpleNamespace(name="t_stp_f_t0002_a00000000", probindex=2),
        SimpleNamespace(name="t_stp_x_e00000001", probindex=1),
        SimpleNamespace(name="t_stp_x_e00000000", probindex=0),
    )
    probindex_of = lambda variable: variable.probindex
    assert variable_names_by_probindex(
        variables, 3, probindex_of=probindex_of
    ) == (
        "t_stp_x_e00000000",
        "t_stp_x_e00000001",
        "t_stp_f_t0002_a00000000",
    )

    with pytest.raises(ScipIdentityError, match="count mismatch"):
        variable_names_by_probindex(variables[:2], 3, probindex_of=probindex_of)
    duplicate = (
        SimpleNamespace(name="a", probindex=0),
        SimpleNamespace(name="b", probindex=0),
    )
    with pytest.raises(ScipIdentityError, match="duplicate SCIP variable probindex"):
        variable_names_by_probindex(duplicate, 2, probindex_of=probindex_of)
    outside = (
        SimpleNamespace(name="a", probindex=0),
        SimpleNamespace(name="b", probindex=2),
    )
    with pytest.raises(ScipIdentityError, match="outside Ecole rows"):
        variable_names_by_probindex(outside, 2, probindex_of=probindex_of)


def test_probindex_bridge_rejects_any_nonfrozen_stack(monkeypatch, tmp_path):
    _probindex_symbol.cache_clear()
    monkeypatch.setenv("STEINER_SOLVER_STACK_ID", "system-scip-must-not-load")
    with pytest.raises(ScipIdentityError, match="frozen Steiner wrapper"):
        scip_variable_probindex(SimpleNamespace(ptr=lambda: 1))

    _probindex_symbol.cache_clear()
    monkeypatch.setenv("STEINER_SOLVER_STACK_ID", "scip804-ecole081-pyscipopt430")
    monkeypatch.setenv("STEINER_SCIP_VERSION", "8.0.4")
    monkeypatch.setenv("SCIPOPTDIR", str(tmp_path))
    with pytest.raises(ScipIdentityError, match="repository-frozen prefix"):
        scip_variable_probindex(SimpleNamespace(ptr=lambda: 1))
    _probindex_symbol.cache_clear()


def test_transformed_names_and_parallel_edges_map_one_to_one():
    assert original_variable_name("t_t_stp_x_e00000001") == "stp_x_e00000001"
    assert edge_id_from_scip_variable_name("t_t_stp_x_e00000001") == 1
    state = with_legal_edge_actions(_full_state(), np.asarray((0, 1)), _metadata())
    assert state.candidate_names == (
        "t_stp_x_e00000000",
        "t_t_stp_x_e00000001",
    )
    assert state.candidate_edge_ids.tolist() == [0, 1]
    with pytest.raises(ValueError, match="not a binary"):
        with_legal_edge_actions(_full_state(), np.asarray((2,)), _metadata())
    with pytest.raises(ValueError, match="duplicate"):
        with_legal_edge_actions(_full_state(), np.asarray((0, 0)), _metadata())


def test_candidate_closure_has_stable_maps_and_exact_one_round_logits():
    full = with_legal_edge_actions(_full_state(), np.asarray((0,)), _metadata())
    closure = candidate_exact_closure(full)
    assert closure.constraint_global_ids.tolist() == [0, 1]
    assert closure.variable_global_ids.tolist() == [0, 1, 2]
    assert closure.variable_names == full.variable_names[:3]
    assert closure.candidate_indices.tolist() == [0]
    assert closure.candidate_edge_ids.tolist() == [0]

    torch.set_num_threads(1)
    torch.manual_seed(404)
    model = MilpBipartiteGCNN(embedding_dim=16, hidden_dim=24).eval()
    with torch.inference_mode():
        full_logits = score_state(model, full)
        closure_logits = score_state(model, closure)
    torch.testing.assert_close(full_logits, closure_logits, rtol=0.0, atol=1.0e-6)
    assert parameter_count(model) > 0
    assert len(model_state_sha256(model)) == 64


def test_b0_config_is_strict_and_keeps_registered_train_snapshot(monkeypatch):
    config = load_b0_config(CONFIG)
    assert config["architecture"]["variable_features"] == 19
    assert config["architecture"]["constraint_features"] == 5
    assert config["architecture"]["edge_features"] == 1
    assert config["architecture"]["aggregation"] == "sum"
    assert config["model_seed"] == 404
    assert config["snapshot"]["generator_seed"] == 100300
    assert config["gate"]["max_full_closure_logit_error"] == 1.0e-5

    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    changed = deepcopy(raw)
    changed["architecture"]["variable_features"] = 25
    with pytest.raises(StrictConfigError, match="19/5/1"):
        temporary = CONFIG.with_name("does-not-exist.yml")
        # Exercise validation without leaving a repository file.
        from steiner_branching.models import milp_gcnn

        monkeypatch.setattr(milp_gcnn, "load_yaml_mapping", lambda _path: changed)
        load_b0_config(temporary)

    unexpected = deepcopy(raw)
    unexpected["architecture"]["unexpected"] = True
    monkeypatch.setattr(milp_gcnn, "load_yaml_mapping", lambda _path: unexpected)
    with pytest.raises(StrictConfigError, match="unknown"):
        load_b0_config(CONFIG)


def test_real_scip_snapshot_is_deterministic_and_passes_gate(tmp_path):
    if not __import__("os").environ.get("STEINER_SOLVER_STACK_ID"):
        pytest.skip("requires the frozen SCIP 8.0.4 wrapper")
    outputs = []
    summaries = []
    for suffix in ("a", "b"):
        snapshot = tmp_path / f"snapshot-{suffix}.json"
        summary = tmp_path / f"summary-{suffix}.json"
        process = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--config",
                str(CONFIG),
                "--snapshot-output",
                str(snapshot),
                "--summary-output",
                str(summary),
            ],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )
        assert process.returncode == 0, process.stderr + process.stdout
        outputs.append(snapshot.read_bytes())
        summaries.append(json.loads(summary.read_text(encoding="utf-8")))
    assert outputs[0] == outputs[1]
    for summary in summaries:
        assert summary["gate"]["overall_pass"]
        assert summary["observation"]["snapshot_count"] == 3
        assert summary["observation"]["mapping_rate"] == 1.0
        assert summary["observation"]["probindex_identity_states"] == 3
        assert summary["observation"]["probindex_identity_variables"] == 3 * 981
        assert summary["gate"]["checks"]["probindex_identity_complete"]
        assert summary["parity"]["max_absolute_logit_error"] <= 1.0e-5
        assert summary["parity"]["argmax_agreement"] == 1.0


def test_frozen_scip_bridge_binds_transformed_x_flow_and_parallel_names():
    if not __import__("os").environ.get("STEINER_SOLVER_STACK_ID"):
        pytest.skip("requires the frozen SCIP 8.0.4 wrapper")
    base = generate_graph(
        GeneratorConfig(
            family="sparse_erdos_renyi", n_nodes=48, n_terminals=5, seed=100300
        )
    )
    first = base.edges[0]
    source = "s04-probindex-parallel-integration"
    graph = canonicalize_raw(
        RawSteinerInstance(
            name=source,
            node_ids=base.nodes,
            edges=tuple(RawEdge(edge.u, edge.v, edge.cost) for edge in base.edges)
            + (RawEdge(first.u, first.v, first.cost + 0.5),),
            terminals=base.terminals,
            source=source,
            source_sha256=sha256_text(source),
        )
    )
    build = build_mcf(graph, configure_correctness_profile=False, hide_output=True)
    configure_p1(
        build.model,
        load_s03_config(
            REPO / "configs/steiner/experiments/s03_branchability_pilot_v1.yml"
        ),
        "relpscost",
    )
    observation_function = SteinerNodeBipartite()
    environment = ecole.environment.Branching(
        observation_function=observation_function, pseudo_candidates=False
    )
    environment.seed(0)
    observation, action_set, _reward, done, _info = environment.reset(
        ecole.scip.Model.from_pyscipopt(build.model)
    )
    assert not done and observation is not None and action_set is not None
    original_names = {original_variable_name(name) for name in observation.variable_names}
    assert {item.variable_name for item in build.metadata.edge_variables} <= original_names
    assert any(name.startswith("stp_f_") for name in original_names)
    parallel_ids = [
        edge.edge_id
        for edge in graph.edges
        if (edge.u, edge.v) == (first.u, first.v)
    ]
    assert len(parallel_ids) == 2
    assert {
        f"stp_x_e{edge_id:08d}" for edge_id in parallel_ids
    } <= original_names
    legal = with_legal_edge_actions(observation, action_set, build.metadata)
    assert legal.candidate_count > 0
    assert transformed_variable_names_by_probindex(
        environment.model.as_pyscipopt(), observation.variable_features.shape[0]
    ) == observation.variable_names
