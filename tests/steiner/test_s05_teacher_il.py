from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path

import ecole
import numpy as np
import pytest
import torch

from steiner_branching.config import StrictConfigError
from steiner_branching.data.generate import GeneratorConfig, generate_graph
from steiner_branching.learning.imitation import (
    CUBLAS_WORKSPACE_CONFIG,
    enable_cuda_determinism,
    fit_train_normalization,
    listwise_cross_entropy,
    load_checkpoint_bundle,
    model_predictions,
    normalize_state,
    normalized_sb_regret,
    offline_baseline_predictions,
    ranking_metrics,
    save_checkpoint_bundle,
    spearman_rank_correlation,
    train_one_epoch,
)
from steiner_branching.learning.teacher_data import (
    StrongBranchLabels,
    TeacherSample,
    assert_no_split_leakage,
    expand_pilot_tasks,
    file_sha256,
    immutable_array,
    load_s05_config,
    load_teacher_sample,
    s05_config_sha256,
    write_teacher_sample,
)
from steiner_branching.milp.mcf import build_mcf
from steiner_branching.models.milp_gcnn import MilpBipartiteGCNN
from steiner_branching.solver.bipartite_observation import (
    SteinerNodeBipartite,
    make_bipartite_state,
    with_legal_edge_actions,
)
from steiner_branching.solver.branchability import configure_p1, load_s03_config
from steiner_branching.solver.strong_branching import (
    OrderedStrongBranchObservation,
    StrongBranchingError,
    StrongBranchingTeacher,
    align_labels_to_probindices,
)


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/steiner/experiments/s05_teacher_il_pilot_v1.yml"
CONFIG_V2 = REPO / "configs/steiner/experiments/s05_teacher_il_pilot_v2.yml"
CONFIG_V3 = REPO / "configs/steiner/experiments/s05_teacher_il_pilot_v3.yml"
B0_CONFIG = REPO / "configs/steiner/models/b0_milp_gcnn_v1.yml"


def _load_script_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state():
    variables = np.zeros((3, 19), dtype=np.float32)
    variables[:2, 1] = 1.0
    variables[2, 4] = 1.0
    variables[:2, 9] = (0.25, 0.5)
    return make_bipartite_state(
        constraint_features=np.arange(10, dtype=np.float32).reshape(2, 5),
        variable_features=variables,
        edge_indices=np.asarray(((0, 0, 1), (0, 2, 1)), dtype=np.int64),
        edge_features=np.asarray((1.0, 0.5, -1.0), dtype=np.float32),
        variable_names=("t_stp_x_e00000000", "t_stp_x_e00000001", "t_stp_f_t0002_a00000000"),
        candidate_indices=np.asarray((0, 1), dtype=np.int64),
        candidate_edge_ids=np.asarray((0, 1), dtype=np.int64),
    )


def _labels(*, valid: bool = True) -> StrongBranchLabels:
    return StrongBranchLabels(
        candidate_probindices=immutable_array((0, 1), np.int64),
        scores=immutable_array((1.0, 3.0) if valid else (1.0, np.nan), np.float64),
        down_bounds=immutable_array((2.0, 4.0), np.float64),
        up_bounds=immutable_array((2.5, 4.5), np.float64),
        down_valid=immutable_array((True, valid), np.bool_),
        up_valid=immutable_array((True, valid), np.bool_),
        down_infeasible=immutable_array((False, False), np.bool_),
        up_infeasible=immutable_array((False, False), np.bool_),
        lp_errors=immutable_array((False, not valid), np.bool_),
        node_number=1,
        depth=0,
        elapsed_seconds=0.25,
        lp_iterations_delta=10,
        strong_lp_iterations_delta=10,
        strong_calls_delta=2,
    )


def _sample(*, split: str = "train", graph_sha256: str = "a" * 64) -> TeacherSample:
    return TeacherSample(
        task_id=f"{split}-task",
        split=split,
        family="sparse_erdos_renyi",
        graph_sha256=graph_sha256,
        teacher_seed=1001,
        state_index=0,
        state=_state(),
        labels=_labels(),
        pseudocost_scores=immutable_array((0.25, 0.75), np.float64),
    )


def test_s05_config_freezes_splits_seeds_and_learning_curve(monkeypatch):
    config = load_s05_config(CONFIG)
    tasks = expand_pilot_tasks(config)
    assert len(tasks) == 10
    assert {task.split for task in tasks} == {"train", "validation_iid"}
    assert {task.family for task in tasks} == {
        "sparse_erdos_renyi", "random_geometric", "grid_with_holes",
        "community_block", "bridge_bottleneck",
    }
    assert {task.teacher_seed for task in tasks} == {1001}
    assert config["training"]["learning_curve_train_states"] == [16, 32, 64]
    assert len(s05_config_sha256(config)) == 64

    bad = copy.deepcopy(config)
    bad["pilot_instances"][0]["generator_seed"] = 300000
    with pytest.raises(StrictConfigError, match="belongs to test_iid"):
        from steiner_branching.learning import teacher_data

        monkeypatch.setattr(teacher_data, "load_yaml_mapping", lambda _path: bad)
        teacher_data.load_s05_config("unused.yml")


def test_s05_pilot_v2_adds_only_two_s03_branchable_train_instances(monkeypatch):
    config = load_s05_config(CONFIG_V2)
    tasks = expand_pilot_tasks(config)
    assert len(tasks) == 12
    assert config["experiment_id"] == "s05-teacher-il-pilot-v2"
    assert config["artifacts"] == {
        "raw_root": "results/steiner/raw/s05/s05-teacher-il-pilot-v2",
        "report_root": "results/steiner/s05/s05-teacher-il-pilot-v2",
    }
    added = {
        (task.split, task.family, task.generator_seed)
        for task in tasks
        if task.generator_seed in {100303, 100315}
    }
    assert added == {
        ("train", "sparse_erdos_renyi", 100303),
        ("train", "grid_with_holes", 100315),
    }
    assert {task.teacher_seed for task in tasks} == {1001}

    changed = copy.deepcopy(config)
    changed["pilot_instances"][-1]["generator_seed"] = 100317
    from steiner_branching.learning import teacher_data

    monkeypatch.setattr(teacher_data, "load_yaml_mapping", lambda _path: changed)
    with pytest.raises(StrictConfigError, match="identities/order changed"):
        teacher_data.load_s05_config("unused.yml")


def test_s05_pilot_v3_changes_only_registered_cuda_determinism_and_paths(monkeypatch):
    v2 = load_s05_config(CONFIG_V2)
    v3 = load_s05_config(CONFIG_V3)
    assert v3["experiment_id"] == "s05-teacher-il-pilot-v3"
    assert v3["pilot_instances"] == v2["pilot_instances"]
    assert v3["teacher"] == v2["teacher"]
    assert v3["gate"] == v2["gate"]
    assert v3["training"]["deterministic_algorithms"] is True
    assert v3["training"]["cublas_workspace_config"] == CUBLAS_WORKSPACE_CONFIG
    assert v3["artifacts"] == {
        "raw_root": "results/steiner/raw/s05/s05-teacher-il-pilot-v3",
        "report_root": "results/steiner/s05/s05-teacher-il-pilot-v3",
    }

    changed = copy.deepcopy(v3)
    changed["training"]["cublas_workspace_config"] = ":16:8"
    from steiner_branching.learning import teacher_data

    monkeypatch.setattr(teacher_data, "load_yaml_mapping", lambda _path: changed)
    with pytest.raises(StrictConfigError, match="determinism controls changed"):
        teacher_data.load_s05_config("unused.yml")


def test_cuda_determinism_contract_fails_closed_without_registered_environment(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="before the Python process starts"):
        enable_cuda_determinism(expected_workspace_config=CUBLAS_WORKSPACE_CONFIG)


def test_pilot_training_seed_shards_are_registered_and_disjoint(tmp_path):
    trainer = _load_script_module("s05_train_script", "scripts/steiner/train_s05_il.py")
    expected = [101, 202, 303]
    assert trainer.select_training_seeds(expected, None) == (101, 202, 303)
    assert trainer.select_training_seeds(expected, [101, 303]) == (101, 303)
    assert trainer.training_report_path(
        tmp_path, (101, 303), (101, 202, 303)
    ) == tmp_path / "pilot_training_shards" / "seeds-101-303.json"
    assert trainer.training_report_path(
        tmp_path, (101, 202, 303), (101, 202, 303)
    ) == tmp_path / "pilot_training_report.json"
    with pytest.raises(ValueError, match="duplicates"):
        trainer.select_training_seeds(expected, [101, 101])
    with pytest.raises(ValueError, match="not registered"):
        trainer.select_training_seeds(expected, [404])


def test_parallel_pilot_reports_merge_only_when_seed_matrix_is_complete(tmp_path):
    aggregator = _load_script_module(
        "s05_aggregate_script", "scripts/steiner/aggregate_s05_pilot_training.py"
    )
    config = load_s05_config(CONFIG)
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    manifest_path = raw_root / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    config["artifacts"]["raw_root"] = str(raw_root)
    config["artifacts"]["report_root"] = str(tmp_path / "reports")
    config_digest = s05_config_sha256(config)
    manifest_digest = file_sha256(manifest_path)
    git_commit = "a" * 40
    audited_target = "b" * 40

    def write_report(name: str, seeds: list[int]) -> Path:
        path = tmp_path / name
        report = {
            "schema_version": 1,
            "stage": "S05",
            "experiment_id": config["experiment_id"],
            "started_at_utc": "2026-09-03T00:00:00Z",
            "finished_at_utc": "2026-09-03T00:01:00Z",
            "git_commit": git_commit,
            "s04_audited_tag_target": audited_target,
            "config_sha256": config_digest,
            "data_manifest": str(manifest_path),
            "data_manifest_sha256": manifest_digest,
            "train_valid_states": 64,
            "validation_valid_states": 16,
            "report_kind": "pilot_seed_shard",
            "selected_training_seeds": seeds,
            "expected_training_seeds": [101, 202, 303],
            "offline_baselines": {"random": {"normalized_sb_regret": 0.5}},
            "cuda_determinism": {
                "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
                "deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
            },
            "runs": [
                {
                    "training_seed": seed,
                    "train_state_count": count,
                    "status": "completed",
                }
                for count in (16, 32, 64)
                for seed in seeds
            ],
            "formal_gate_evaluated": False,
            "status": "completed",
        }
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    left = write_report("left.json", [101, 303])
    right = write_report("right.json", [202])
    output = tmp_path / "aggregate.json"
    merged = aggregator.aggregate_reports(
        config=config,
        input_paths=[left, right],
        output_path=output,
        git_commit=git_commit,
        audited_target=audited_target,
    )
    assert output.is_file()
    assert merged["status"] == "completed"
    assert merged["report_kind"] == "pilot_aggregate"
    assert merged["selected_training_seeds"] == [101, 202, 303]
    assert len(merged["runs"]) == 9

    duplicate = write_report("duplicate.json", [101])
    with pytest.raises(ValueError, match="multiple shards"):
        aggregator.aggregate_reports(
            config=config,
            input_paths=[left, right, duplicate],
            output_path=output,
            git_commit=git_commit,
            audited_target=audited_target,
        )
    with pytest.raises(ValueError, match="incomplete"):
        aggregator.aggregate_reports(
            config=config,
            input_paths=[left],
            output_path=output,
            git_commit=git_commit,
            audited_target=audited_target,
        )


def test_teacher_label_alignment_is_by_probindex_and_fails_closed():
    aligned = align_labels_to_probindices(_labels(), np.asarray((1, 0)))
    assert aligned.candidate_probindices.tolist() == [1, 0]
    assert aligned.scores.tolist() == [3.0, 1.0]
    with pytest.raises(StrongBranchingError, match="not a bijection"):
        align_labels_to_probindices(_labels(), np.asarray((0, 2)))
    with pytest.raises(StrongBranchingError, match="not a unique"):
        align_labels_to_probindices(_labels(), np.asarray((0, 0)))
    assert not _labels(valid=False).state_valid
    assert np.isnan(_labels(valid=False).scores[1])


def test_teacher_shard_round_trip_checks_file_and_semantic_identity(tmp_path):
    sample = _sample()
    path = tmp_path / "state.npz"
    hashes = write_teacher_sample(path, sample)
    restored = load_teacher_sample(path, expected_file_sha256=hashes["file_sha256"])
    assert restored.semantic_sha256 == sample.semantic_sha256 == hashes["semantic_sha256"]
    assert restored.state.sha256 == sample.state.sha256
    assert restored.labels.scores.tolist() == [1.0, 3.0]
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_teacher_sample(path, expected_file_sha256="0" * 64)

    invalid = TeacherSample(
        task_id=sample.task_id,
        split=sample.split,
        family=sample.family,
        graph_sha256=sample.graph_sha256,
        teacher_seed=sample.teacher_seed,
        state_index=1,
        state=sample.state,
        labels=_labels(valid=False),
        pseudocost_scores=sample.pseudocost_scores,
    )
    invalid_path = tmp_path / "invalid-state.npz"
    invalid_hashes = write_teacher_sample(invalid_path, invalid)
    restored_invalid = load_teacher_sample(
        invalid_path, expected_file_sha256=invalid_hashes["file_sha256"]
    )
    assert not restored_invalid.labels.state_valid
    assert np.isnan(restored_invalid.labels.scores[1])

    validation = _sample(split="validation_iid", graph_sha256="b" * 64)
    assert_no_split_leakage((sample, validation))
    with pytest.raises(ValueError, match="crosses splits"):
        assert_no_split_leakage((_sample(), _sample(split="validation_iid")))


def test_train_only_normalization_listwise_loss_and_metrics():
    train = _sample()
    validation = _sample(split="validation_iid", graph_sha256="b" * 64)
    stats = fit_train_normalization((train,))
    normalized = normalize_state(train.state, stats)
    assert np.isfinite(normalized.variable_features).all()
    with pytest.raises(ValueError, match="train samples only"):
        fit_train_normalization((validation,))

    logits = torch.tensor((0.0, 2.0), requires_grad=True)
    loss = listwise_cross_entropy(logits, train.labels.scores, temperature=1.0)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None
    assert normalized_sb_regret(train.labels.scores, 1, tie_tolerance=1.0e-9) == 0.0
    assert normalized_sb_regret(train.labels.scores, 0, tie_tolerance=1.0e-9) == 1.0
    assert spearman_rank_correlation(
        np.asarray((0.0, 1.0)), train.labels.scores
    ) == pytest.approx(1.0)
    metrics = ranking_metrics((train,), (np.asarray((0.0, 1.0)),), tie_tolerance=1.0e-9)
    assert metrics["top1_accuracy"] == 1.0 and metrics["normalized_sb_regret"] == 0.0
    for baseline in ("random", "most_infeasible", "pseudocost"):
        predictions = offline_baseline_predictions((train,), baseline=baseline, seed=7)
        assert predictions[0].shape == (2,) and np.isfinite(predictions[0]).all()


def test_cpu_training_and_checkpoint_reload_reproduce_logits(tmp_path):
    sample = _sample()
    stats = fit_train_normalization((sample,))
    torch.manual_seed(101)
    model = MilpBipartiteGCNN(embedding_dim=64, hidden_dim=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    loss = train_one_epoch(
        model,
        (sample,),
        stats,
        optimizer,
        device=torch.device("cpu"),
        temperature=1.0,
        gradient_clip_norm=1.0,
    )
    assert np.isfinite(loss)
    manifest = save_checkpoint_bundle(
        output_dir=tmp_path / "checkpoint",
        model=model,
        stats=stats,
        manifest_fields={
            "config_sha256": "1" * 64,
            "data_manifest_sha256": "2" * 64,
            "git_commit": "3" * 40,
            "training_seed": 101,
            "train_state_count": 1,
            "validation_metrics": {"normalized_sb_regret": 0.0},
            "model_config_sha256": file_sha256(B0_CONFIG),
            "bipartite_schema_id": "milp_bipartite_v1",
            "solver_stack_id": "scip804-ecole081-pyscipopt430",
            "pytorch_version": torch.__version__,
            "cuda_determinism": {
                "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
                "deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
            },
        },
    )
    restored, restored_stats, restored_manifest = load_checkpoint_bundle(
        manifest, model_config_path=B0_CONFIG, device=torch.device("cpu")
    )
    assert restored_manifest["training_seed"] == 101
    assert restored_stats == stats
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cuda_repeated_inference_and_checkpoint_reload_are_bit_exact(tmp_path):
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        pytest.skip("requires CUBLAS_WORKSPACE_CONFIG before pytest starts")
    contract = enable_cuda_determinism(
        expected_workspace_config=CUBLAS_WORKSPACE_CONFIG
    )
    sample = _sample()
    stats = fit_train_normalization((sample,))
    device = torch.device("cuda:0")
    torch.manual_seed(101)
    model = MilpBipartiteGCNN(embedding_dim=64, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    loss = train_one_epoch(
        model,
        (sample,),
        stats,
        optimizer,
        device=device,
        temperature=1.0,
        gradient_clip_norm=1.0,
    )
    assert np.isfinite(loss)
    first = model_predictions(model, (sample,), stats, device=device)[0]
    second = model_predictions(model, (sample,), stats, device=device)[0]
    np.testing.assert_array_equal(first, second)
    manifest = save_checkpoint_bundle(
        output_dir=tmp_path / "cuda-checkpoint",
        model=model,
        stats=stats,
        manifest_fields={
            "config_sha256": "1" * 64,
            "data_manifest_sha256": "2" * 64,
            "git_commit": "3" * 40,
            "training_seed": 101,
            "train_state_count": 1,
            "validation_metrics": {"normalized_sb_regret": 0.0},
            "model_config_sha256": file_sha256(B0_CONFIG),
            "bipartite_schema_id": "milp_bipartite_v1",
            "solver_stack_id": "scip804-ecole081-pyscipopt430",
            "pytorch_version": torch.__version__,
            "cuda_determinism": contract,
        },
    )
    restored, restored_stats, _ = load_checkpoint_bundle(
        manifest, model_config_path=B0_CONFIG, device=device
    )
    assert restored_stats == stats
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0.0, atol=0.0)
    reloaded = model_predictions(restored, (sample,), restored_stats, device=device)[0]
    np.testing.assert_array_equal(first, reloaded)


def test_real_frozen_scip_teacher_records_child_validity_and_probindex_identity():
    if not __import__("os").environ.get("STEINER_SOLVER_STACK_ID"):
        pytest.skip("requires the frozen SCIP 8.0.4 wrapper")
    graph = generate_graph(
        GeneratorConfig(
            family="sparse_erdos_renyi", n_nodes=48, n_terminals=5, seed=100300
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
    environment = ecole.environment.Branching(
        observation_function=OrderedStrongBranchObservation(
            SteinerNodeBipartite(),
            StrongBranchingTeacher(iteration_limit=10000),
            include_ecole_reference=True,
        ),
        pseudo_candidates=False,
    )
    environment.seed(1001)
    observation, action_set, _reward, done, _info = environment.reset(
        ecole.scip.Model.from_pyscipopt(build.model)
    )
    assert not done and observation is not None and action_set is not None
    state = with_legal_edge_actions(observation["state"], action_set, build.metadata)
    labels = align_labels_to_probindices(observation["teacher"], action_set)
    assert np.array_equal(labels.candidate_probindices, state.candidate_indices)
    assert labels.state_valid and np.isfinite(labels.scores).all()
    np.testing.assert_allclose(
        labels.scores,
        np.asarray(observation["ecole_teacher"])[state.candidate_indices],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert labels.strong_calls_delta >= state.candidate_count
    assert labels.strong_lp_iterations_delta > 0
    assert len(state.candidate_edge_ids) == state.candidate_count
