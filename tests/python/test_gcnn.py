from pathlib import Path

import numpy as np
import torch

from rl_branching.gcnn_config import GCNNTrainingConfig
from rl_branching.gcnn_dqn import GraphDoubleDQNLearner, stable_graph_argmax
from rl_branching.gcnn_model import BipartiteGCNNQNetwork, export_gcnn_torchscript
from rl_branching.graph_features import (
    GraphState,
    RunningGraphNormalizer,
    aviation_constraint_category,
)
from rl_branching.graph_replay import PrioritizedReplayBuffer
from rl_branching.replay import ReplayExperience


def immutable(values, dtype=np.float32):
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def graph_state(offset=0.0):
    return GraphState(
        row_features=immutable(np.arange(56).reshape(4, 14) + offset),
        variable_features=immutable(np.arange(95).reshape(5, 19) + offset),
        edge_indices=immutable(
            [[0, 0, 1, 1, 2, 3, 3], [0, 2, 1, 3, 4, 0, 4]], np.int64
        ),
        edge_features=immutable(
            [[1.0, 0.5, 1.0], [-1.0, -0.5, -1.0], [2.0, 0.7, 1.0],
             [1.0, 0.3, 1.0], [-2.0, -0.8, -1.0], [1.0, 0.5, 1.0],
             [-1.0, -0.5, -1.0]]
        ),
        global_features=immutable(np.arange(14) + offset),
        variable_categories=immutable(np.eye(6)[[0, 1, 2, 3, 5]]),
        row_categories=immutable(np.eye(6)[[0, 1, 2, 5]]),
        actions=immutable([0, 2, 4], np.int64),
        candidate_names=("t_m_0", "t_z_0", "other"),
    )


def model_for_state(bins=1):
    torch.manual_seed(0)
    model = BipartiteGCNNQNetwork(embedding_dim=16, hidden_dim=32, distributional_bins=bins)
    normalizer = RunningGraphNormalizer()
    normalizer.update(graph_state())
    normalizer.update(graph_state(1.0))
    model.set_normalization(normalizer.statistics())
    return model


def test_constraint_categories_and_graph_validation():
    assert aviation_constraint_category("t_flow_balance") == 0
    assert aviation_constraint_category("abs1") == 1
    assert aviation_constraint_category("topo_seq1") == 2
    assert aviation_constraint_category("zlower") == 3
    assert aviation_constraint_category("imbalance") == 4
    assert aviation_constraint_category("cut_pool") == 5
    graph_state().validate()


def test_per_sampling_and_priority_update():
    first = graph_state()
    replay = PrioritizedReplayBuffer(8, seed=0)
    for position in range(3):
        replay.add(ReplayExperience(first, position, -1.0, None, 0.0, 1))
    batch = replay.sample(2, gradient_step=4)
    assert len(batch.experiences) == 2
    assert batch.weights.shape == (2,)
    replay.update_priorities(batch.indices, np.asarray([0.5, 2.0]))


def test_scalar_and_hl_gauss_updates_are_finite():
    first = graph_state()
    second = graph_state(1.0)
    for bins in (1, 18):
        online = model_for_state(bins)
        target = model_for_state(bins)
        learner = GraphDoubleDQNLearner(
            online,
            target,
            torch.device("cpu"),
            learning_rate=0.001,
            gamma=1.0,
            gradient_clip=5.0,
            target_tau=0.1,
        )
        replay = PrioritizedReplayBuffer(8, seed=0)
        replay.add(ReplayExperience(first, 1, -2.0, second, 1.0, 3))
        replay.add(ReplayExperience(second, 0, -1.0, None, 0.0, 1))
        batch = replay.sample(2, gradient_step=0)
        metrics = learner.update(batch)
        assert np.isfinite(
            [metrics.loss, metrics.td_error, metrics.q_mean, metrics.q_std]
        ).all()
        replay.update_priorities(batch.indices, metrics.priorities)
        action, position, rank, values = learner.select_action(
            first, 0.0, np.random.default_rng(0)
        )
        assert action == first.actions[position]
        assert 1 <= rank <= first.candidate_count
        assert values.shape == (first.candidate_count,)


def test_torchscript_full_q_parity(tmp_path: Path):
    state = graph_state()
    model = model_for_state().eval()
    tensors = tuple(
        torch.tensor(values)
        for values in (
            state.row_features,
            state.variable_features,
            state.edge_indices,
            state.edge_features,
            state.global_features,
            state.variable_categories,
            state.row_categories,
            state.actions,
        )
    )
    with torch.no_grad():
        expected = model(*tensors)
    path = tmp_path / "gcnn.pt"
    export_gcnn_torchscript(model, path)
    loaded = torch.jit.load(str(path)).eval()
    with torch.no_grad():
        actual = loaded(*tensors)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert stable_graph_argmax(actual.numpy(), state) == stable_graph_argmax(
        expected.numpy(), state
    )


def test_feature_ablation_switches_ignore_masked_inputs():
    state = graph_state()
    changed = GraphState(
        row_features=state.row_features,
        variable_features=state.variable_features,
        edge_indices=state.edge_indices,
        edge_features=state.edge_features,
        global_features=immutable(np.full(14, 1000.0)),
        variable_categories=immutable(np.roll(state.variable_categories, 1, axis=1)),
        row_categories=immutable(np.roll(state.row_categories, 1, axis=1)),
        actions=state.actions,
        candidate_names=state.candidate_names,
    )
    torch.manual_seed(0)
    model = BipartiteGCNNQNetwork(
        embedding_dim=16,
        hidden_dim=32,
        use_aviation_categories=False,
        use_global_features=False,
    ).eval()
    first = tuple(torch.tensor(getattr(state, name)) for name in (
        "row_features", "variable_features", "edge_indices", "edge_features",
        "global_features", "variable_categories", "row_categories", "actions",
    ))
    second = tuple(torch.tensor(getattr(changed, name)) for name in (
        "row_features", "variable_features", "edge_indices", "edge_features",
        "global_features", "variable_categories", "row_categories", "actions",
    ))
    with torch.no_grad():
        torch.testing.assert_close(model(*first), model(*second), rtol=0.0, atol=0.0)


def test_all_gcnn_configs_load():
    for name, mode in (
        ("gcnn_smoke.yaml", "scalar"),
        ("gcnn_pilot.yaml", "scalar"),
        ("gcnn_hlgauss.yaml", "hl_gauss"),
        ("ablation_nstep1.yaml", "scalar"),
        ("ablation_hlgauss.yaml", "hl_gauss"),
        ("ablation_no_categories.yaml", "scalar"),
        ("ablation_no_global.yaml", "scalar"),
    ):
        config = GCNNTrainingConfig.from_yaml(Path("configs/rl") / name)
        assert config.model.loss_mode == mode
        assert config.optimization.n_step == (1 if name == "ablation_nstep1.yaml" else 3)
        assert config.optimization.gamma == 1.0
