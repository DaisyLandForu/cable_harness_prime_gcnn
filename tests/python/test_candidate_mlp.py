from pathlib import Path

import numpy as np
import torch

from rl_branching.candidate_features import (
    AVIATION_VARIABLE_CATEGORIES,
    CandidateState,
    RunningFeatureNormalizer,
    aviation_variable_category,
    extract_candidate_state,
)
from rl_branching.candidate_model import CandidateQNetwork, export_torchscript
from rl_branching.dqn import DoubleDQNLearner, stable_argmax_position
from rl_branching.observation import BipartiteObservation, GLOBAL_FEATURE_NAMES
from rl_branching.replay import NStepAccumulator, OneStepExperience, ReplayBuffer
from rl_branching.training_config import MLPTrainingConfig


def immutable(values, dtype=np.float32):
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def candidate_state(offset=0.0):
    return CandidateState(
        variable_features=immutable(np.arange(57, dtype=np.float32).reshape(3, 19) + offset),
        global_features=immutable(np.arange(len(GLOBAL_FEATURE_NAMES), dtype=np.float32) + offset),
        category_features=immutable(np.eye(6, dtype=np.float32)[[0, 1, 5]]),
        actions=immutable([4, 9, 12], np.int64),
        variable_names=("t_m_0_0", "t_z_1_2_0", "auxiliary"),
    )


def test_category_and_action_mask_extraction():
    assert AVIATION_VARIABLE_CATEGORIES[aviation_variable_category("t_m_3_0")] == "m"
    assert AVIATION_VARIABLE_CATEGORIES[aviation_variable_category("t_z_1_2_0")] == "z"
    assert AVIATION_VARIABLE_CATEGORIES[aviation_variable_category("unknown")] == "other"

    observation = BipartiteObservation(
        row_features=immutable(np.zeros((2, 5))),
        variable_features=immutable(np.arange(95).reshape(5, 19)),
        edge_indices=immutable(np.array([[0, 1], [1, 3]]), np.int64),
        edge_values=immutable([1.0, -1.0]),
        global_features=immutable(np.arange(len(GLOBAL_FEATURE_NAMES))),
        variable_names=("t_y_0_0", "t_m_0_0", "t_f_0_1_0", "t_z_0_1_0", "other"),
    )
    state = extract_candidate_state(observation, immutable([3, 1], np.int64))
    assert state.actions.tolist() == [3, 1]
    assert state.variable_names == ("t_z_0_1_0", "t_m_0_0")
    assert state.variable_features[:, 0].tolist() == [57.0, 19.0]
    assert np.argmax(state.category_features, axis=1).tolist() == [1, 0]


def test_global_state_excludes_wall_clock_time():
    assert "solving_time" not in GLOBAL_FEATURE_NAMES


def test_three_step_return_and_terminal_flush():
    states = [candidate_state(float(index)) for index in range(5)]
    accumulator = NStepAccumulator(n_steps=3, gamma=1.0)
    assert not accumulator.append(OneStepExperience(states[0], 0, -1.0, states[1], 1.0))
    assert not accumulator.append(OneStepExperience(states[1], 1, -2.0, states[2], 1.0))
    emitted = accumulator.append(OneStepExperience(states[2], 2, -3.0, states[3], 1.0))
    assert len(emitted) == 1
    assert emitted[0].reward == -6.0
    assert emitted[0].next_state is states[3]
    assert emitted[0].n_steps == 3

    emitted = accumulator.append(OneStepExperience(states[3], 0, -4.0, None, 0.0))
    assert [experience.reward for experience in emitted] == [-9.0, -7.0, -4.0]
    assert all(experience.bootstrap_mask == 0.0 for experience in emitted)


def test_double_dqn_update_and_candidate_legality():
    torch.manual_seed(0)
    online = CandidateQNetwork((32, 16))
    target = CandidateQNetwork((32, 16))
    normalizer = RunningFeatureNormalizer()
    first = candidate_state()
    second = candidate_state(1.0)
    normalizer.update(first)
    normalizer.update(second)
    online.set_normalization(normalizer.statistics())
    learner = DoubleDQNLearner(
        online,
        target,
        torch.device("cpu"),
        learning_rate=0.001,
        gamma=1.0,
        gradient_clip=5.0,
        target_update_interval=2,
    )
    accumulator = NStepAccumulator(3, 1.0)
    replay = ReplayBuffer(10, seed=0)
    for experience in accumulator.append(OneStepExperience(first, 1, -1.0, second, 0.0)):
        replay.add(experience)
    assert len(replay) == 1
    metrics = learner.update(replay.sample(1))
    assert np.isfinite([metrics.loss, metrics.td_error, metrics.q_mean, metrics.q_std]).all()
    action, position, rank, q_values = learner.select_action(first, 0.0, np.random.default_rng(0))
    assert action == int(first.actions[position])
    assert action in first.actions
    assert 1 <= rank <= first.candidate_count
    assert q_values.shape == (first.candidate_count,)


def test_stable_argmax_uses_variable_name_for_q_ties():
    first = candidate_state()
    q_values = np.asarray([2.0, 2.0, 1.0], dtype=np.float32)
    assert stable_argmax_position(q_values, first) == 0

    permuted = CandidateState(
        variable_features=immutable(first.variable_features[[1, 0, 2]]),
        global_features=first.global_features,
        category_features=immutable(first.category_features[[1, 0, 2]]),
        actions=immutable(first.actions[[1, 0, 2]], np.int64),
        variable_names=(first.variable_names[1], first.variable_names[0], first.variable_names[2]),
    )
    selected = stable_argmax_position(q_values, permuted)
    assert permuted.variable_names[selected] == first.variable_names[0]


def test_checkpoint_export_reload_parity(tmp_path: Path):
    torch.manual_seed(3)
    model = CandidateQNetwork((16, 8)).eval()
    state = candidate_state()
    variables = torch.from_numpy(state.variable_features.copy())
    globals_ = torch.from_numpy(state.global_features.copy()).expand(state.candidate_count, -1)
    categories = torch.from_numpy(state.category_features.copy())
    with torch.no_grad():
        expected = model(variables, globals_, categories)
    path = tmp_path / "model.pt"
    export_torchscript(model, path)
    reloaded = torch.jit.load(str(path)).eval()
    with torch.no_grad():
        actual = reloaded(variables, globals_, categories)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert int(actual.argmax()) == int(expected.argmax())


def test_all_phase5_configs_load():
    for name, steps in (("smoke.yaml", 500), ("pilot.yaml", 5000), ("full_mlp.yaml", 20000)):
        config = MLPTrainingConfig.from_yaml(Path("configs/rl") / name)
        assert config.optimization.total_gradient_steps == steps
        assert config.optimization.n_step == 3
        assert config.optimization.gamma == 1.0
