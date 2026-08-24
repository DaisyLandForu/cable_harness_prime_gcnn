import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rl_branching.gcnn_config import GCNNOptimizationConfig, GCNNTrainingConfig
from rl_branching.gcnn_dqn import GraphDoubleDQNLearner, stable_graph_argmax
from rl_branching.gcnn_trainer import InstanceQuotaGuard, load_shared_normalizer
from rl_branching.gcnn_model import BipartiteGCNNQNetwork, export_gcnn_torchscript
from rl_branching.graph_features import (
    GraphState,
    RunningGraphNormalizer,
    aviation_constraint_category,
    candidate_twohop_state,
    graph_state_storage_bytes,
    transition_storage_bytes,
)
from rl_branching.graph_replay import (
    DualPoolGraphReplay,
    DualPoolQuotaUnfillable,
    LOGICAL_BATCH_SIZE,
    PrioritizedBatch,
    PrioritizedReplayBuffer,
    ReplayHandle,
)
from rl_branching.replay import ReplayExperience


def immutable(values, dtype=np.float32):
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def graph_state(offset=0.0):
    return GraphState(
        row_features=immutable(np.arange(56).reshape(4, 14) + offset),
        variable_features=immutable(np.arange(125).reshape(5, 25) + offset),
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


def test_union_twohop_keeps_noncandidate_neighbors_on_candidate_rows():
    # Row 0: candidate 0 and non-candidate 1. Row 1: isolated non-candidate 2.
    full = GraphState(
        row_features=immutable(np.arange(28).reshape(2, 14)),
        variable_features=immutable(np.arange(75).reshape(3, 25)),
        edge_indices=immutable([[0, 0, 1], [0, 1, 2]], np.int64),
        edge_features=immutable([[1.0, 0.5, 1.0], [2.0, 1.0, 1.0], [-1.0, -0.5, -1.0]]),
        global_features=immutable(np.arange(14)),
        variable_categories=immutable(np.eye(6)[[0, 1, 2]]),
        row_categories=immutable(np.eye(6)[[0, 1]]),
        actions=immutable([0], np.int64),
        candidate_names=("z_1_2_0",),
        variable_names=("z_1_2_0", "t_f_9", "t_m_3_0"),
    )
    twohop = candidate_twohop_state(full)
    assert twohop.variable_names == ("z_1_2_0", "t_f_9")
    assert twohop.actions.tolist() == [0]
    assert twohop.row_features.shape[0] == 1
    assert twohop.edge_indices.tolist() == [[0, 0], [0, 1]]


def test_union_twohop_keeps_candidate_rows_and_all_row_variables():
    full = graph_state()
    twohop = candidate_twohop_state(full)
    # candidates 0/2/4 touch rows 0,2,3; those rows only contain variables 0,2,4.
    assert twohop.variable_features.shape[0] == 3
    assert twohop.row_features.shape[0] == 3
    assert twohop.actions.tolist() == [0, 1, 2]
    assert twohop.candidate_names == full.candidate_names
    assert graph_state_storage_bytes(twohop) > 0
    assert transition_storage_bytes(twohop, twohop) == (
        2 * graph_state_storage_bytes(twohop) + 64
    )


def test_dual_pool_can_keep_four_large_transitions():
    replay = DualPoolGraphReplay(
        seed=0,
        large_count_limit=8,
        large_byte_limit=64 * 1024,
        medium_count_limit=16,
        medium_byte_limit=64 * 1024,
    )
    sample = graph_state()
    nbytes = transition_storage_bytes(sample, sample)
    assert replay.can_hold_large(4, nbytes)
    for _ in range(4):
        replay.add(ReplayExperience(sample, 0, -1.0, sample, 1.0, 1), "large")
        replay.add(ReplayExperience(sample, 1, -1.0, sample, 1.0, 1), "medium")
    snapshot = replay.snapshot()
    assert snapshot.large_count == 4
    assert snapshot.can_sample_large_quota
    too_big = DualPoolGraphReplay(seed=0, large_byte_limit=nbytes * 3)
    assert not too_big.can_hold_large(4, nbytes)


def _fill_dual_pool(replay: DualPoolGraphReplay, medium: int, large: int) -> None:
    sample = graph_state()
    for index in range(medium):
        replay.add(ReplayExperience(sample, index, -1.0, sample, 1.0, 1), "medium")
    for index in range(large):
        replay.add(ReplayExperience(sample, 100 + index, -2.0, sample, 1.0, 1), "large")


def test_dual_pool_sample_returns_prioritized_batch_and_12_4_mix():
    replay = DualPoolGraphReplay(seed=0)
    _fill_dual_pool(replay, medium=20, large=6)
    batch = replay.sample_logical_batch(gradient_step=0)
    assert isinstance(batch, PrioritizedBatch)
    assert len(batch.experiences) == 16
    assert len(batch.handles) == 16
    assert batch.weights.shape == (16,)
    assert np.isfinite(batch.weights).all()
    assert float(batch.weights.max()) == 1.0
    pools = [handle.pool for handle in batch.handles]
    assert pools.count("large") == 4
    assert pools.count("medium") == 12


def test_dual_pool_falls_back_to_16_medium_when_large_is_3():
    replay = DualPoolGraphReplay(seed=1)
    _fill_dual_pool(replay, medium=20, large=3)
    batch = replay.sample_logical_batch()
    assert [handle.pool for handle in batch.handles] == ["medium"] * 16
    replay.add(ReplayExperience(graph_state(), 0, -1.0, graph_state(), 1.0, 1), "large")
    mixed = replay.sample_logical_batch()
    assert [handle.pool for handle in mixed.handles].count("large") == 4


def test_dual_pool_priority_update_and_sampling_bias():
    replay = DualPoolGraphReplay(
        seed=0,
        alpha=1.0,
        beta_start=0.0,
        beta_steps=1,
        medium_sample_quota=4,
        large_sample_quota=4,
    )
    sample = graph_state()
    ids = [
        replay.add(ReplayExperience(sample, index, -1.0, sample, 1.0, 1), "medium")
        for index in range(8)
    ]
    _fill_dual_pool(replay, medium=0, large=4)
    hot = ReplayHandle("medium", ids[0])
    replay.update_priorities([hot], np.asarray([100.0]))
    for other in ids[1:]:
        replay.update_priorities([ReplayHandle("medium", other)], np.asarray([1.0e-5]))
    hits = 0
    for step in range(80):
        batch = replay.sample_logical_batch(gradient_step=step)
        if any(handle.entry_id == ids[0] for handle in batch.handles if handle.pool == "medium"):
            hits += 1
    assert hits >= 40


def test_dual_pool_eviction_invalidates_handle_and_keeps_new_ids():
    replay = DualPoolGraphReplay(
        seed=0,
        medium_count_limit=2,
        medium_byte_limit=64 * 1024 * 1024,
    )
    sample = graph_state()
    first = replay.add(ReplayExperience(sample, 0, -1.0, sample, 1.0, 1), "medium")
    second = replay.add(ReplayExperience(sample, 1, -1.0, sample, 1.0, 1), "medium")
    third = replay.add(ReplayExperience(sample, 2, -1.0, sample, 1.0, 1), "medium")
    assert replay.snapshot().evictions_by_count == 1
    assert third != first
    try:
        replay.update_priorities([ReplayHandle("medium", first)], np.asarray([1.0]))
        raised = False
    except KeyError:
        raised = True
    assert raised
    replay.update_priorities([ReplayHandle("medium", second)], np.asarray([0.3]))
    replay.update_priorities([( "medium", third)], np.asarray([0.4]))


def test_dual_pool_same_seed_is_reproducible():
    def draw(seed: int) -> list[tuple[str, int]]:
        replay = DualPoolGraphReplay(seed=seed)
        _fill_dual_pool(replay, medium=16, large=5)
        batch = replay.sample_logical_batch(gradient_step=3)
        return [(handle.pool, handle.entry_id) for handle in batch.handles]

    assert draw(11) == draw(11)
    assert draw(11) != draw(12)


def test_dual_pool_exact_is_weights_when_pool_size_equals_quota():
    replay = DualPoolGraphReplay(
        seed=0,
        alpha=1.0,
        beta_start=1.0,
        beta_steps=1,
        medium_sample_quota=4,
        large_sample_quota=4,
    )
    sample = graph_state()
    medium_ids = [
        replay.add(ReplayExperience(sample, index, -1.0, sample, 1.0, 1), "medium")
        for index in range(4)
    ]
    large_ids = [
        replay.add(ReplayExperience(sample, 10 + index, -2.0, sample, 1.0, 1), "large")
        for index in range(4)
    ]
    medium_priorities = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    large_priorities = np.asarray([4.0, 3.0, 2.0, 1.0], dtype=np.float64)
    replay.update_priorities(
        [ReplayHandle("medium", entry_id) for entry_id in medium_ids],
        medium_priorities,
    )
    replay.update_priorities(
        [ReplayHandle("large", entry_id) for entry_id in large_ids],
        large_priorities,
    )
    batch = replay.sample_logical_batch(gradient_step=1)
    assert len(batch.handles) == 8
    medium_prob = medium_priorities / medium_priorities.sum()
    large_prob = large_priorities / large_priorities.sum()
    expected = []
    for handle in batch.handles:
        if handle.pool == "large":
            probability = large_prob[large_ids.index(handle.entry_id)]
            expected.append((4.0 * probability) ** -1.0)
        else:
            probability = medium_prob[medium_ids.index(handle.entry_id)]
            expected.append((4.0 * probability) ** -1.0)
    expected = np.asarray(expected, dtype=np.float64)
    expected /= expected.max()
    assert np.allclose(batch.weights, expected, atol=1.0e-6)


def test_dual_pool_quota_unfillable_when_medium_is_short():
    replay = DualPoolGraphReplay(seed=0)
    _fill_dual_pool(replay, medium=3, large=4)
    try:
        replay.sample_logical_batch()
        raised = False
    except DualPoolQuotaUnfillable as error:
        raised = True
        assert "quota_unfillable" in str(error)
    assert raised


def test_dual_pool_beta_anneals_and_stays_normalized():
    replay = DualPoolGraphReplay(seed=0, beta_start=0.4, beta_steps=10)
    _fill_dual_pool(replay, medium=16, large=4)
    early = replay.sample_logical_batch(gradient_step=0)
    late = DualPoolGraphReplay(seed=0, beta_start=0.4, beta_steps=10)
    _fill_dual_pool(late, medium=16, large=4)
    late_batch = late.sample_logical_batch(gradient_step=10)
    assert np.isfinite(early.weights).all() and float(early.weights.max()) == 1.0
    assert np.isfinite(late_batch.weights).all() and float(late_batch.weights.max()) == 1.0


def test_shared_normalization_is_readonly_after_freeze():
    normalizer = RunningGraphNormalizer()
    normalizer.update(graph_state())
    normalizer.freeze()
    try:
        normalizer.update(graph_state(1.0))
        raised = False
    except RuntimeError as error:
        raised = True
        assert "read-only" in str(error)
    assert raised
    shared = RunningGraphNormalizer.from_statistics(normalizer.statistics())
    assert shared.frozen


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
        assert config.optimization.batch_size == LOGICAL_BATCH_SIZE


def test_stale_batch_size_is_rejected():
    try:
        GCNNOptimizationConfig(batch_size=2)
        raised = False
    except ValueError as error:
        raised = True
        assert "batch_size=16" in str(error)
    assert raised


def test_microbatch_one_backward_before_next_forward():
    events = []
    first = graph_state()
    second = graph_state(1.0)
    online = model_for_state()
    target = model_for_state()

    def forward_hook(_module, _args):
        if torch.is_grad_enabled() and _module.training:
            events.append("forward")

    def backward_hook(_module, _grad_input, _grad_output):
        events.append("backward")

    online.register_forward_pre_hook(forward_hook)
    online.register_full_backward_hook(backward_hook)

    learner = GraphDoubleDQNLearner(
        online,
        target,
        torch.device("cpu"),
        learning_rate=0.001,
        gamma=1.0,
        gradient_clip=5.0,
        target_tau=0.1,
    )
    zero_calls = {"n": 0}
    step_calls = {"n": 0}
    original_zero = learner.optimizer.zero_grad
    original_step = learner.optimizer.step

    def counted_zero(*args, **kwargs):
        zero_calls["n"] += 1
        events.append("zero_grad")
        return original_zero(*args, **kwargs)

    def counted_step(*args, **kwargs):
        step_calls["n"] += 1
        events.append("step")
        return original_step(*args, **kwargs)

    learner.optimizer.zero_grad = counted_zero
    learner.optimizer.step = counted_step
    replay = PrioritizedReplayBuffer(8, seed=0)
    replay.add(ReplayExperience(first, 1, -2.0, second, 1.0, 3))
    replay.add(ReplayExperience(second, 0, -1.0, None, 0.0, 1))
    metrics = learner.update(replay.sample(2, gradient_step=0))
    assert zero_calls["n"] == 1
    assert step_calls["n"] == 1
    assert events[0] == "zero_grad"
    assert events[-1] == "step"
    flow = [item for item in events if item in {"forward", "backward"}]
    forwards = [index for index, item in enumerate(flow) if item == "forward"]
    first_backward = flow.index("backward")
    assert len(forwards) >= 2
    assert forwards[1] > first_backward
    assert np.isfinite(
        [metrics.loss, metrics.td_error, metrics.gradient_norm, metrics.q_mean]
    ).all()


def test_instance_quota_guard_skips_after_three_zero_decision_episodes():
    guard = InstanceQuotaGuard(("real_06", "real_08"), skip_after=3)
    assert guard.next_instance() == "real_06"
    assert guard.record_episode("real_06", 0) is False
    assert guard.record_episode("real_06", 0) is False
    assert guard.record_episode("real_06", 0) is True
    assert "real_06" in guard.unfillable
    assert guard.next_instance() == "real_08"
    assert guard.record_episode("real_08", 2) is False
    assert guard.record_episode("real_08", 0) is False
    assert guard.zero_streak["real_08"] == 1
    only = InstanceQuotaGuard(("real_09",), skip_after=3)
    for _ in range(3):
        only.record_episode("real_09", 0)
    try:
        only.next_instance()
        raised = False
    except DualPoolQuotaUnfillable as error:
        raised = True
        assert "quota_unfillable" in str(error)
    assert raised


def test_shared_normalization_missing_path_fails_fast(tmp_path: Path):
    missing = tmp_path / "missing_normalization.json"
    try:
        load_shared_normalizer(missing)
        raised = False
    except FileNotFoundError as error:
        raised = True
        assert "normalization_path is set but does not exist" in str(error)
    assert raised
    normalizer = RunningGraphNormalizer()
    normalizer.update(graph_state())
    normalizer.update(graph_state(1.0))
    present = tmp_path / "normalization.json"
    present.write_text(json.dumps(normalizer.to_json()), encoding="utf-8")
    loaded, sha256 = load_shared_normalizer(present)
    assert loaded.frozen
    assert len(sha256) == 64
    assert sha256 == hashlib.sha256(present.read_bytes()).hexdigest()


def test_smoke_and_pilot_use_dualpool_logical_batch():
    smoke = GCNNTrainingConfig.from_yaml(Path("configs/rl/gcnn_smoke.yaml"))
    pilot = GCNNTrainingConfig.from_yaml(Path("configs/rl/gcnn_prim_feat_pilot.yaml"))
    assert smoke.optimization.batch_size == LOGICAL_BATCH_SIZE
    assert pilot.optimization.batch_size == LOGICAL_BATCH_SIZE
    assert any(Path(path).stem == "real_02" for path in smoke.train_instances)
    assert any(Path(path).stem == "real_02" for path in pilot.train_instances)
