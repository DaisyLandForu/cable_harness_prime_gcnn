from pathlib import Path

import numpy as np
import pytest

from rl_branching import BBMDPBranchingEnv, BBMDPConfig, RewardMode


# Small real instances (real_06/08/09) can finish at the root under
# project-production-v1. syn_medium still reaches a legal LP branch quickly.
REAL_INSTANCE = Path("data/instances/train/syn_medium_s101.cip")


def test_production_profile_parameters():
    config = BBMDPConfig(seed=7, node_limit=3)
    parameters = config.scip_parameters()
    assert "nodeselection/dfs/stdpriority" not in parameters
    assert "separating/maxrounds" not in parameters
    assert "limits/restarts" not in parameters
    assert parameters["parallel/maxnthreads"] == 1
    assert parameters["lp/threads"] == 1
    assert parameters["branching/preferbinary"] is True
    assert parameters["randomization/randomseedshift"] == 7
    assert parameters["randomization/permuteconss"] is True
    assert parameters["randomization/permutevars"] is True


def test_gamma_other_than_one_is_rejected():
    with pytest.raises(ValueError, match="gamma=1"):
        BBMDPConfig(gamma=0.99)


def test_done_classification_is_explicit():
    assert BBMDPBranchingEnv._classify_done(True, "optimal") == (True, False)
    assert BBMDPBranchingEnv._classify_done(True, "timelimit") == (False, True)
    assert BBMDPBranchingEnv._classify_done(True, "scip_error") == (False, True)
    assert BBMDPBranchingEnv._classify_done(False, "unknown") == (False, False)


def test_real_transition_contract_and_terminal_bootstrap():
    config = BBMDPConfig(
        seed=0,
        time_limit=60.0,
        node_limit=3,
        reward_mode=RewardMode.NEGATIVE_NODE_INCREMENT,
    )
    environment = BBMDPBranchingEnv(config)
    state = environment.reset(REAL_INSTANCE)
    assert not state.terminated and not state.truncated
    assert state.observation is not None
    state.observation.validate()
    assert state.action_set.size > 0
    assert environment.scip_parameter("parallel/maxnthreads") == 1
    assert environment.scip_parameter("branching/preferbinary") is True
    assert environment.scip_parameter("heuristics/rens/freq") == 50
    assert environment.scip_parameter("nodeselection/dfs/stdpriority") == 0
    assert environment.scip_parameter("separating/maxrounds") == -1
    assert str(environment.scip_parameter("estimation/restarts/restartpolicy")) != "n"
    assert int(environment.scip_parameter("limits/restarts")) != 0

    first_action_set = state.action_set.copy()
    retained_observation = state.observation
    transitions = []
    while not state.terminated and not state.truncated:
        action = int(state.action_set[0])
        assert environment.candidate_mapping_is_current(action)
        invalid_action = int(state.observation.variable_features.shape[0] + 1)
        with pytest.raises(ValueError, match="not in the current action set"):
            environment.step(invalid_action)
        expected_name = retained_observation.variable_names[action] if not transitions else environment.candidate_name(action)
        assert environment.candidate_name(action) == expected_name
        nodes_before = state.info["node_count"]
        transition = environment.step(action)
        transitions.append(transition)
        assert transition.action in state.action_set
        assert 0 <= transition.action_position < state.action_set.size
        assert transition.reward == -(transition.info["node_count"] - nodes_before)
        if transition.next_observation is not None:
            transition.next_observation.validate()
            assert transition.next_action_set.max() < transition.next_observation.variable_features.shape[0]
        state = environment.current_state

    assert transitions
    assert state.truncated
    assert state.info["status"] == "nodelimit"
    assert transitions[-1].bootstrap_mask == 0.0
    assert transitions[-1].next_observation is None
    assert transitions[-1].next_action_set.size == 0
    tree = environment.search_tree
    assert tree.visited_node_ids
    assert all(isinstance(node_id, int) for node_id in tree.visited_node_ids)
    assert all(parent is None or isinstance(parent, int) for parent in tree.parent_by_node.values())

    retained_name = retained_observation.variable_names[int(first_action_set[0])]
    retained_shape = retained_observation.variable_features.shape
    environment.close()
    assert isinstance(retained_name, str)
    assert retained_observation.variable_features.shape == retained_shape
    assert not retained_observation.variable_features.flags.writeable


def test_scip_caps_widen_when_bootstrap_is_enabled():
    config = BBMDPConfig(
        seed=0,
        time_limit=60.0,
        node_limit=3,
        bootstrap_on_truncation=True,
    )
    assert config.time_limit == 60.0
    assert config.node_limit == 3
    assert config.scip_search_limits() == (65.0, 4)


def test_hybrid_reward_identity_on_real_instance():
    config = BBMDPConfig(
        seed=0,
        time_limit=60.0,
        node_limit=3,
        reward_mode=RewardMode.HYBRID_NODE_LP,
    )
    environment = BBMDPBranchingEnv(config)
    state = environment.reset(REAL_INSTANCE)
    n0 = int(state.info["node_count"])
    lp0 = int(state.info["lp_iterations"])
    rewards = []
    while not state.terminated and not state.truncated:
        transition = environment.step(int(state.action_set[0]))
        rewards.append(float(transition.reward))
        assert transition.info["total_reward"] == transition.reward
        assert abs(
            transition.info["node_reward"] + transition.info["lp_reward"] - transition.reward
        ) <= 1.0e-12
        state = environment.current_state
    n_t = int(state.info["node_count"])
    lp_t = int(state.info["lp_iterations"])
    if n_t >= n0 and lp_t >= lp0:
        from rl_branching.config import hybrid_identity_holds

        assert hybrid_identity_holds(rewards, n0, n_t, lp0, lp_t)
    environment.close()


def test_live_budget_truncation_keeps_bootstrap_state():
    config = BBMDPConfig(
        seed=0,
        time_limit=60.0,
        node_limit=3,
        bootstrap_on_truncation=True,
        reward_mode=RewardMode.HYBRID_NODE_LP,
    )
    environment = BBMDPBranchingEnv(config)
    state = environment.reset(REAL_INSTANCE)
    last = None
    while not state.terminated and not state.truncated:
        last = environment.mark_live_budget_truncation(
            environment.step(int(state.action_set[0]))
        )
        state = environment.current_state
    assert last is not None
    assert last.truncated
    assert not last.terminated
    assert last.info.get("trainer_truncated") is True
    assert last.bootstrap_mask == 1.0
    assert last.next_observation is not None
    assert last.next_action_set.size > 0
    assert state.observation is not None
    environment.close()


def test_timeout_is_an_explicit_truncation():
    environment = BBMDPBranchingEnv(
        BBMDPConfig(seed=0, time_limit=0.001, node_limit=1000)
    )
    state = environment.reset(REAL_INSTANCE)
    assert state.truncated
    assert not state.terminated
    assert state.info["status"] == "timelimit"
    assert state.observation is None
    assert state.action_set.size == 0
    environment.close()
