from pathlib import Path

import numpy as np
import pytest

from rl_branching import BBMDPBranchingEnv, BBMDPConfig, RewardMode


REAL_INSTANCE = Path("data/instances/train/real_06.cip")


def test_controlled_profile_parameters():
    config = BBMDPConfig(seed=7, node_limit=3)
    parameters = config.scip_parameters()
    assert parameters["nodeselection/dfs/stdpriority"] == 1_000_000
    assert parameters["separating/maxrounds"] == 0
    assert parameters["estimation/restarts/restartpolicy"] == "n"
    assert parameters["limits/restarts"] == 0
    assert parameters["parallel/maxnthreads"] == 1
    assert parameters["lp/threads"] == 1
    assert parameters["randomization/randomseedshift"] == 7


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
    assert environment.scip_parameter("nodeselection/dfs/stdpriority") == 1_000_000
    assert environment.scip_parameter("separating/maxrounds") == 0
    assert environment.scip_parameter("estimation/restarts/restartpolicy") == "n"
    assert environment.scip_parameter("parallel/maxnthreads") == 1

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
