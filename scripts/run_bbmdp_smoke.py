#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from rl_branching import BBMDPBranchingEnv, BBMDPConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run a controlled BBMDP branching episode")
    parser.add_argument("--config", type=Path, default=Path("configs/rl/environment_smoke.yaml"))
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--policy", choices=("random", "mostinf"), default="random")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def choose_action(state, policy, random_engine):
    if policy == "random":
        return int(random_engine.choice(state.action_set))
    fractionalities = state.observation.variable_features[state.action_set, 9]
    scores = np.minimum(fractionalities, 1.0 - fractionalities)
    return int(state.action_set[int(np.argmax(scores))])


def main():
    args = parse_args()
    config = BBMDPConfig.from_yaml(args.config)
    random_engine = np.random.default_rng(config.seed)
    environment = BBMDPBranchingEnv(config)
    state = environment.reset(args.instance)
    transitions = []

    while not state.terminated and not state.truncated:
        action = choose_action(state, args.policy, random_engine)
        action_name = environment.candidate_name(action)
        nodes_before = state.info["node_count"]
        transition = environment.step(action)
        transitions.append(
            {
                "step": len(transitions),
                "node_id": state.info["current_node_id"],
                "parent_node_id": state.info["parent_node_id"],
                "depth": state.info["depth"],
                "candidate_count": int(state.action_set.size),
                "action": action,
                "action_position": transition.action_position,
                "action_name": action_name,
                "reward": transition.reward,
                "nodes_before": nodes_before,
                "nodes_after": transition.info["node_count"],
                "next_candidate_count": int(transition.next_action_set.size),
                "terminated": transition.terminated,
                "truncated": transition.truncated,
                "bootstrap_mask": transition.bootstrap_mask,
                "status": transition.info["status"],
                "variable_shape": list(transition.observation.variable_features.shape),
                "constraint_shape": list(transition.observation.row_features.shape),
                "edge_count": int(transition.observation.edge_values.size),
                "next_variable_shape": (
                    None
                    if transition.next_observation is None
                    else list(transition.next_observation.variable_features.shape)
                ),
                "next_constraint_shape": (
                    None
                    if transition.next_observation is None
                    else list(transition.next_observation.row_features.shape)
                ),
            }
        )
        state = environment.current_state

    summary = {
        "instance": str(args.instance),
        "policy": args.policy,
        "seed": config.seed,
        "reward_mode": config.reward_mode.value,
        "gamma": config.gamma,
        "transition_count": len(transitions),
        "status": state.info["status"],
        "terminated": state.terminated,
        "truncated": state.truncated,
        "nodes": state.info["node_count"],
        "lp_iterations": state.info["lp_iterations"],
        "solving_time": state.info["solving_time"],
        "total_reward": float(sum(item["reward"] for item in transitions)),
        "visited_tree_nodes": len(environment.search_tree.visited_node_ids),
        "transitions": transitions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    environment.close()
    print(json.dumps({key: value for key, value in summary.items() if key != "transitions"}, indent=2))


if __name__ == "__main__":
    main()
