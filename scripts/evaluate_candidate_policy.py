#!/usr/bin/env python3
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from rl_branching.candidate_features import extract_candidate_state
from rl_branching.dqn import stable_argmax_position
from rl_branching.environment import BBMDPBranchingEnv
from rl_branching.training_config import MLPTrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one exported Candidate MLP episode")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--instance", required=True, type=Path)
    parser.add_argument("--method", choices=("random", "rl"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--node-limit", type=int)
    args = parser.parse_args()

    config = MLPTrainingConfig.from_yaml(args.artifact_dir / "config.yaml")
    policy = None
    if args.method == "rl":
        policy = torch.jit.load(
            str(args.artifact_dir / "best_model_scripted.pt"), map_location="cpu"
        ).eval()
    rng = np.random.default_rng(args.policy_seed)
    environment_config = replace(config.environment, seed=args.seed)
    if args.node_limit is not None:
        environment_config = replace(environment_config, node_limit=args.node_limit)
    environment = BBMDPBranchingEnv(environment_config)
    state = environment.reset(args.instance)
    actions: list[int] = []
    decisions: list[dict] = []
    root_fingerprint = None
    reward = 0.0
    while not (state.terminated or state.truncated):
        compact = extract_candidate_state(state.observation, state.action_set)
        if policy is None:
            position = int(rng.integers(compact.candidate_count))
            q_values = None
        else:
            variables = torch.tensor(compact.variable_features, dtype=torch.float32)
            globals_ = torch.tensor(compact.global_features, dtype=torch.float32).expand(
                compact.candidate_count, -1
            )
            categories = torch.tensor(compact.category_features, dtype=torch.float32)
            with torch.no_grad():
                q_values = policy(variables, globals_, categories).numpy()
            if not np.isfinite(q_values).all():
                raise FloatingPointError("model produced NaN or Inf")
            position = stable_argmax_position(q_values, compact)
        if root_fingerprint is None:
            root_fingerprint = {
                "candidate_count": compact.candidate_count,
                "actions_sha256": hashlib.sha256(compact.actions.tobytes()).hexdigest(),
                "variable_features_sha256": hashlib.sha256(
                    compact.variable_features.tobytes()
                ).hexdigest(),
                "global_features_sha256": hashlib.sha256(
                    compact.global_features.tobytes()
                ).hexdigest(),
            }
            if q_values is not None:
                top_positions = np.argsort(q_values)[-5:][::-1]
                root_fingerprint["top_q"] = [
                    {
                        "action": int(compact.actions[top_position]),
                        "variable_name": compact.variable_names[top_position],
                        "q_value": float(q_values[top_position]),
                    }
                    for top_position in top_positions
                ]
        action = int(compact.actions[position])
        decision = {
            "action": action,
            "variable_name": compact.variable_names[position],
            "depth": int(state.info.get("depth", -1)),
            "node_id": state.info.get("current_node_id"),
        }
        if q_values is not None:
            ordered = np.sort(q_values)
            decision["q_value"] = float(q_values[position])
            decision["q_margin"] = (
                float(ordered[-1] - ordered[-2]) if ordered.size > 1 else None
            )
            decision["q_ties_1e7"] = int(np.count_nonzero(q_values >= q_values.max() - 1e-7))
        decisions.append(decision)
        transition = environment.step(action)
        actions.append(action)
        reward += transition.reward
        state = environment.current_state

    action_bytes = np.asarray(actions, dtype=np.int64).tobytes()
    result = {
        "method": args.method,
        "instance": str(args.instance),
        "seed": args.seed,
        "policy_seed": args.policy_seed,
        "status": str(state.info.get("status", "unknown")),
        "nodes": int(state.info.get("node_count", 0)),
        "solving_time": float(state.info.get("solving_time", 0.0)),
        "reward": reward,
        "transitions": len(actions),
        "root_fingerprint": root_fingerprint,
        "first_actions": actions[:20],
        "first_decisions": decisions[:20],
        "action_trace_sha256": hashlib.sha256(action_bytes).hexdigest(),
    }
    environment.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
