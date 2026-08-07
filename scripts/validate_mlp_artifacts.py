#!/usr/bin/env python3
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from rl_branching.candidate_features import extract_candidate_state
from rl_branching.candidate_model import CandidateQNetwork
from rl_branching.environment import BBMDPBranchingEnv
from rl_branching.training_config import MLPTrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Candidate MLP checkpoints and TorchScript parity")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--instance", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    required = (
        "best_model.pt",
        "best_model_scripted.pt",
        "last_model.pt",
        "last_model_scripted.pt",
        "config.yaml",
        "feature_schema.json",
        "normalization.json",
        "training_history.csv",
        "evaluation.csv",
        "summary.json",
    )
    missing = [name for name in required if not (args.artifact_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing artifacts: {missing}")

    config = MLPTrainingConfig.from_yaml(args.artifact_dir / "config.yaml")
    checkpoint = torch.load(args.artifact_dir / "best_model.pt", map_location="cpu", weights_only=False)
    eager = CandidateQNetwork(checkpoint["hidden_sizes"])
    eager.load_state_dict(checkpoint["model_state_dict"])
    eager.eval()
    scripted = torch.jit.load(str(args.artifact_dir / "best_model_scripted.pt"), map_location="cpu").eval()

    env = BBMDPBranchingEnv(replace(config.environment, seed=args.seed, node_limit=2))
    state = env.reset(args.instance)
    compact = extract_candidate_state(state.observation, state.action_set)
    variables = torch.tensor(compact.variable_features, dtype=torch.float32)
    globals_ = torch.tensor(compact.global_features, dtype=torch.float32).expand(compact.candidate_count, -1)
    categories = torch.tensor(compact.category_features, dtype=torch.float32)
    with torch.no_grad():
        eager_q = eager(variables, globals_, categories).numpy()
        scripted_q = scripted(variables, globals_, categories).numpy()
    env.close()

    if not np.isfinite(eager_q).all() or not np.isfinite(scripted_q).all():
        raise FloatingPointError("model output contains NaN or Inf")
    max_error = float(np.max(np.abs(eager_q - scripted_q)))
    if max_error > args.tolerance:
        raise AssertionError(f"TorchScript max error {max_error} exceeds {args.tolerance}")
    if int(np.argmax(eager_q)) != int(np.argmax(scripted_q)):
        raise AssertionError("TorchScript argmax differs from eager model")

    with (args.artifact_dir / "training_history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    losses = np.asarray([float(row["loss"]) for row in rows if row["loss"]], dtype=np.float64)
    if not losses.size or not np.isfinite(losses).all():
        raise FloatingPointError("training history has no finite losses")

    np.savez_compressed(
        args.artifact_dir / "parity_observation.npz",
        variable_features=compact.variable_features,
        global_features=compact.global_features,
        category_features=compact.category_features,
        actions=compact.actions,
        eager_q=eager_q,
        scripted_q=scripted_q,
    )
    result = {
        "instance": str(args.instance),
        "seed": args.seed,
        "candidate_count": compact.candidate_count,
        "max_abs_error": max_error,
        "tolerance": args.tolerance,
        "eager_argmax_position": int(np.argmax(eager_q)),
        "scripted_argmax_position": int(np.argmax(scripted_q)),
        "selected_action": int(compact.actions[int(np.argmax(eager_q))]),
        "loss_rows": int(losses.size),
        "loss_min": float(losses.min()),
        "loss_max": float(losses.max()),
    }
    (args.artifact_dir / "parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
