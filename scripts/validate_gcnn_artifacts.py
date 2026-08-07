#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from rl_branching.gcnn_config import GCNNTrainingConfig
from rl_branching.gcnn_dqn import stable_graph_argmax
from rl_branching.gcnn_trainer import load_gcnn_model
from rl_branching.graph_features import GraphState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    args = parser.parse_args()
    config = GCNNTrainingConfig.from_yaml(args.artifact_dir / "config.yaml")
    device = torch.device(args.device)
    model = load_gcnn_model(args.artifact_dir / "best_model.pt", config, device)
    scripted = torch.jit.load(
        str(args.artifact_dir / "best_model_scripted.pt"), map_location=device
    ).eval()
    fixture = np.load(args.artifact_dir / "parity_observation.npz")
    state = GraphState(
        row_features=fixture["row_features"],
        variable_features=fixture["variable_features"],
        edge_indices=fixture["edge_indices"],
        edge_features=fixture["edge_features"],
        global_features=fixture["global_features"],
        variable_categories=fixture["variable_categories"],
        row_categories=fixture["row_categories"],
        actions=fixture["candidate_indices"],
        candidate_names=tuple(str(value) for value in fixture["candidate_names"]),
    )
    tensors = tuple(
        torch.tensor(value, device=device)
        for value in (
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
        eager_q = model(*tensors).cpu().numpy()
        scripted_q = scripted(*tensors).cpu().numpy()
    maximum_error = float(np.max(np.abs(eager_q - scripted_q)))
    eager_argmax = stable_graph_argmax(eager_q, state)
    scripted_argmax = stable_graph_argmax(scripted_q, state)
    with (args.artifact_dir / "training_history.csv").open(encoding="utf-8") as stream:
        updates = [row for row in csv.DictReader(stream) if row["event"] == "train_update"]
    finite_losses = bool(updates) and all(
        np.isfinite(float(row["loss"])) and np.isfinite(float(row["td_error"]))
        for row in updates
    )
    result = {
        "device": args.device,
        "candidate_count": state.candidate_count,
        "max_abs_error": maximum_error,
        "tolerance": args.tolerance,
        "eager_argmax_position": eager_argmax,
        "scripted_argmax_position": scripted_argmax,
        "argmax_equal": eager_argmax == scripted_argmax,
        "finite_training_losses": finite_losses,
        "passed": maximum_error <= args.tolerance
        and eager_argmax == scripted_argmax
        and finite_losses,
    }
    (args.artifact_dir / "parity.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
