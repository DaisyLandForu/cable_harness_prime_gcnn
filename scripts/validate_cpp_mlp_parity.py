#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Python and C++ TorchScript Candidate MLP outputs")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--binary", default=Path("build/model_runner_parity"), type=Path)
    parser.add_argument("--output-dir", default=Path("results/phase6/parity"), type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    observation = np.load(args.artifact_dir / "parity_observation.npz")
    variables = np.asarray(observation["variable_features"], dtype=np.float32)
    globals_ = np.asarray(observation["global_features"], dtype=np.float32)
    categories = np.asarray(observation["category_features"], dtype=np.float32)
    candidate_count = variables.shape[0]
    if variables.shape[1] != 19 or globals_.shape != (14,) or categories.shape != (candidate_count, 6):
        raise ValueError("parity observation does not match schema version 2")

    scripted = torch.jit.load(
        str(args.artifact_dir / "best_model_scripted.pt"), map_location=args.device
    ).eval()
    with torch.no_grad():
        python_q = scripted(
            torch.from_numpy(variables).to(args.device),
            torch.from_numpy(globals_).to(args.device).expand(candidate_count, -1),
            torch.from_numpy(categories).to(args.device),
        ).cpu().numpy()

    fixture_path = args.output_dir / f"observation_{args.device}.txt"
    with fixture_path.open("w") as stream:
        stream.write(f"{candidate_count}\n")
        repeated_globals = np.broadcast_to(globals_, (candidate_count, globals_.size))
        for row in np.concatenate((variables, repeated_globals, categories), axis=1):
            stream.write(" ".join(format(float(value), ".9g") for value in row) + "\n")

    cpp_path = args.output_dir / f"cpp_q_{args.device}.csv"
    command = [
        str(args.binary),
        str(args.artifact_dir / "best_model_scripted.pt"),
        args.device,
        str(fixture_path),
        str(cpp_path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"C++ parity runner failed: {completed.stderr.strip()}")
    cpp_q = np.loadtxt(cpp_path, dtype=np.float32, ndmin=1)
    if cpp_q.shape != python_q.shape:
        raise AssertionError(f"C++ shape {cpp_q.shape} differs from Python {python_q.shape}")

    python_path = args.output_dir / f"python_q_{args.device}.csv"
    with python_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows((float(value),) for value in python_q)
    max_abs_error = float(np.max(np.abs(python_q - cpp_q)))
    python_argmax = int(np.argmax(python_q))
    cpp_argmax = int(np.argmax(cpp_q))
    result = {
        "device": args.device,
        "candidate_count": candidate_count,
        "max_abs_error": max_abs_error,
        "tolerance": args.tolerance,
        "python_argmax_position": python_argmax,
        "cpp_argmax_position": cpp_argmax,
        "argmax_equal": python_argmax == cpp_argmax,
        "python_q_path": str(python_path),
        "cpp_q_path": str(cpp_path),
    }
    if max_abs_error > args.tolerance or python_argmax != cpp_argmax:
        raise AssertionError(json.dumps(result, indent=2))
    (args.output_dir / f"summary_{args.device}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
