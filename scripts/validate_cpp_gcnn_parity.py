#!/usr/bin/env python3
import argparse
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import torch


def write_fixture(path: Path, data) -> None:
    arrays = {
        "row_features": np.asarray(data["row_features"], dtype="<f4"),
        "variable_features": np.asarray(data["variable_features"], dtype="<f4"),
        "edge_indices": np.asarray(data["edge_indices"], dtype="<i8"),
        "edge_features": np.asarray(data["edge_features"], dtype="<f4"),
        "global_features": np.asarray(data["global_features"], dtype="<f4"),
        "variable_categories": np.asarray(data["variable_categories"], dtype="<f4"),
        "row_categories": np.asarray(data["row_categories"], dtype="<f4"),
        "candidate_indices": np.asarray(data["candidate_indices"], dtype="<i8"),
    }
    with path.open("wb") as stream:
        stream.write(b"GCNNP001")
        stream.write(
            struct.pack(
                "<QQQQ",
                arrays["row_features"].shape[0],
                arrays["variable_features"].shape[0],
                arrays["edge_features"].shape[0],
                arrays["candidate_indices"].size,
            )
        )
        for values in (
            arrays["row_features"],
            arrays["variable_features"],
            arrays["edge_indices"][0],
            arrays["edge_indices"][1],
            arrays["edge_features"],
            arrays["global_features"],
            arrays["variable_categories"],
            arrays["row_categories"],
            arrays["candidate_indices"],
        ):
            stream.write(np.ascontiguousarray(values).tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.artifact_dir / "parity_observation.npz")
    fixture = args.output_dir / f"gcnn_observation_{args.device}.bin"
    cpp_output = args.output_dir / f"cpp_q_{args.device}.csv"
    python_output = args.output_dir / f"python_q_{args.device}.csv"
    write_fixture(fixture, data)

    device = torch.device(args.device)
    model_path = args.artifact_dir / "best_model_scripted.pt"
    model = torch.jit.load(str(model_path), map_location=device).eval()
    tensors = tuple(
        torch.tensor(data[name], device=device)
        for name in (
            "row_features",
            "variable_features",
            "edge_indices",
            "edge_features",
            "global_features",
            "variable_categories",
            "row_categories",
            "candidate_indices",
        )
    )
    with torch.no_grad():
        python_q = model(*tensors).cpu().numpy()
    np.savetxt(python_output, python_q, fmt="%.17g")
    subprocess.run(
        [str(args.binary), str(model_path), str(fixture), args.device, str(cpp_output)],
        check=True,
    )
    cpp_q = np.loadtxt(cpp_output, dtype=np.float32, ndmin=1)
    maximum_error = float(np.max(np.abs(python_q - cpp_q)))
    result = {
        "device": args.device,
        "candidate_count": int(python_q.size),
        "max_abs_error": maximum_error,
        "tolerance": args.tolerance,
        "python_argmax_position": int(np.argmax(python_q)),
        "cpp_argmax_position": int(np.argmax(cpp_q)),
        "argmax_equal": int(np.argmax(python_q)) == int(np.argmax(cpp_q)),
        "passed": maximum_error <= args.tolerance
        and int(np.argmax(python_q)) == int(np.argmax(cpp_q)),
    }
    (args.output_dir / f"summary_{args.device}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
