#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from rl_branching.gcnn_trainer import build_shared_normalization, train_gcnn


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the bipartite GCNN-DQN")
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--build-normalization",
        action="store_true",
        help="Build the shared read-only normalization.json and exit",
    )
    parser.add_argument(
        "--normalization-output",
        type=Path,
        default=Path("results/probes/shared_normalization.json"),
    )
    parser.add_argument("--states-per-instance", type=int, default=2)
    args = parser.parse_args()
    if args.build_normalization:
        print(
            json.dumps(
                build_shared_normalization(
                    args.normalization_output,
                    states_per_instance=args.states_per_instance,
                ),
                indent=2,
            )
        )
        return
    if args.config is None:
        parser.error("--config is required unless --build-normalization")
    print(json.dumps(train_gcnn(args.config), indent=2))


if __name__ == "__main__":
    main()
