#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from rl_branching.gcnn_trainer import train_gcnn


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the phase-7 bipartite GCNN-DQN")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(train_gcnn(args.config), indent=2))


if __name__ == "__main__":
    main()
