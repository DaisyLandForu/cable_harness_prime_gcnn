#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from rl_branching.trainer import train_candidate_mlp


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the phase-5 Candidate MLP-DQN")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    summary = train_candidate_mlp(args.config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
