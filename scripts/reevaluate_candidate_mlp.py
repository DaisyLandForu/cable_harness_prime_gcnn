#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from rl_branching.trainer import reevaluate_candidate_mlp


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate a Candidate MLP with independent policy seeds")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(reevaluate_candidate_mlp(args.artifact_dir), indent=2))


if __name__ == "__main__":
    main()
