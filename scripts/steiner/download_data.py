#!/usr/bin/env python3
"""Download only preregistered PACE odd development instances."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

from steiner_branching.data.download import download_pace_development


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pace-track", type=int, default=1, choices=(1, 2))
    parser.add_argument("--instances", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/steiner/raw/pace2018-development"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = download_pace_development(
        track=args.pace_track,
        instance_numbers=tuple(args.instances),
        destination=args.destination,
    )
    print(
        json.dumps(
            {
                "destination": str(args.destination),
                "records": [asdict(record) for record in records],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
