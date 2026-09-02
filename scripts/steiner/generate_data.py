#!/usr/bin/env python3
"""Generate a deterministic synthetic Steiner dataset and canonical manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

from steiner_branching.data.generate import SyntheticDatasetConfig, generate_graph
from steiner_branching.data.manifest import DatasetManifest, InstanceRecord, write_manifest
from steiner_branching.data.split import split_for_synthetic_seed
from steiner_branching.data.write import write_stp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/steiner/data/synthetic_v1.yml"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SyntheticDatasetConfig.from_yaml(args.config)
    records: list[InstanceRecord] = []
    for specification in config.instances:
        graph = generate_graph(specification.generator_config())
        split = split_for_synthetic_seed(specification.seed)
        relative_path = Path(split) / f"{specification.instance_id}.stp"
        write_stp(graph, args.output_root / relative_path)
        records.append(
            InstanceRecord(
                instance_id=specification.instance_id,
                base_lineage=f"synthetic:{specification.family}:{specification.seed}",
                split=split,
                source=graph.source,
                source_sha256=graph.source_sha256,
                graph_sha256=graph.graph_sha256,
                relative_path=relative_path.as_posix(),
            )
        )
    manifest = DatasetManifest(
        schema_version=1,
        manifest_id=config.dataset_id,
        records=tuple(records),
    )
    manifest_path = write_manifest(manifest, args.output_root / "manifest.json")
    print(
        json.dumps(
            {
                "instance_count": len(records),
                "manifest": str(manifest_path),
                "manifest_sha256": manifest.sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
