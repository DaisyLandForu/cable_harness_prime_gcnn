from pathlib import Path

import pytest

from steiner_branching.data.manifest import (
    DatasetManifest,
    InstanceRecord,
    read_manifest,
    write_manifest,
)
from steiner_branching.data.split import split_for_synthetic_seed


def record(instance_id: str, lineage: str, split: str) -> InstanceRecord:
    return InstanceRecord(
        instance_id=instance_id,
        base_lineage=lineage,
        split=split,
        source="synthetic",
        source_sha256="a" * 64,
        graph_sha256="b" * 64,
        relative_path=f"{instance_id}.stp",
    )


def test_frozen_seed_ranges_are_exact():
    assert split_for_synthetic_seed(100000) == "train"
    assert split_for_synthetic_seed(200000) == "validation_iid"
    assert split_for_synthetic_seed(300000) == "test_iid"
    assert split_for_synthetic_seed(400000) == "development_ood"
    assert split_for_synthetic_seed(500000) == "test_ood"
    with pytest.raises(ValueError):
        split_for_synthetic_seed(99999)


def test_manifest_roundtrip_hash_and_lineage_guard(tmp_path: Path):
    manifest = DatasetManifest(
        schema_version=1,
        manifest_id="toy-manifest",
        records=(record("a", "lineage-a", "train"), record("b", "lineage-b", "validation_iid")),
    )
    output = write_manifest(manifest, tmp_path / "manifest.json")
    assert read_manifest(output) == manifest
    assert read_manifest(output).sha256 == manifest.sha256
    with pytest.raises(ValueError, match="crosses splits"):
        DatasetManifest(
            schema_version=1,
            manifest_id="leak",
            records=(record("a", "same", "train"), record("b", "same", "validation_iid")),
        )
