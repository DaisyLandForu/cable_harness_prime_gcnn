"""Governed download helpers for PACE odd development instances only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import urllib.request

from .canonical import sha256_bytes


PACE_REPOSITORY = "https://raw.githubusercontent.com/PACE-challenge/SteinerTree-PACE-2018-instances"
PACE_REVISION = "4df73cea9c311faea7d03e6d6bffa8733c34a1aa"


@dataclass(frozen=True)
class DownloadRecord:
    relative_path: str
    source_url: str
    sha256: str
    size_bytes: int


def validate_pace_development_request(track: int, instance_numbers: tuple[int, ...]) -> None:
    if track not in {1, 2}:
        raise ValueError("PACE development downloader supports Track 1 or 2")
    if not instance_numbers:
        raise ValueError("at least one instance number is required")
    if len(set(instance_numbers)) != len(instance_numbers):
        raise ValueError("duplicate PACE instance number")
    for number in instance_numbers:
        if isinstance(number, bool) or not isinstance(number, int):
            raise ValueError("PACE instance number must be an integer")
        if number < 1 or number > 100:
            raise ValueError("PACE instance number must be in 1..100")
        if number % 2 == 0:
            raise ValueError("PACE even instances are sealed final test and cannot be downloaded here")


def _download(url: str, destination: Path, *, relative_path: str) -> DownloadRecord:
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(destination)
    return DownloadRecord(
        relative_path=relative_path,
        source_url=url,
        sha256=sha256_bytes(content),
        size_bytes=len(content),
    )


def download_pace_development(
    *, track: int, instance_numbers: tuple[int, ...], destination: Path | str
) -> tuple[DownloadRecord, ...]:
    validate_pace_development_request(track, instance_numbers)
    root = Path(destination)
    records: list[DownloadRecord] = []
    for number in sorted(instance_numbers):
        relative = f"Track{track}/instance{number:03d}.gr"
        url = f"{PACE_REPOSITORY}/{PACE_REVISION}/{relative}"
        records.append(_download(url, root / relative, relative_path=relative))
    for relative in (f"track{track}.csv", "LICENSE"):
        url = f"{PACE_REPOSITORY}/{PACE_REVISION}/{relative}"
        records.append(_download(url, root / relative, relative_path=relative))
    manifest = {
        "schema_version": 1,
        "dataset": f"PACE2018-Track{track}-odd-development",
        "source_revision": PACE_REVISION,
        "records": [asdict(record) for record in records],
    }
    manifest_path = root / "download_manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    return tuple(records)
