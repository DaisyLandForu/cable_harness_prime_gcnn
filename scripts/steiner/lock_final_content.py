#!/usr/bin/env python3
"""Hash sealed final-test bytes without parsing or solving any instance."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import tarfile
import time
from typing import Any
import urllib.request

import yaml


PACE_RAW = "https://raw.githubusercontent.com/PACE-challenge/SteinerTree-PACE-2018-instances"
STEINLIB_ARCHIVE = "https://steinlib.zib.de/download/{family}.tgz"
DIMACS_ARCHIVE = "https://dimacs11.zib.de/contest/instances/SPG.tgz"
SOURCE_NOTICES = (
    (
        "pace2018",
        "license",
        f"{PACE_RAW}/4df73cea9c311faea7d03e6d6bffa8733c34a1aa/LICENSE",
    ),
    ("steinlib", "source_notice_not_explicit_license", "https://steinlib.zib.de/testset.php"),
    (
        "dimacs11",
        "source_notice_not_explicit_license",
        "https://dimacs11.zib.de/competition.html",
    ),
)
EXPECTED_STEINLIB_COUNTS = {"D": 20, "E": 20, "I320": 100, "2R": 27, "DIW": 21}
EXPECTED_DIMACS_COUNT = 50


@dataclass(frozen=True)
class ContentRecord:
    relative_path: str
    sha256: str
    size_bytes: int


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _fetch(url: str, *, cache_dir: Path | None) -> bytes:
    cache_path = None
    if cache_dir is not None:
        cache_path = cache_dir / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.bin"
        if cache_path.is_file():
            return cache_path.read_bytes()
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "steiner-content-lock/1"})
            with urllib.request.urlopen(request, timeout=120) as response:
                content = response.read()
            if not content:
                raise RuntimeError(f"empty response from {url}")
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".tmp")
                temporary.write_bytes(content)
                temporary.replace(cache_path)
            return content
        except Exception as error:  # network failures are retried, then surfaced
            last_error = error
            if attempt < 2:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"failed to download {url}: {last_error}") from last_error


def _record(relative_path: str, content: bytes) -> ContentRecord:
    return ContentRecord(
        relative_path=relative_path,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )


def _stp_members(archive: bytes, *, expected_count: int) -> tuple[ContentRecord, ...]:
    records: list[ContentRecord] = []
    with tarfile.open(fileobj=BytesIO(archive), mode="r:gz") as stream:
        for member in stream.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe archive member path: {member.name!r}")
            if member.issym() or member.islnk():
                raise ValueError(f"archive links are not permitted: {member.name!r}")
            if not member.isfile() or path.suffix.lower() != ".stp":
                continue
            extracted = stream.extractfile(member)
            if extracted is None:
                raise ValueError(f"could not read archive member: {member.name!r}")
            records.append(_record(path.as_posix(), extracted.read()))
    records.sort(key=lambda item: item.relative_path)
    if len(records) != expected_count:
        raise ValueError(
            f"archive instance count mismatch: expected={expected_count}, actual={len(records)}"
        )
    if len({record.relative_path for record in records}) != len(records):
        raise ValueError("archive contains duplicate instance paths")
    return tuple(records)


def _pace_suite(suite: dict[str, Any], *, cache_dir: Path | None) -> dict[str, Any]:
    selector = suite["selector"]
    revision = suite["source_revision"]
    requests: list[tuple[str, str]] = []
    for number in range(selector["start"], selector["stop"] + 1, selector["step"]):
        if number % 2 != 0:
            raise ValueError("final PACE content lock must contain only even selectors")
        relative = selector["path_template"].format(number=number)
        requests.append((relative, f"{PACE_RAW}/{revision}/{relative}"))
    with ThreadPoolExecutor(max_workers=8) as executor:
        contents = tuple(
            executor.map(
                lambda request: _fetch(request[1], cache_dir=cache_dir),
                requests,
            )
        )
    records = [
        _record(relative, content)
        for (relative, _), content in zip(requests, contents, strict=True)
    ]
    rendered = [asdict(record) for record in records]
    return {
        "suite_id": suite["suite_id"],
        "source_revision": revision,
        "distribution_kind": "revision_pinned_files",
        "instance_count": len(records),
        "aggregate_members_sha256": _content_sha256(rendered),
        "members": rendered,
    }


def _steinlib_suite(suite: dict[str, Any], *, cache_dir: Path | None) -> dict[str, Any]:
    archives: list[dict[str, Any]] = []
    total = 0
    for family in suite["selector"]["families"]:
        if family not in EXPECTED_STEINLIB_COUNTS:
            raise ValueError(f"unexpected sealed SteinLib family: {family}")
        url = STEINLIB_ARCHIVE.format(family=family)
        content = _fetch(url, cache_dir=cache_dir)
        members = _stp_members(content, expected_count=EXPECTED_STEINLIB_COUNTS[family])
        rendered = [asdict(record) for record in members]
        total += len(members)
        archives.append(
            {
                "family": family,
                "source_url": url,
                "archive_sha256": hashlib.sha256(content).hexdigest(),
                "archive_size_bytes": len(content),
                "instance_count": len(members),
                "aggregate_members_sha256": _content_sha256(rendered),
                "members": rendered,
            }
        )
    return {
        "suite_id": suite["suite_id"],
        "source_revision": suite["source_revision"],
        "distribution_kind": "family_archives",
        "instance_count": total,
        "archives": archives,
    }


def _dimacs_suite(suite: dict[str, Any], *, cache_dir: Path | None) -> dict[str, Any]:
    content = _fetch(DIMACS_ARCHIVE, cache_dir=cache_dir)
    members = _stp_members(content, expected_count=EXPECTED_DIMACS_COUNT)
    rendered = [asdict(record) for record in members]
    return {
        "suite_id": suite["suite_id"],
        "source_revision": suite["source_revision"],
        "distribution_kind": "complete_archive",
        "source_url": DIMACS_ARCHIVE,
        "archive_sha256": hashlib.sha256(content).hexdigest(),
        "archive_size_bytes": len(content),
        "instance_count": len(members),
        "aggregate_members_sha256": _content_sha256(rendered),
        "members": rendered,
    }


def build_lock(manifest_path: Path, *, cache_dir: Path | None) -> dict[str, Any]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or not manifest.get("sealed")
        or manifest.get("learning_runs_total") != 0
        or manifest.get("result_artifacts") != []
    ):
        raise ValueError("final selector manifest must be sealed and unrun")
    suites: list[dict[str, Any]] = []
    for suite in manifest["suites"]:
        if suite["status"] != "sealed" or suite["tuning_allowed"] or suite["learning_runs"] != 0:
            raise ValueError(f"suite is not sealed and unrun: {suite['suite_id']}")
        kind = suite["selector"]["kind"]
        if kind == "numeric_sequence":
            suites.append(_pace_suite(suite, cache_dir=cache_dir))
        elif kind == "complete_families":
            suites.append(_steinlib_suite(suite, cache_dir=cache_dir))
        elif kind == "complete_archive":
            suites.append(_dimacs_suite(suite, cache_dir=cache_dir))
        else:
            raise ValueError(f"unsupported final selector kind: {kind}")
    source_notices = []
    for source_id, notice_kind, url in SOURCE_NOTICES:
        content = _fetch(url, cache_dir=cache_dir)
        source_notices.append(
            {
                "source_id": source_id,
                "notice_kind": notice_kind,
                "source_url": url,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        )
    return {
        "schema_version": 1,
        "content_lock_id": "steiner-spg-final-content-v1",
        "selector_manifest_id": manifest["manifest_id"],
        "canonical_entries_sha256": manifest["canonical_entries_sha256"],
        "locked_stage": "S02",
        "operation": "byte_hash_only_no_parse_no_solve",
        "learning_runs_total": 0,
        "source_notices": source_notices,
        "suites": suites,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/steiner/splits/final_test_v1.yml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock = build_lock(args.manifest, cache_dir=args.cache_dir)
    rendered = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "content_lock_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                "instance_count": sum(suite["instance_count"] for suite in lock["suites"]),
                "operation": lock["operation"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
