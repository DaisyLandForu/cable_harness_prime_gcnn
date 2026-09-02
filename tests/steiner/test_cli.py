from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/steiner"


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )


def test_generate_cli_is_deterministic_and_writes_manifest(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    command = str(SCRIPTS / "generate_data.py")
    one = run_cli(command, "--output-root", str(first))
    two = run_cli(command, "--output-root", str(second))
    assert one.returncode == two.returncode == 0, (one.stderr, two.stderr)
    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest == second_manifest
    assert len(first_manifest["records"]) == 5
    for record in first_manifest["records"]:
        relative = Path(record["relative_path"])
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


@pytest.mark.skipif(
    not os.environ.get("LD_LIBRARY_PATH"),
    reason="locked SCIP shared-library path not configured",
)
def test_build_and_check_clis_emit_valid_outputs(tmp_path: Path):
    lp_path = tmp_path / "path.lp"
    metadata_path = tmp_path / "problem_meta.json"
    build = run_cli(
        str(SCRIPTS / "build_milp.py"),
        "tests/steiner/fixtures/triangle.stp",
        "--lp-output",
        str(lp_path),
        "--metadata-output",
        str(metadata_path),
    )
    assert build.returncode == 0, build.stderr
    assert lp_path.is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["formulation_id"] == "rooted_mcf_v1"
    assert [item["variable_name"] for item in metadata["edge_variables"]] == [
        "stp_x_e00000000",
        "stp_x_e00000001",
        "stp_x_e00000002",
    ]

    checked = run_cli(
        str(SCRIPTS / "check_solution.py"),
        "tests/steiner/fixtures/triangle.stp",
        "--known-objective",
        "2",
    )
    assert checked.returncode == 0, checked.stderr
    result = json.loads(checked.stdout)
    assert result["status"] == "optimal"
    assert result["checker_feasible"]
    assert result["known_objective_matches"]
    assert result["objective"] == pytest.approx(2.0)


def test_download_cli_rejects_sealed_even_selector_before_network(tmp_path: Path):
    result = run_cli(
        str(SCRIPTS / "download_data.py"),
        "--instances",
        "2",
        "--destination",
        str(tmp_path / "must-not-exist"),
    )
    assert result.returncode != 0
    assert "sealed final test" in result.stderr
    assert not (tmp_path / "must-not-exist").exists()
