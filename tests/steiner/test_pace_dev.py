from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

pytest.importorskip("pyscipopt")

from steiner_branching.data.pace import parse_pace
from steiner_branching.milp.mcf import build_mcf
from steiner_branching.milp.validate import solve_and_validate


PACE_ROOT = os.environ.get("STEINER_PACE_DEV_ROOT")


@pytest.mark.skipif(not PACE_ROOT, reason="PACE odd development data path not provided")
def test_pace_track1_instance001_matches_published_optimum():
    root = Path(PACE_ROOT)
    graph = parse_pace(root / "Track1/instance001.gr")
    with (root / "track1.csv").open(newline="", encoding="utf-8") as stream:
        rows = {row["paceName"].strip(): float(row["opt"]) for row in csv.DictReader(stream)}
    expected = rows["instance001.gr"]
    result = solve_and_validate(build_mcf(graph))
    assert result.status == "optimal"
    assert result.objective == pytest.approx(expected)
    assert result.check is not None and result.check.feasible
