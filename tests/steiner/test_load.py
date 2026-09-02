from pathlib import Path

import pytest

from steiner_branching.data.load import load_graph
from steiner_branching.data.steinlib import UnsupportedSteinerFormat


def test_load_graph_dispatches_only_frozen_extensions(tmp_path: Path):
    assert load_graph("tests/steiner/fixtures/path.stp").name == "path"
    assert load_graph("tests/steiner/fixtures/path.gr").name == "path"
    with pytest.raises(UnsupportedSteinerFormat, match="extension"):
        load_graph(tmp_path / "instance.txt")
