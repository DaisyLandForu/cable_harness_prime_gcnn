from pathlib import Path

from rl_branching.config import BBMDPConfig
from rl_branching.scip_profile import (
    FORBIDDEN_PRODUCTION_KEYS,
    canonicalize_profile_value,
    ecole_params_from_profile,
    parse_scip_set,
    profile_dump,
    resolve_scip_profile,
    sha256_file,
)


PROFILE = Path("configs/scip/project-production-v1.set")


def test_production_profile_has_no_training_overrides():
    entries = parse_scip_set(PROFILE)
    names = {name for name, _ in entries}
    assert names.isdisjoint(FORBIDDEN_PRODUCTION_KEYS)
    assert "branching/preferbinary" in names
    assert "parallel/maxnthreads" in names
    assert "heuristics/rens/freq" in names
    params = ecole_params_from_profile(PROFILE)
    assert params["parallel/minnthreads"] == 1
    assert params["parallel/maxnthreads"] == 1
    assert params["lp/threads"] == 1
    assert params["branching/preferbinary"] is True
    assert params["heuristics/rens/freq"] == 50
    assert params["heuristics/alns/priority"] == 90000


def test_bbmdp_config_loads_production_profile():
    config = BBMDPConfig(seed=7, node_limit=3)
    assert Path(config.scip_profile) == resolve_scip_profile()
    parameters = config.scip_parameters()
    for key in FORBIDDEN_PRODUCTION_KEYS:
        assert key not in parameters
    assert parameters["parallel/maxnthreads"] == 1
    assert parameters["lp/threads"] == 1
    assert parameters["randomization/randomseedshift"] == 7
    assert parameters["limits/nodes"] == 3
    assert parameters["branching/preferbinary"] is True


def test_profile_dump_is_canonical_and_hashed():
    entries = parse_scip_set(PROFILE)
    dump = profile_dump(entries)
    assert "branching/preferbinary = TRUE" in dump
    assert "limits/gap = 0" in dump
    assert canonicalize_profile_value("TRUE") == "TRUE"
    assert canonicalize_profile_value("0") == "0"
    assert len(sha256_file(PROFILE)) == 64
