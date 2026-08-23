from pathlib import Path

from rl_branching.config import BBMDPConfig
from rl_branching.scip_profile import (
    EFFECTIVE_SEARCH_PARAM_NAMES,
    FORBIDDEN_PRODUCTION_KEYS,
    canonicalize_live_param,
    canonicalize_profile_value,
    dump_effective_search_params,
    ecole_params_from_profile,
    parse_scip_set,
    profile_dump,
    resolve_scip_profile,
    sha256_file,
    sha256_text,
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
    assert parameters["randomization/permuteconss"] is True
    assert parameters["randomization/permutevars"] is True
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
    assert "separating/maxrounds" in EFFECTIVE_SEARCH_PARAM_NAMES
    assert "estimation/restarts/restartpolicy" in EFFECTIVE_SEARCH_PARAM_NAMES
    assert "randomization/permutevars" in EFFECTIVE_SEARCH_PARAM_NAMES
    assert "randomization/permuteconss" in EFFECTIVE_SEARCH_PARAM_NAMES
    assert canonicalize_live_param("c") == "'c'"
    applied = profile_dump(entries)
    assert sha256_text(applied) == "ffec5443d40f7e92f1c547d345206054c3cfd3a88dda04322df4f5aa38bc0741"


def test_effective_dump_reads_the_same_key_set():
    values = {name: 1 for name in EFFECTIVE_SEARCH_PARAM_NAMES}
    values["branching/preferbinary"] = True
    values["randomization/permuteconss"] = True
    values["randomization/permutevars"] = True
    values["estimation/restarts/restartpolicy"] = "c"
    values["limits/time"] = 3600.0
    values["limits/nodes"] = -1
    values["limits/restarts"] = 2147483647
    dump = dump_effective_search_params(values.__getitem__)
    assert "limits/time = 3600" in dump
    assert "estimation/restarts/restartpolicy = 'c'" in dump
    assert dump == dump_effective_search_params(values.__getitem__)
