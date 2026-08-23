"""DSU-Prime local-bound and six-dimensional feature tests."""

from pathlib import Path
import json
import subprocess

import numpy as np
from pyscipopt import SCIP_PARAMSETTING, Branchrule, Model, SCIP_RESULT

from rl_branching.prim_bias import (
    PRIM_VARIABLE_FEATURE_NAMES,
    build_dsu_layers,
    dsu_feature_vector_for_name,
    parse_z,
    prim_variable_feature_matrix,
)
from rl_branching.scip_profile import load_production_scip_params

FIXTURE_CIP = Path("results/probes/dsu_sixdim.cip")
FIXTURE_SET = Path("results/probes/dsu_sixdim.set")
FIXTURE_CPP = Path("results/probes/dsu_sixdim_cpp_first_state.json")
GRAPH_PROBE = Path("build/graph_probe")
# Keep production cuts on; only freeze the tiny MIP so SCIP must branch.
FIXTURE_OVERRIDES = """
presolving/maxrounds = 0
misc/usesymmetry = 0
heuristics/adaptivediving/freq = -1
heuristics/alns/freq = -1
heuristics/clique/freq = -1
heuristics/completesol/freq = -1
heuristics/conflictdiving/freq = -1
heuristics/crossover/freq = -1
heuristics/distributiondiving/freq = -1
heuristics/farkasdiving/freq = -1
heuristics/feaspump/freq = -1
heuristics/fracdiving/freq = -1
heuristics/gins/freq = -1
heuristics/guideddiving/freq = -1
heuristics/indicator/freq = -1
heuristics/intshifting/freq = -1
heuristics/linesearchdiving/freq = -1
heuristics/locks/freq = -1
heuristics/lpface/freq = -1
heuristics/mpec/freq = -1
heuristics/multistart/freq = -1
heuristics/nlpdiving/freq = -1
heuristics/objpscostdiving/freq = -1
heuristics/ofins/freq = -1
heuristics/oneopt/freq = -1
heuristics/padm/freq = -1
heuristics/pscostdiving/freq = -1
heuristics/randrounding/freq = -1
heuristics/rens/freq = -1
heuristics/reoptsols/freq = -1
heuristics/rins/freq = -1
heuristics/rootsoldiving/freq = -1
heuristics/rounding/freq = -1
heuristics/shiftandpropagate/freq = -1
heuristics/shifting/freq = -1
heuristics/simplerounding/freq = -1
heuristics/subnlp/freq = -1
heuristics/trivial/freq = -1
heuristics/trivialnegation/freq = -1
heuristics/trysol/freq = -1
heuristics/undercover/freq = -1
heuristics/vbounds/freq = -1
heuristics/veclendiving/freq = -1
heuristics/zirounding/freq = -1
constraints/SOS1/sepafreq = -1
constraints/SOS2/sepafreq = -1
constraints/and/sepafreq = -1
constraints/cardinality/sepafreq = -1
constraints/cumulative/sepafreq = -1
constraints/indicator/sepafreq = -1
constraints/knapsack/sepafreq = -1
constraints/linear/sepafreq = -1
constraints/linking/sepafreq = -1
constraints/logicor/sepafreq = -1
constraints/nonlinear/sepafreq = -1
constraints/or/sepafreq = -1
constraints/orbisack/sepafreq = -1
constraints/setppc/sepafreq = -1
constraints/symresack/sepafreq = -1
constraints/varbound/sepafreq = -1
constraints/xor/sepafreq = -1
separating/aggregation/freq = -1
separating/clique/freq = -1
separating/cmir/freq = -1
separating/disjunctive/freq = -1
separating/flowcover/freq = -1
separating/gomory/freq = -1
separating/gomorymi/freq = -1
separating/impliedbounds/freq = -1
separating/knapsackcover/freq = -1
separating/mcf/freq = -1
separating/minor/freq = -1
separating/mixing/freq = -1
separating/rapidlearning/freq = -1
separating/rlt/freq = -1
separating/strongcg/freq = -1
separating/zerohalf/freq = -1
"""


def test_dsu_ignores_lp_and_uses_local_lb_only():
    names = ["z_1_2_0", "z_2_3_0"]
    layers = build_dsu_layers(names, [0.0, 1.0])
    assert 0 in layers
    assert layers[0].contains(2) and layers[0].contains(3)
    assert not layers[0].contains(1)


def test_dsu_frontier_merge_cycle_unseen_and_nonzero_ratios():
    names = ["z_1_2_0", "z_3_4_0", "z_2_3_0", "z_5_6_0", "z_1_9_0", "m_1_0"]
    layers = build_dsu_layers(names, [1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    cycle = dsu_feature_vector_for_name("z_1_2_0", layers)
    merge = dsu_feature_vector_for_name("z_2_3_0", layers)
    unseen = dsu_feature_vector_for_name("z_5_6_0", layers)
    frontier = dsu_feature_vector_for_name("z_1_9_0", layers)
    other = dsu_feature_vector_for_name("m_1_0", layers)
    assert cycle[:4] == (0.0, 0.0, 1.0, 0.0)
    assert merge[:4] == (0.0, 1.0, 0.0, 0.0)
    assert unseen[:4] == (0.0, 0.0, 0.0, 1.0)
    assert frontier[:4] == (1.0, 0.0, 0.0, 0.0)
    assert other == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert cycle[4] == cycle[5] == 0.5
    assert merge[4] == merge[5] == 0.5
    assert frontier[4] == 0.5 and frontier[5] == 0.0
    assert unseen[4] == unseen[5] == 0.0


class _StopFirst(Branchrule):
    def branchexeclp(self, allowaddcons):
        self.model.interruptSolve()
        return {"result": SCIP_RESULT.DIDNOTRUN}


def _write_dsu_fixture(path: Path) -> None:
    model = Model()
    model.hideOutput()
    model.addVar(vtype="B", name="z_1_2_0", lb=1.0, ub=1.0)
    model.addVar(vtype="B", name="z_3_4_0", lb=1.0, ub=1.0)
    z23 = model.addVar(vtype="B", name="z_2_3_0")
    z56 = model.addVar(vtype="B", name="z_5_6_0")
    z19 = model.addVar(vtype="B", name="z_1_9_0")
    m10 = model.addVar(vtype="B", name="m_1_0")
    xa = model.addVar(vtype="B", name="x_a")
    xb = model.addVar(vtype="B", name="x_b")
    xc = model.addVar(vtype="B", name="x_c")
    model.addCons(xa + xb + xc >= 1.5)
    model.setObjective(xa + 1.5 * xb + 1.25 * xc, "minimize")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.writeProblem(str(path))


def _python_dsu_by_name(cip: Path) -> dict[str, np.ndarray]:
    model = Model()
    model.hideOutput()
    model.readProblem(str(cip))
    for name, value in load_production_scip_params(seed=0, time_limit=10.0, node_limit=-1).items():
        model.setParam(name, value)
    model.setHeuristics(SCIP_PARAMSETTING.OFF)
    model.setSeparating(SCIP_PARAMSETTING.OFF)
    model.setParam("presolving/maxrounds", 0)
    model.setParam("misc/usesymmetry", 0)
    model.includeBranchrule(_StopFirst(), "stopfirst", "stop", 1000000, -1, 1.0)
    model.optimize()
    variables = model.getVars(transformed=True)
    names = [str(variable.name) for variable in variables]
    lower_bounds = [float(variable.getLbLocal()) for variable in variables]
    matrix = prim_variable_feature_matrix(names, lower_bounds=lower_bounds)
    keyed = {}
    for name, row in zip(names, matrix):
        key = name[2:] if name.startswith("t_") else name
        keyed[key] = row
    assert any(bound > 0.5 for bound, name in zip(lower_bounds, names) if parse_z(name))
    return keyed


def test_dsu_sixdim_scip_fixture_matches_cpp_extractor():
    if not GRAPH_PROBE.is_file():
        raise AssertionError("build/graph_probe is required for DSU fixture parity")
    _write_dsu_fixture(FIXTURE_CIP)
    FIXTURE_SET.write_text(
        Path("configs/scip/project-production-v1.set").read_text(encoding="utf-8")
        + "\n"
        + FIXTURE_OVERRIDES,
        encoding="utf-8",
    )
    python_rows = _python_dsu_by_name(FIXTURE_CIP)
    subprocess.run(
        [
            str(GRAPH_PROBE),
            "--instance",
            str(FIXTURE_CIP),
            "--scip-profile",
            str(FIXTURE_SET),
            "--output",
            str(FIXTURE_CPP),
            "--dump-state",
            "--time-limit",
            "10",
            "--seed",
            "0",
            "--threads",
            "1",
        ],
        check=True,
    )
    cpp = json.loads(FIXTURE_CPP.read_text())
    names = cpp["full"]["variable_names"]
    features = np.asarray(cpp["full"]["variable_features"], dtype=np.float32).reshape(
        int(cpp["full"]["variable_count"]), 25
    )
    cpp_rows = {}
    for name, row in zip(names, features[:, 19:25]):
        key = name[2:] if name.startswith("t_") else name
        cpp_rows[key] = row
    expected = {
        "z_1_2_0": (0.0, 0.0, 1.0, 0.0, 0.5, 0.5),
        "z_3_4_0": (0.0, 0.0, 1.0, 0.0, 0.5, 0.5),
        "z_2_3_0": (0.0, 1.0, 0.0, 0.0, 0.5, 0.5),
        "z_5_6_0": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        "z_1_9_0": (1.0, 0.0, 0.0, 0.0, 0.5, 0.0),
        "m_1_0": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    }
    for name, vector in expected.items():
        assert name in python_rows and name in cpp_rows
        assert np.allclose(python_rows[name], vector, atol=1.0e-6)
        assert np.allclose(cpp_rows[name], vector, atol=1.0e-6)
    assert list(PRIM_VARIABLE_FEATURE_NAMES) == [
        "prim_frontier",
        "prim_merge",
        "prim_cycle",
        "prim_unseen",
        "prim_src_component_ratio",
        "prim_dst_component_ratio",
    ]
