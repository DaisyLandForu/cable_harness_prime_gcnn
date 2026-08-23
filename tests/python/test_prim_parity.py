"""C0.3: grown-set / bias-mode parity helpers (Python side)."""

from rl_branching.prim_bias import (
    bias_score_for_name,
    build_grown_sets,
    prim_feature_vector_for_name,
    prim_score_for_name,
)


def test_grown_sets_match_cpp_lb_or_lp():
    names = ["z_1_2_0", "z_2_3_0", "m_1_0"]
    # z_1_2 fixed by lb=1 but lp=0; z_2_3 lp-active
    lp = [0.0, 0.8, 0.0]
    lb = [1.0, 0.0, 0.0]
    grown_lp_only = build_grown_sets(names, lp)
    grown_cpp = build_grown_sets(names, lp, lower_bounds=lb)
    assert grown_lp_only[0] == {2, 3}
    assert grown_cpp[0] == {1, 2, 3}


def test_topology_mode_drops_empty_s_prior():
    grown = {}
    assert prim_score_for_name("z_1_2_0", grown, empty_s_z_prior=True) == 0.5
    assert prim_score_for_name("z_1_2_0", grown, empty_s_z_prior=False) == 0.0
    assert bias_score_for_name("z_1_2_0", grown, mode="topology", depth=0) == 0.0
    assert bias_score_for_name("z_1_2_0", grown, mode="prim", depth=0) == 0.5


def test_z_and_root_z_modes():
    grown = {0: {1, 2}}
    assert bias_score_for_name("z_8_9_0", grown, mode="z", depth=5) == 1.0
    assert bias_score_for_name("m_1_0", grown, mode="z", depth=5) == 0.0
    assert bias_score_for_name("z_8_9_0", grown, mode="root_z", depth=0) == 1.0
    assert bias_score_for_name("z_8_9_0", grown, mode="root_z", depth=1) == 0.0


def test_feature_vector_stable():
    grown = {0: {1, 2}}
    assert prim_feature_vector_for_name("z_2_9_0", grown)[0] == 1.0
    assert prim_feature_vector_for_name("z_1_2_0", grown)[1] == 1.0
