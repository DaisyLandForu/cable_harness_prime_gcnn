"""DSU-Prime local-bound and six-dimensional feature tests."""

from rl_branching.prim_bias import (
    build_dsu_layers,
    dsu_feature_vector_for_name,
)


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
