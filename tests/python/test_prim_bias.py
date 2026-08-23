from rl_branching.prim_bias import (
    apply_prim_bias,
    build_grown_sets,
    parse_z,
    prim_feature_vector_for_name,
    prim_score_for_name,
    prim_variable_feature_matrix,
    stable_argmax_with_scores,
)


def test_parse_z_and_grown_sets():
    assert parse_z("z_1_2_0").src == 1
    assert parse_z("t_z_3_4_1").dst == 4
    grown = build_grown_sets(
        ["z_1_2_0", "z_2_5_0", "m_1_0", "z_9_8_0"],
        [1.0, 0.1, 1.0, 0.0],
    )
    assert grown[0] == {1, 2}


def test_prim_scores_prefer_cut_edges():
    grown = {0: {1, 2}}
    assert prim_score_for_name("z_2_9_0", grown) == 1.0
    assert prim_score_for_name("z_1_2_0", grown) == -0.5
    assert prim_score_for_name("z_8_9_0", grown) == 0.25
    assert prim_score_for_name("m_2_0", grown) == 0.3


def test_bias_changes_argmax_when_q_close():
    q = [1.0, 1.01, 0.2]
    names = ["z_8_9_0", "z_2_9_0", "m_7_0"]
    grown = {0: {1, 2}}
    biased, prim = apply_prim_bias(q, names, lambda_prim=0.5, grown=grown)
    assert prim[1] == 1.0
    assert biased[1] > biased[0]
    pos = stable_argmax_with_scores(biased, names, [10, 20, 30])
    assert pos == 1


def test_gating_disables_bias():
    q = [1.0, 1.01]
    names = ["z_8_9_0", "z_2_9_0"]
    grown = {0: {1, 2}}
    biased, prim = apply_prim_bias(
        q, names, lambda_prim=0.5, grown=grown, depth=0, prim_min_depth=1
    )
    assert list(prim) == [0.0, 0.0]
    assert list(biased) == [1.0, 1.01]

    biased2, prim2 = apply_prim_bias(
        q, names, lambda_prim=0.5, grown={}, depth=2, prim_require_grown=True
    )
    assert list(prim2) == [0.0, 0.0]
    assert list(biased2) == [1.0, 1.01]


def test_prim_variable_features():
    grown = {0: {1, 2}}
    assert prim_feature_vector_for_name("z_2_9_0", grown) == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert prim_feature_vector_for_name("z_1_2_0", grown) == (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    assert prim_feature_vector_for_name("z_8_9_0", grown) == (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert prim_feature_vector_for_name("m_2_0", grown)[4] == 1.0
    matrix = prim_variable_feature_matrix(
        ["z_1_2_0", "z_2_9_0"], solution_values=[1.0, 0.1]
    )
    assert matrix.shape == (2, 6)
    assert matrix[1, 0] == 1.0  # cut relative to grown from z_1_2_0
