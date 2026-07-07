from intervention_core import dot, normalize
from run_semantic_control_scan import mean_delta, paired_text, subtract_projection
from analyze_semantic_control_scan import exact_sign_flip_p_numpy, sign_matrix


def test_mean_delta():
    assert mean_delta([[3.0, 5.0], [5.0, 9.0]], [[1.0, 1.0], [1.0, 3.0]]) == [3.0, 5.0]


def test_paired_text():
    assert paired_text("Base.", "Control.") == "Base. Control."


def test_numpy_sign_flip_matches_small_exact_result():
    assert exact_sign_flip_p_numpy([1.0, 1.0], sign_matrix(2)) == 0.5


def test_subtract_projection_removes_basis_component():
    residual = normalize(subtract_projection([1.0, 1.0], [[1.0, 0.0]]))
    assert abs(dot(residual, [1.0, 0.0])) < 1e-12
    assert residual == [0.0, 1.0]
