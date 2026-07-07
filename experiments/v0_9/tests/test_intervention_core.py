import math

from intervention_core import (
    dot,
    exact_sign_flip_p,
    linear_slope,
    norm,
    normalize,
    orthogonal_random_direction,
)


def test_random_direction_is_unit_and_orthogonal() -> None:
    reference = normalize([1.0, 2.0, 3.0, 4.0])
    random_direction = orthogonal_random_direction(reference, 42)
    assert math.isclose(norm(random_direction), 1.0, rel_tol=1e-9)
    assert abs(dot(reference, random_direction)) < 1e-9


def test_linear_slope() -> None:
    assert math.isclose(linear_slope([-2, -1, 0, 1, 2], [-4, -2, 0, 2, 4]), 2.0)


def test_sign_flip_detects_consistent_slopes() -> None:
    assert exact_sign_flip_p([1.0] * 6) == 2 / 64
