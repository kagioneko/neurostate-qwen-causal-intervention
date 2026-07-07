from build_direction import mean_vector, paired_prompts, subtract


def test_paired_prompts_are_balanced() -> None:
    pairs = paired_prompts()

    assert len(pairs) == 24
    assert all("proceed" in proceed or "begin" in proceed for proceed, _ in pairs)
    assert all("hesitate" in hesitate or "pause" in hesitate for _, hesitate in pairs)


def test_vector_helpers() -> None:
    assert subtract([3.0, 2.0], [1.0, 5.0]) == [2.0, -3.0]
    assert mean_vector([[1.0, 3.0], [5.0, 7.0]]) == [3.0, 5.0]
