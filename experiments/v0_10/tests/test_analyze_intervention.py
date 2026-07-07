from analyze_intervention import metrics_for_layer, random_distribution_rows


def make_row(layer: int, kind: str, direction_index: int, prompt_id: int, alpha: float, contrast: float) -> dict:
    return {
        "target_layer": layer,
        "direction_kind": kind,
        "direction_index": direction_index,
        "prompt_id": prompt_id,
        "alpha": alpha,
        "logit_contrast": contrast,
        "logit_contrast_proceed_hesitate": contrast,
        "positive_word_count_proceed_hesitate": 0,
        "negative_word_count_proceed_hesitate": 0,
        "active_word_count": 0,
        "calm_word_count": 0,
    }


def test_metrics_for_layer_groups_by_target_layer() -> None:
    rows = [
        make_row(11, "semantic", 0, 0, -1.0, 1.0),
        make_row(11, "semantic", 0, 0, 1.0, 2.0),
        make_row(11, "semantic", 0, 1, -1.0, 0.5),
        make_row(11, "semantic", 0, 1, 1.0, 1.5),
        make_row(11, "random", 0, 0, -2.0, 0.2),
        make_row(11, "random", 0, 0, 2.0, 0.6),
        make_row(13, "semantic", 0, 0, -1.0, 10.0),
        make_row(13, "semantic", 0, 0, 1.0, 11.0),
    ]

    metrics = metrics_for_layer(rows, 11)

    assert metrics["layer_index"] == 11
    assert metrics["contrast_bank"] == "proceed_hesitate"
    assert metrics["prompt_count"] == 2
    assert metrics["random_direction_count"] == 1
    assert metrics["random_directions_beating_semantic_abs"] == 0


def test_random_distribution_rows_orders_by_abs_slope() -> None:
    rows = random_distribution_rows({0: 0.1, 1: -0.4, 2: 0.2}, semantic_mean=0.2)

    assert [row["direction_index"] for row in rows] == [1, 2, 0]
    assert rows[0]["beats_semantic_abs"] is True
    assert rows[-1]["beats_semantic_abs"] is False
