from analyze_position_scan import metrics_for_position
from run_position_scan import parse_int_list


def make_row(position: int, kind: str, direction_index: int, prompt_id: int, alpha: float, contrast: float) -> dict:
    return {
        "target_layer": 13,
        "token_position": position,
        "direction_kind": kind,
        "direction_index": direction_index,
        "prompt_id": prompt_id,
        "alpha": alpha,
        "logit_contrast": contrast,
        "logit_contrast_proceed_hesitate": contrast,
    }


def test_parse_int_list_accepts_negative_positions() -> None:
    assert parse_int_list("-1,-2, -4") == [-1, -2, -4]


def test_metrics_for_position_filters_position() -> None:
    rows = [
        make_row(-1, "semantic", 0, 0, -1.0, 0.0),
        make_row(-1, "semantic", 0, 0, 1.0, 2.0),
        make_row(-1, "random", 0, 0, -2.0, 0.0),
        make_row(-1, "random", 0, 0, 2.0, 1.0),
        make_row(-2, "semantic", 0, 0, -1.0, 10.0),
        make_row(-2, "semantic", 0, 0, 1.0, 20.0),
    ]

    metrics = metrics_for_position(rows, -1, "proceed_hesitate")

    assert metrics["token_position"] == -1
    assert metrics["prompt_count"] == 1
    assert metrics["random_direction_count"] == 1
    assert metrics["semantic_mean_logit_slope"] == 1.0
