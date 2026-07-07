from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from analyze_intervention import contrast_field
from intervention_core import exact_sign_flip_p, linear_slope


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def semantic_slopes(rows: list[dict], token_position: int, contrast_bank: str) -> list[float]:
    grouped = defaultdict(list)
    field = contrast_field(contrast_bank)
    for row in rows:
        if int(row["token_position"]) == token_position and row["direction_kind"] == "semantic":
            grouped[int(row["prompt_id"])].append(row)
    slopes = []
    for prompt_rows in grouped.values():
        prompt_rows.sort(key=lambda row: float(row["alpha"]))
        slopes.append(
            linear_slope(
                [float(row["alpha"]) for row in prompt_rows],
                [float(row[field]) for row in prompt_rows],
            )
        )
    return slopes


def random_slopes(rows: list[dict], token_position: int, contrast_bank: str) -> list[float]:
    grouped = defaultdict(list)
    field = contrast_field(contrast_bank)
    for row in rows:
        if int(row["token_position"]) == token_position and row["direction_kind"] == "random":
            grouped[int(row["direction_index"])].append(row)
    slopes = []
    for direction_rows in grouped.values():
        negative = [float(row[field]) for row in direction_rows if float(row["alpha"]) < 0]
        positive = [float(row[field]) for row in direction_rows if float(row["alpha"]) > 0]
        slopes.append((sum(positive) / len(positive) - sum(negative) / len(negative)) / 4.0)
    return slopes


def metrics_for_position(rows: list[dict], token_position: int, contrast_bank: str) -> dict[str, object]:
    sem_slopes = semantic_slopes(rows, token_position, contrast_bank)
    rand_slopes = random_slopes(rows, token_position, contrast_bank)
    mean_semantic = sum(sem_slopes) / len(sem_slopes)
    random_abs = sorted(abs(value) for value in rand_slopes)
    semantic_abs = abs(mean_semantic)
    random_beating = sum(value >= semantic_abs for value in random_abs)
    return {
        "token_position": token_position,
        "prompt_count": len(sem_slopes),
        "random_direction_count": len(rand_slopes),
        "semantic_mean_logit_slope": mean_semantic,
        "semantic_median_logit_slope": sorted(sem_slopes)[len(sem_slopes) // 2],
        "semantic_sign_flip_p": exact_sign_flip_p(sem_slopes),
        "random_abs_slope_mean": sum(random_abs) / len(random_abs) if random_abs else 0.0,
        "random_abs_slope_median": random_abs[len(random_abs) // 2] if random_abs else 0.0,
        "random_abs_slope_max": random_abs[-1] if random_abs else 0.0,
        "random_directions_beating_semantic_abs": random_beating,
        "semantic_abs_slope_random_quantile": (
            sum(value <= semantic_abs for value in random_abs) / len(random_abs)
            if random_abs
            else None
        ),
        "random_specificity_rank_p": (
            (1 + random_beating) / (len(rand_slopes) + 1)
            if rand_slopes
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contrast-bank", choices=("action_calm", "proceed_hesitate"), default="proceed_hesitate")
    args = parser.parse_args()

    rows = load_rows(args.runs)
    positions = sorted({int(row["token_position"]) for row in rows})
    layer = int(rows[0]["target_layer"])
    results = [metrics_for_position(rows, position, args.contrast_bank) for position in positions]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": "position_scan",
        "contrast_bank": args.contrast_bank,
        "target_layer": layer,
        "position_count": len(results),
        "position_results": results,
    }
    (args.output_dir / "position_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "position_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    report = [
        "# Qwen NeuroState Token Position Scan",
        "",
        f"- contrast bank: `{args.contrast_bank}`",
        f"- target layer: {layer}",
        f"- positions: {', '.join(str(position) for position in positions)}",
        "",
        "| token position | prompts | random dirs | mean slope | sign-flip p | specificity p | semantic quantile |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        report.append(
            "| {token_position} | {prompt_count} | {random_direction_count} | "
            "{semantic_mean_logit_slope:.6f} | {semantic_sign_flip_p:.6f} | "
            "{random_specificity_rank_p:.6f} | {semantic_abs_slope_random_quantile:.6f} |".format(**item)
        )
    report.extend([
        "",
        "Negative positions are counted from the end of the prompt sequence.",
        "Position `-1` is the previous experiment's last-token intervention site.",
    ])
    (args.output_dir / "position_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
