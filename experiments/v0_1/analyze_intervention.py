from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from intervention_core import exact_sign_flip_p, linear_slope


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def semantic_slopes(rows: list[dict]) -> list[float]:
    grouped = defaultdict(list)
    for row in rows:
        if row["direction_kind"] == "semantic":
            grouped[int(row["prompt_id"])].append(row)
    slopes = []
    for prompt_rows in grouped.values():
        prompt_rows.sort(key=lambda row: float(row["alpha"]))
        slopes.append(
            linear_slope(
                [float(row["alpha"]) for row in prompt_rows],
                [float(row["logit_contrast"]) for row in prompt_rows],
            )
        )
    return slopes


def random_slopes(rows: list[dict]) -> list[float]:
    grouped = defaultdict(list)
    for row in rows:
        if row["direction_kind"] == "random":
            grouped[int(row["direction_index"])].append(row)
    slopes = []
    for direction_rows in grouped.values():
        negative = [float(row["logit_contrast"]) for row in direction_rows if float(row["alpha"]) < 0]
        positive = [float(row["logit_contrast"]) for row in direction_rows if float(row["alpha"]) > 0]
        slopes.append((sum(positive) / len(positive) - sum(negative) / len(negative)) / 4.0)
    return slopes


def output_summary(rows: list[dict]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        if row["direction_kind"] == "semantic":
            grouped[float(row["alpha"])].append(row)
    return [
        {
            "alpha": alpha,
            "n": len(items),
            "mean_logit_contrast": sum(float(row["logit_contrast"]) for row in items) / len(items),
            "mean_active_words": sum(int(row["active_word_count"]) for row in items) / len(items),
            "mean_calm_words": sum(int(row["calm_word_count"]) for row in items) / len(items),
        }
        for alpha, items in sorted(grouped.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.runs)
    sem_slopes = semantic_slopes(rows)
    rand_slopes = random_slopes(rows)
    mean_semantic = sum(sem_slopes) / len(sem_slopes)
    specificity_p = (
        1 + sum(abs(value) >= abs(mean_semantic) for value in rand_slopes)
    ) / (len(rand_slopes) + 1)
    result = {
        "mode": rows[0]["mode"],
        "prompt_count": len(sem_slopes),
        "random_direction_count": len(rand_slopes),
        "semantic_mean_logit_slope": mean_semantic,
        "semantic_median_logit_slope": sorted(sem_slopes)[len(sem_slopes) // 2],
        "semantic_sign_flip_p": exact_sign_flip_p(sem_slopes),
        "random_abs_slope_mean": sum(abs(value) for value in rand_slopes) / len(rand_slopes),
        "random_specificity_rank_p": specificity_p,
    }
    summaries = output_summary(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "causal_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "alpha_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    report = [
        "# Qwen NeuroState Causal Intervention",
        "",
        f"- mode: `{result['mode']}`",
        f"- prompts: {result['prompt_count']}",
        f"- random orthogonal directions: {result['random_direction_count']}",
        f"- semantic mean logit slope: {result['semantic_mean_logit_slope']:.6f}",
        f"- exact paired sign-flip p: {result['semantic_sign_flip_p']:.6f}",
        f"- random-direction specificity rank p: {result['random_specificity_rank_p']:.6f}",
        "",
        "| alpha | mean logit contrast | active words | calm words |",
        "|---:|---:|---:|---:|",
    ]
    for row in summaries:
        report.append(
            "| {alpha:+.1f} | {mean_logit_contrast:.4f} | "
            "{mean_active_words:.3f} | {mean_calm_words:.3f} |".format(**row)
        )
    report.extend([
        "",
        "A causal claim requires both a non-zero paired slope and specificity relative to random directions.",
        "The sign indicates orientation only; a reversed sign does not invalidate causal influence.",
    ])
    (args.output_dir / "causal_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
