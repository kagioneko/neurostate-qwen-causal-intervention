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


def rows_for_layer(rows: list[dict], layer_index: int | None) -> list[dict]:
    if layer_index is None:
        return rows
    return [row for row in rows if int(row["target_layer"]) == layer_index]


def contrast_field(bank_name: str) -> str:
    return "logit_contrast" if bank_name == "action_calm" else f"logit_contrast_{bank_name}"


def positive_count_field(bank_name: str) -> str:
    return "active_word_count" if bank_name == "action_calm" else f"positive_word_count_{bank_name}"


def negative_count_field(bank_name: str) -> str:
    return "calm_word_count" if bank_name == "action_calm" else f"negative_word_count_{bank_name}"


def semantic_slopes(
    rows: list[dict],
    layer_index: int | None = None,
    contrast_bank: str = "proceed_hesitate",
) -> list[float]:
    grouped = defaultdict(list)
    for row in rows_for_layer(rows, layer_index):
        if row["direction_kind"] == "semantic":
            grouped[int(row["prompt_id"])].append(row)
    slopes = []
    field = contrast_field(contrast_bank)
    for prompt_rows in grouped.values():
        prompt_rows.sort(key=lambda row: float(row["alpha"]))
        slopes.append(
            linear_slope(
                [float(row["alpha"]) for row in prompt_rows],
                [float(row[field]) for row in prompt_rows],
            )
        )
    return slopes


def random_slopes(
    rows: list[dict],
    layer_index: int | None = None,
    contrast_bank: str = "proceed_hesitate",
) -> list[float]:
    grouped = defaultdict(list)
    for row in rows_for_layer(rows, layer_index):
        if row["direction_kind"] == "random":
            grouped[int(row["direction_index"])].append(row)
    slopes = []
    field = contrast_field(contrast_bank)
    for direction_rows in grouped.values():
        negative = [float(row[field]) for row in direction_rows if float(row["alpha"]) < 0]
        positive = [float(row[field]) for row in direction_rows if float(row["alpha"]) > 0]
        slopes.append((sum(positive) / len(positive) - sum(negative) / len(negative)) / 4.0)
    return slopes


def output_summary(
    rows: list[dict],
    layer_index: int | None = None,
    contrast_bank: str = "proceed_hesitate",
) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows_for_layer(rows, layer_index):
        if row["direction_kind"] == "semantic":
            grouped[float(row["alpha"])].append(row)
    pos_field = positive_count_field(contrast_bank)
    neg_field = negative_count_field(contrast_bank)
    field = contrast_field(contrast_bank)
    return [
        {
            "alpha": alpha,
            "n": len(items),
            "mean_logit_contrast": sum(float(row[field]) for row in items) / len(items),
            "mean_positive_words": sum(int(row[pos_field]) for row in items) / len(items),
            "mean_negative_words": sum(int(row[neg_field]) for row in items) / len(items),
        }
        for alpha, items in sorted(grouped.items())
    ]


def metrics_for_layer(
    rows: list[dict],
    layer_index: int,
    contrast_bank: str = "proceed_hesitate",
) -> dict[str, object]:
    sem_slopes = semantic_slopes(rows, layer_index=layer_index, contrast_bank=contrast_bank)
    rand_slopes = random_slopes(rows, layer_index=layer_index, contrast_bank=contrast_bank)
    mean_semantic = sum(sem_slopes) / len(sem_slopes)
    return {
        "layer_index": layer_index,
        "contrast_bank": contrast_bank,
        "prompt_count": len(sem_slopes),
        "random_direction_count": len(rand_slopes),
        "semantic_mean_logit_slope": mean_semantic,
        "semantic_median_logit_slope": sorted(sem_slopes)[len(sem_slopes) // 2],
        "semantic_sign_flip_p": exact_sign_flip_p(sem_slopes),
        "random_abs_slope_mean": sum(abs(value) for value in rand_slopes) / len(rand_slopes) if rand_slopes else 0.0,
        "random_specificity_rank_p": (
            (1 + sum(abs(value) >= abs(mean_semantic) for value in rand_slopes)) / (len(rand_slopes) + 1)
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
    layers = sorted({int(row["target_layer"]) for row in rows})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if len(layers) == 1:
        layer_index = layers[0]
        result = metrics_for_layer(rows, layer_index, contrast_bank=args.contrast_bank)
        result["mode"] = rows[0]["mode"]
        summaries = output_summary(rows, layer_index=layer_index, contrast_bank=args.contrast_bank)
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
            f"- contrast bank: `{result['contrast_bank']}`",
            f"- layer: {result['layer_index']}",
            f"- prompts: {result['prompt_count']}",
            f"- random orthogonal directions: {result['random_direction_count']}",
            f"- semantic mean logit slope: {result['semantic_mean_logit_slope']:.6f}",
            f"- exact paired sign-flip p: {result['semantic_sign_flip_p']:.6f}",
            f"- random-direction specificity rank p: {result['random_specificity_rank_p']:.6f}",
            "",
            "| alpha | mean logit contrast | positive words | negative words |",
            "|---:|---:|---:|---:|",
        ]
        for row in summaries:
            report.append(
                "| {alpha:+.1f} | {mean_logit_contrast:.4f} | "
                "{mean_positive_words:.3f} | {mean_negative_words:.3f} |".format(**row)
            )
        report.extend([
            "",
            "A causal claim requires both a non-zero paired slope and specificity relative to random directions.",
            "The sign indicates orientation only; a reversed sign does not invalidate causal influence.",
        ])
        (args.output_dir / "causal_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    layer_results = [metrics_for_layer(rows, layer_index, contrast_bank=args.contrast_bank) for layer_index in layers]
    result = {
        "mode": rows[0]["mode"],
        "contrast_bank": args.contrast_bank,
        "layer_count": len(layer_results),
        "layer_results": layer_results,
    }
    (args.output_dir / "causal_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "layer_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "layer_index",
                "contrast_bank",
                "prompt_count",
                "random_direction_count",
                "semantic_mean_logit_slope",
                "semantic_median_logit_slope",
                "semantic_sign_flip_p",
                "random_abs_slope_mean",
                "random_specificity_rank_p",
            ],
        )
        writer.writeheader()
        writer.writerows(layer_results)
    report = [
        "# Qwen NeuroState Causal Intervention",
        "",
        f"- mode: `{result['mode']}`",
        f"- contrast bank: `{result['contrast_bank']}`",
        f"- layers: {', '.join(str(layer) for layer in layers)}",
        f"- layer_count: {result['layer_count']}",
        "",
        "| layer | prompts | random dirs | mean slope | sign-flip p | specificity p |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in layer_results:
        report.append(
            "| {layer_index} | {prompt_count} | {random_direction_count} | "
            "{semantic_mean_logit_slope:.6f} | {semantic_sign_flip_p:.6f} | "
            "{random_specificity_rank_p:.6f} |".format(**item)
        )
    report.extend([
        "",
        "Layer sweeps are for locating where the signal lives; they do not by themselves establish specificity.",
    ])
    (args.output_dir / "causal_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
