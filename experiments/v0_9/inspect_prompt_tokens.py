from __future__ import annotations

import argparse
import csv
from pathlib import Path

from intervention_core import PROMPTS
from run_intervention import MODEL_NAME


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=str, default="-12,-10,-8,-6,-4,-2,-1")
    parser.add_argument("--prompt-count", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    positions = parse_int_list(args.positions)
    rows = []
    for prompt_id, prompt in enumerate(PROMPTS[: args.prompt_count]):
        text = f"Task: {prompt}\nResponse:"
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        for position in positions:
            token_index = len(tokens) + position if position < 0 else position
            token = tokens[token_index] if 0 <= token_index < len(tokens) else "<OUT>"
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "token_count": len(tokens),
                    "token_position": position,
                    "token_index": token_index,
                    "token": token,
                    "prompt": prompt,
                    "full_text": text,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "prompt_position_tokens.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# Qwen Prompt Position Tokens",
        "",
        f"- model: `{MODEL_NAME}`",
        f"- prompt count: {args.prompt_count}",
        f"- positions: {', '.join(str(position) for position in positions)}",
        "",
        "| prompt | token count | position | index | token |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        report.append(
            "| {prompt_id} | {token_count} | {token_position} | {token_index} | `{token}` |".format(**row)
        )
    report.extend(
        [
            "",
            "Negative positions are counted from the end of the prompt sequence.",
            "The full prompt text is available in `prompt_position_tokens.csv`.",
        ]
    )
    report_path = args.output_dir / "prompt_position_tokens.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
