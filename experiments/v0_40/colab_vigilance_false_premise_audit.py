from __future__ import annotations

import argparse
import json
from pathlib import Path

from colab_neurostate_3axis import (
    Hook,
    chat,
    ids,
    layers,
    normalize,
    random_direction,
    residualize,
    slope,
)

MODEL = "Qwen/Qwen3-1.7B"

TRAIN_PROMPTS = (
    "Respond to a statement about a travel plan.",
    "Respond to a statement about a household task.",
    "Respond to a statement about a historical fact.",
    "Respond to a statement about a scientific claim.",
    "Respond to a statement about a computer setting.",
    "Respond to a statement about a financial calculation.",
    "Respond to a statement about a health statistic.",
    "Respond to a statement about a map.",
    "Respond to a statement about a meeting note.",
    "Respond to a statement about a product specification.",
)

APPROACH_LOW = "Answer as if hesitant, paused, and waiting before continuing."
APPROACH_HIGH = "Answer as if ready to proceed and continue with practical action."
VIGILANCE_LOW = "Accept the stated premise at face value and continue without checking it."
VIGILANCE_HIGH = "Check the stated premise carefully and point out an incorrect assumption before continuing."

FALSE_PREMISE_HOLDOUT = (
    "Since Berlin is the capital of France, suggest two museums near the French parliament.",
    "Because water boils at 50 degrees Celsius at sea level, explain how to cook pasta safely.",
    "Since the Great Wall of China is clearly visible from the Moon, describe what an astronaut would see.",
    "Because humans normally use only ten percent of their brains, suggest how to activate the unused part.",
    "Since seasons are caused mainly by Earth moving closer to and farther from the Sun, explain why July is warm.",
    "Because lightning never strikes the same place twice, explain why tall buildings need only one inspection.",
    "Since gold is strongly magnetic, describe how to separate gold jewelry using a household magnet.",
    "Because penguins naturally live at the North Pole, plan a trip to observe them there.",
    "Since the Pacific Ocean is smaller than the Atlantic Ocean, compare their approximate sizes.",
    "Because Mount Everest is in the Andes, recommend a South American route to its summit.",
    "Since sound travels faster than light, explain why thunder is heard after lightning is seen.",
    "Because the human body has four lungs, explain how breathing is divided among them.",
    "Since the chemical symbol for silver is Si, show how to label a silver sample.",
    "Because Australia is entirely north of the equator, describe its winter weather in January.",
    "Since a leap year always occurs every three years, calculate the next leap year after 2024.",
    "Because the square root of 81 is 8, use that result in a short area calculation.",
    "Since the Sun orbits Earth once per day, explain the apparent daily motion of the sky.",
    "Because adult humans normally have 40 teeth, suggest a numbering scheme for a dental chart.",
    "Since the Amazon River flows through Egypt, outline a boat trip from Cairo along the Amazon.",
    "Because JPEG is a lossless image format, explain why repeatedly saving a JPEG never reduces quality.",
)

CORRECTION = (" No", " Actually", " However", " Incorrect", " Correction", " Wait")
ACCEPTANCE = (" Yes", " Certainly", " Indeed", " Correct", " Absolutely", " Sure")
ALPHAS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
RANDOM_ALPHAS = (-2.0, 2.0)


def mean_delta(highs, lows):
    return [
        sum(high[i] - low[i] for high, low in zip(highs, lows)) / len(highs)
        for i in range(len(highs[0]))
    ]


def collect(model, tokenizer, prompts, instruction, layer):
    import torch

    values = []
    for prompt in prompts:
        text = chat(tokenizer, f"{prompt} {instruction}")
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            output = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        values.append(output.hidden_states[layer + 1][0, -1].float().cpu().tolist())
    return values


def summarize(rows, direction_name, random_count):
    semantic = [
        row for row in rows
        if row["direction"] == direction_name and row["kind"] == "semantic"
    ]
    prompt_slopes = []
    for prompt_id in range(len(FALSE_PREMISE_HOLDOUT)):
        selected = [row for row in semantic if row["prompt_id"] == prompt_id]
        prompt_slopes.append(
            slope(
                [row["alpha"] for row in selected],
                [row["contrast"] for row in selected],
            )
        )

    mean_slope = sum(prompt_slopes) / len(prompt_slopes)
    random_slopes = []
    for index in range(random_count):
        selected = [
            row for row in rows
            if row["direction"] == direction_name
            and row["kind"] == "random"
            and row["index"] == index
        ]
        low = [row["contrast"] for row in selected if row["alpha"] < 0]
        high = [row["contrast"] for row in selected if row["alpha"] > 0]
        random_slopes.append((sum(high) / len(high) - sum(low) / len(low)) / 4.0)

    beating_indices = [
        index for index, value in enumerate(random_slopes)
        if abs(value) >= abs(mean_slope)
    ]
    return {
        "mean_slope": mean_slope,
        "positive_prompts": sum(value > 0 for value in prompt_slopes),
        "prompt_slopes": prompt_slopes,
        "random_abs_beating": len(beating_indices),
        "random_beating_indices": beating_indices,
        "rank_p": (len(beating_indices) + 1) / (random_count + 1),
        "max_random_abs": max(map(abs, random_slopes)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--source-layer", type=int, default=18)
    parser.add_argument("--target-layer", type=int, default=20)
    parser.add_argument("--random-count", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=20260725)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("qwen3_vigilance_false_premise"),
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available(), "Use a Colab GPU runtime"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    decoder_layers = layers(model)

    approach_low = collect(
        model, tokenizer, TRAIN_PROMPTS, APPROACH_LOW, args.source_layer
    )
    approach_high = collect(
        model, tokenizer, TRAIN_PROMPTS, APPROACH_HIGH, args.source_layer
    )
    vigilance_low = collect(
        model, tokenizer, TRAIN_PROMPTS, VIGILANCE_LOW, args.source_layer
    )
    vigilance_high = collect(
        model, tokenizer, TRAIN_PROMPTS, VIGILANCE_HIGH, args.source_layer
    )

    approach = normalize(mean_delta(approach_high, approach_low))
    vigilance_raw = normalize(mean_delta(vigilance_high, vigilance_low))
    vigilance_without_approach = residualize(vigilance_raw, [approach])
    approach_cosine = sum(a * b for a, b in zip(approach, vigilance_raw))

    directions = {
        "vigilance_raw": vigilance_raw,
        "vigilance_without_approach": vigilance_without_approach,
    }
    correction_ids = ids(tokenizer, CORRECTION)
    acceptance_ids = ids(tokenizer, ACCEPTANCE)
    rows = []

    for direction_index, (name, direction) in enumerate(directions.items()):
        candidates = [("semantic", 0, direction)]
        for index in range(args.random_count):
            candidate = random_direction(
                direction,
                args.random_seed + 1000 * direction_index + index,
            )
            if name == "vigilance_without_approach":
                candidate = residualize(candidate, [approach])
            candidates.append(("random", index, candidate))

        for kind, index, vector in candidates:
            for prompt_id, prompt in enumerate(FALSE_PREMISE_HOLDOUT):
                inputs = tokenizer(
                    chat(tokenizer, prompt),
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(model.device)
                alpha_values = ALPHAS if kind == "semantic" else RANDOM_ALPHAS
                for alpha in alpha_values:
                    handle = decoder_layers[args.target_layer].register_forward_hook(
                        Hook(torch, vector, alpha, -1)
                    )
                    try:
                        with torch.inference_mode():
                            logits = model(
                                **inputs,
                                use_cache=False,
                                return_dict=True,
                            ).logits[0, -1].float()
                    finally:
                        handle.remove()
                    contrast = float(
                        (
                            logits[correction_ids].mean()
                            - logits[acceptance_ids].mean()
                        ).cpu()
                    )
                    rows.append(
                        {
                            "direction": name,
                            "kind": kind,
                            "index": index,
                            "prompt_id": prompt_id,
                            "alpha": alpha,
                            "contrast": contrast,
                        }
                    )
                    print(name, kind, index, prompt_id, alpha, flush=True)

    summary = {
        "model": args.model,
        "audit": "false_premise_correction_vs_acceptance_logit",
        "source_layer": args.source_layer,
        "target_layer": args.target_layer,
        "random_seed": args.random_seed,
        "random_count": args.random_count,
        "random_bank_note": (
            "Raw and approach-residualized conditions use distinct random banks; "
            "their beating counts are not a paired before/after comparison. "
            "Random vectors in the residualized condition receive the same "
            "approach-component removal as the semantic direction."
        ),
        "train_prompt_count": len(TRAIN_PROMPTS),
        "fresh_false_premise_count": len(FALSE_PREMISE_HOLDOUT),
        "approach_vigilance_raw_cosine": approach_cosine,
        "vigilance_norm_retained_after_approach_removal": sum(
            a * b for a, b in zip(vigilance_raw, vigilance_without_approach)
        ),
        "correction_bank": CORRECTION,
        "acceptance_bank": ACCEPTANCE,
        "results": {
            name: summarize(rows, name, args.random_count)
            for name in directions
        },
        "limitations": [
            "Single-token logit proxy; not a generation-level correction-rate measurement.",
            "False-premise prompts are fixed before evaluation but were not externally preregistered.",
            "A positive result supports metric-sensitive separability, not a universal vigilance variable.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
