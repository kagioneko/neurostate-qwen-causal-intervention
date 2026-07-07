from __future__ import annotations

import argparse
import json
from pathlib import Path

from intervention_core import CONTRAST_BANKS, PROMPTS, normalize, orthogonal_random_direction
from run_intervention import MODEL_NAME, RANDOM_ALPHAS, SEMANTIC_ALPHAS, token_ids
from run_position_scan import PositionSteeringHook, decoder_layers, parse_int_list

PROMPT_FORMATS = {
    "task_response": "Task: {prompt}\nResponse:",
    "instruction_answer": "Instruction: {prompt}\nAnswer:",
    "user_assistant": "User: {prompt}\nAssistant:",
}


def parse_format_list(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [value for value in values if value not in PROMPT_FORMATS]
    if unknown:
        raise ValueError(f"unknown prompt formats: {unknown}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-layer", type=int, default=13)
    parser.add_argument("--positions", type=str, default="-2,-1")
    parser.add_argument("--formats", type=str, default="task_response,instruction_answer,user_assistant")
    parser.add_argument("--prompt-count", type=int, default=8)
    parser.add_argument("--random-count", type=int, default=15)
    parser.add_argument("--contrast-bank", choices=tuple(CONTRAST_BANKS), default="proceed_hesitate")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    metadata = json.loads(args.direction_json.read_text(encoding="utf-8"))
    semantic_direction = normalize(metadata["raw_delta"])
    directions = [("semantic", 0, semantic_direction)]
    directions.extend(
        ("random", index, orthogonal_random_direction(semantic_direction, 20260710 + index))
        for index in range(args.random_count)
    )
    positions = parse_int_list(args.positions)
    formats = parse_format_list(args.formats)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model.to(device)
    model.eval()
    layers = decoder_layers(model)
    bank_token_ids = {
        bank_name: (token_ids(tokenizer, positive), token_ids(tokenizer, negative))
        for bank_name, (positive, negative) in CONTRAST_BANKS.items()
    }
    primary_contrast_field = f"logit_contrast_{args.contrast_bank}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "format_position_scan.jsonl"
    run_id = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for format_name in formats:
            template = PROMPT_FORMATS[format_name]
            for token_position in positions:
                for direction_kind, direction_index, direction in directions:
                    alphas = SEMANTIC_ALPHAS if direction_kind == "semantic" else RANDOM_ALPHAS
                    for prompt_id in range(args.prompt_count):
                        text = template.format(prompt=PROMPTS[prompt_id])
                        inputs = tokenizer(text, return_tensors="pt").to(model.device)
                        prompt_len = int(inputs["input_ids"].shape[-1])
                        resolved_position = token_position if token_position >= 0 else prompt_len + token_position
                        for alpha in alphas:
                            hook = PositionSteeringHook(torch, direction, alpha, token_position)
                            registered = layers[args.target_layer].register_forward_hook(hook)
                            try:
                                with torch.inference_mode():
                                    result = model(**inputs, use_cache=False, return_dict=True)
                                    logits = result.logits[0, -1, :].float()
                                    bank_logit_contrasts = {}
                                    for bank_name, (positive_ids, negative_ids) in bank_token_ids.items():
                                        positive_logit = float(logits[positive_ids].mean().cpu())
                                        negative_logit = float(logits[negative_ids].mean().cpu())
                                        bank_logit_contrasts[bank_name] = positive_logit - negative_logit
                            finally:
                                registered.remove()
                            run_id += 1
                            row = {
                                "run_id": run_id,
                                "model": MODEL_NAME,
                                "mode": "format_position_scan",
                                "prompt_format": format_name,
                                "direction_kind": direction_kind,
                                "direction_index": direction_index,
                                "target_layer": args.target_layer,
                                "token_position": token_position,
                                "resolved_token_position": resolved_position,
                                "alpha": alpha,
                                "prompt_id": prompt_id,
                                "prompt": text,
                                **{f"logit_contrast_{bank_name}": value for bank_name, value in bank_logit_contrasts.items()},
                                "logit_contrast": bank_logit_contrasts[args.contrast_bank],
                            }
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            print(
                                f"{run_id:04d} fmt={format_name} pos={token_position:+03d} "
                                f"{direction_kind}:{direction_index:02d} prompt={prompt_id:02d} "
                                f"alpha={alpha:+.1f} contrast={row[primary_contrast_field]:+.3f}"
                            )
    print(output_path)


if __name__ == "__main__":
    main()
