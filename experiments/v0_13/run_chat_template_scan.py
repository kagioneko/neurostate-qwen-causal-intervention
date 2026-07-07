from __future__ import annotations

import argparse
import json
from pathlib import Path

from intervention_core import CONTRAST_BANKS, PROMPTS, normalize, orthogonal_random_direction
from run_intervention import MODEL_NAME, RANDOM_ALPHAS, SEMANTIC_ALPHAS, token_ids
from run_position_scan import PositionSteeringHook, decoder_layers


def build_chat_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-layer", type=int, default=13)
    parser.add_argument("--token-position", type=int, default=-1)
    parser.add_argument("--prompt-count", type=int, default=20)
    parser.add_argument("--random-count", type=int, default=63)
    parser.add_argument("--random-seed", type=int, default=20260713)
    parser.add_argument("--contrast-bank", choices=tuple(CONTRAST_BANKS), default="proceed_hesitate")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    metadata = json.loads(args.direction_json.read_text(encoding="utf-8"))
    semantic_direction = normalize(metadata["raw_delta"])
    directions = [("semantic", 0, semantic_direction)]
    directions.extend(
        ("random", index, orthogonal_random_direction(semantic_direction, args.random_seed + index))
        for index in range(args.random_count)
    )

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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "format_position_scan.jsonl"
    run_id = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for direction_kind, direction_index, direction in directions:
            alphas = SEMANTIC_ALPHAS if direction_kind == "semantic" else RANDOM_ALPHAS
            for prompt_id in range(args.prompt_count):
                prompt = PROMPTS[prompt_id]
                text = build_chat_prompt(tokenizer, prompt)
                inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
                prompt_len = int(inputs["input_ids"].shape[-1])
                resolved_position = args.token_position if args.token_position >= 0 else prompt_len + args.token_position
                boundary_token_id = int(inputs["input_ids"][0, resolved_position])
                boundary_token = tokenizer.convert_ids_to_tokens(boundary_token_id)
                for alpha in alphas:
                    hook = PositionSteeringHook(torch, direction, alpha, args.token_position)
                    registered = layers[args.target_layer].register_forward_hook(hook)
                    try:
                        with torch.inference_mode():
                            result = model(**inputs, use_cache=False, return_dict=True)
                            logits = result.logits[0, -1, :].float()
                            contrasts = {}
                            for bank_name, (positive_ids, negative_ids) in bank_token_ids.items():
                                contrasts[bank_name] = float(
                                    (logits[positive_ids].mean() - logits[negative_ids].mean()).cpu()
                                )
                    finally:
                        registered.remove()
                    run_id += 1
                    row = {
                        "run_id": run_id,
                        "model": MODEL_NAME,
                        "mode": "chat_template_scan",
                        "prompt_format": "qwen_chat_template",
                        "direction_kind": direction_kind,
                        "direction_index": direction_index,
                        "random_seed": args.random_seed,
                        "target_layer": args.target_layer,
                        "token_position": args.token_position,
                        "resolved_token_position": resolved_position,
                        "boundary_token_id": boundary_token_id,
                        "boundary_token": boundary_token,
                        "alpha": alpha,
                        "prompt_id": prompt_id,
                        "prompt": text,
                        **{f"logit_contrast_{name}": value for name, value in contrasts.items()},
                        "logit_contrast": contrasts[args.contrast_bank],
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(
                        f"{run_id:04d} pos={args.token_position:+03d} token={boundary_token!r} "
                        f"{direction_kind}:{direction_index:02d} prompt={prompt_id:02d} "
                        f"alpha={alpha:+.1f} contrast={row['logit_contrast']:+.3f}"
                    )
    print(output_path)


if __name__ == "__main__":
    main()
