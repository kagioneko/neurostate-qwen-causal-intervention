from __future__ import annotations

import argparse
import json
from pathlib import Path

from intervention_core import (
    ACTIVE_WORDS,
    CALM_WORDS,
    CONTRAST_BANKS,
    PROMPTS,
    keyword_counts,
    normalize,
    orthogonal_random_direction,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SEMANTIC_ALPHAS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
RANDOM_ALPHAS = (-2.0, 2.0)


def decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("unsupported decoder layout")


class SteeringHook:
    def __init__(self, torch_module, direction: list[float], alpha: float):
        self.torch = torch_module
        self.direction = direction
        self.alpha = alpha

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        direction = self.torch.tensor(self.direction, device=hidden.device, dtype=hidden.dtype)
        edited = hidden.clone()
        edited[:, -1, :] = edited[:, -1, :] + self.alpha * direction
        return (edited, *output[1:]) if isinstance(output, tuple) else edited


def token_ids(tokenizer, words: tuple[str, ...]) -> list[int]:
    ids = []
    for word in words:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            ids.append(int(encoded[0]))
    return sorted(set(ids))


def parse_int_list(raw: str | None) -> list[int]:
    if not raw:
        return []
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def contrast_fields(bank_name: str) -> tuple[str, str, str]:
    return (
        f"logit_contrast_{bank_name}",
        f"positive_word_count_{bank_name}",
        f"negative_word_count_{bank_name}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction-json", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full", "layer_scan"), default="smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, default=None)
    parser.add_argument("--random-count", type=int, default=None)
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--contrast-bank", choices=tuple(CONTRAST_BANKS), default="proceed_hesitate")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    metadata = json.loads(args.direction_json.read_text(encoding="utf-8"))
    semantic_direction = normalize(metadata["raw_delta"])
    prompt_count = args.prompt_count if args.prompt_count is not None else (6 if args.mode == "smoke" else len(PROMPTS))
    random_count = args.random_count if args.random_count is not None else (5 if args.mode == "smoke" else 63)
    directions = [("semantic", 0, semantic_direction)]
    directions.extend(
        ("random", index, orthogonal_random_direction(semantic_direction, 20260703 + index))
        for index in range(random_count)
    )
    layer_list = parse_int_list(args.layers)

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
    default_layer = int(metadata["decoder_layer_index"])
    target_layers = layer_list if layer_list else [default_layer]
    bank_token_ids = {
        bank_name: (token_ids(tokenizer, positive), token_ids(tokenizer, negative))
        for bank_name, (positive, negative) in CONTRAST_BANKS.items()
    }
    primary_contrast_field = f"logit_contrast_{args.contrast_bank}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"intervention_{args.mode}.jsonl"
    run_id = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for target_layer in target_layers:
            for direction_kind, direction_index, direction in directions:
                alphas = SEMANTIC_ALPHAS if direction_kind == "semantic" else RANDOM_ALPHAS
                for prompt_id in range(prompt_count):
                    text = f"Task: {PROMPTS[prompt_id]}\nResponse:"
                    inputs = tokenizer(text, return_tensors="pt").to(model.device)
                    prompt_len = int(inputs["input_ids"].shape[-1])
                    for alpha in alphas:
                        hook = SteeringHook(torch, direction, alpha)
                        registered = layers[target_layer].register_forward_hook(hook)
                        try:
                            with torch.inference_mode():
                                result = model(**inputs, use_cache=False, return_dict=True)
                                logits = result.logits[0, -1, :].float()
                                bank_logit_contrasts = {}
                                for bank_name, (positive_ids, negative_ids) in bank_token_ids.items():
                                    positive_logit = float(logits[positive_ids].mean().cpu())
                                    negative_logit = float(logits[negative_ids].mean().cpu())
                                    bank_logit_contrasts[bank_name] = positive_logit - negative_logit
                                generated_text = ""
                                features = {}
                                if direction_kind == "semantic":
                                    generated = model.generate(
                                        **inputs,
                                        do_sample=False,
                                        max_new_tokens=12,
                                        use_cache=False,
                                        pad_token_id=tokenizer.eos_token_id,
                                    )
                                    generated_text = tokenizer.decode(
                                        generated[0, prompt_len:], skip_special_tokens=True
                                    ).strip()
                                    for bank_name, (positive_words, negative_words) in CONTRAST_BANKS.items():
                                        counts = keyword_counts(generated_text, positive_words, negative_words)
                                        features[f"positive_word_count_{bank_name}"] = counts["positive_word_count"]
                                        features[f"negative_word_count_{bank_name}"] = counts["negative_word_count"]
                                else:
                                    for bank_name in CONTRAST_BANKS:
                                        features[f"positive_word_count_{bank_name}"] = 0
                                        features[f"negative_word_count_{bank_name}"] = 0
                        finally:
                            registered.remove()
                        run_id += 1
                        row = {
                            "run_id": run_id,
                            "model": MODEL_NAME,
                            "mode": args.mode,
                            "direction_kind": direction_kind,
                            "direction_index": direction_index,
                            "target_layer": target_layer,
                            "alpha": alpha,
                            "prompt_id": prompt_id,
                            "prompt": text,
                            **{f"logit_contrast_{bank_name}": value for bank_name, value in bank_logit_contrasts.items()},
                            "logit_contrast": bank_logit_contrasts[args.contrast_bank],
                            "output": generated_text,
                            **features,
                        }
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        print(
                            f"{run_id:04d} layer={target_layer:02d} {direction_kind}:{direction_index:02d} "
                            f"prompt={prompt_id:02d} alpha={alpha:+.1f} "
                            f"contrast={row[primary_contrast_field]:+.3f}"
                        )
    print(output_path)


if __name__ == "__main__":
    main()
