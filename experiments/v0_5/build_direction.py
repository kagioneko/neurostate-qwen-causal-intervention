from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

CONTRAST_CONTEXTS = (
    "opening a window",
    "preparing tea",
    "starting a small task",
    "checking the time",
    "arranging a desk",
    "walking to a station",
    "reading a map",
    "folding a towel",
    "waiting near an elevator",
    "writing in an empty notebook",
    "hearing distant traffic",
    "closing a drawer",
)

PROCEED_FRAMES = (
    "The response should proceed and continue with the next step about {context}.",
    "The response should begin and move forward with the next step about {context}.",
)

HESITATE_FRAMES = (
    "The response should hesitate and delay the next step about {context}.",
    "The response should pause and wait before the next step about {context}.",
)


def paired_prompts() -> list[tuple[str, str]]:
    pairs = []
    for context in CONTRAST_CONTEXTS:
        for proceed, hesitate in zip(PROCEED_FRAMES, HESITATE_FRAMES):
            pairs.append((proceed.format(context=context), hesitate.format(context=context)))
    return pairs


def decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("unsupported decoder layout")


def subtract(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("cannot average an empty vector list")
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise ValueError("all vectors must have the same dimension")
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(width)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-layer", type=int, default=13)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    hidden_state_index = args.target_layer + 1
    deltas = []
    proceed_vectors = []
    hesitate_vectors = []
    for pair_index, (proceed_text, hesitate_text) in enumerate(paired_prompts()):
        vectors = []
        for text in (proceed_text, hesitate_text):
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                result = model(**inputs, use_cache=False, output_hidden_states=True, return_dict=True)
            vector = result.hidden_states[hidden_state_index][0, -1, :].float().cpu().tolist()
            vectors.append(vector)
        proceed_vector, hesitate_vector = vectors
        proceed_vectors.append(proceed_vector)
        hesitate_vectors.append(hesitate_vector)
        deltas.append(subtract(proceed_vector, hesitate_vector))
        print(f"{pair_index + 1:02d} layer={args.target_layer} built delta")

    raw_delta = mean_vector(deltas)
    mean_proceed = mean_vector(proceed_vectors)
    mean_hesitate = mean_vector(hesitate_vectors)
    raw_delta_norm = sum(value * value for value in raw_delta) ** 0.5
    metadata = {
        "source": "proceed_hesitate_paired_prompts_v0_5",
        "model": MODEL_NAME,
        "field": "proceed_hesitate_last_hidden_delta",
        "pair_count": len(deltas),
        "dimension": len(raw_delta),
        "hidden_state_index": hidden_state_index,
        "decoder_layer_index": args.target_layer,
        "raw_delta_norm": raw_delta_norm,
        "mean_proceed_norm": sum(value * value for value in mean_proceed) ** 0.5,
        "mean_hesitate_norm": sum(value * value for value in mean_hesitate) ** 0.5,
        "positive_label": "proceed",
        "negative_label": "hesitate",
        "raw_delta": raw_delta,
    }
    args.output.write_text(json.dumps(metadata, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
