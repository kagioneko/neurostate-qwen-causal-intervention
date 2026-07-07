from __future__ import annotations

import argparse
import json
from pathlib import Path

from intervention_core import PROMPTS, dot, normalize
from run_chat_template_scan import build_chat_prompt
from run_intervention import MODEL_NAME, SEMANTIC_ALPHAS, token_ids
from run_position_scan import PositionSteeringHook, decoder_layers

CONTROL_SPECS = {
    "certainty": ("Answer confidently and decisively.", "Answer uncertainly and tentatively."),
    "sentiment": ("Use a positive and optimistic tone.", "Use a negative and pessimistic tone."),
    "verbosity": ("Give a detailed and elaborate answer.", "Give a concise and brief answer."),
    "politeness": ("Answer politely and respectfully.", "Answer bluntly and rudely."),
}

MEASURE_BANKS = {
    "proceed_hesitate": ((" proceed", " continue", " advance", " move", " begin", " start"), (" hesitate", " delay", " pause", " wait", " stop", " hold")),
    "certainty_uncertainty": ((" certainly", " definitely", " clearly", " surely", " will"), (" perhaps", " maybe", " possibly", " uncertain", " might")),
    "positive_negative": ((" good", " positive", " happy", " hopeful", " pleasant"), (" bad", " negative", " sad", " bleak", " unpleasant")),
    "verbose_concise": ((" detailed", " elaborate", " furthermore", " additionally", " because"), (" brief", " concise", " simply", " short", " direct")),
    "polite_blunt": ((" please", " kindly", " thank", " appreciate", " respectfully"), (" blunt", " rude", " obviously", " simply", " just")),
}


def paired_text(prompt: str, instruction: str) -> str:
    return f"{prompt} {instruction}"


def mean_delta(positive: list[list[float]], negative: list[list[float]]) -> list[float]:
    if len(positive) != len(negative) or not positive:
        raise ValueError("paired non-empty activations required")
    return [
        sum(pos[i] - neg[i] for pos, neg in zip(positive, negative)) / len(positive)
        for i in range(len(positive[0]))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-layer", type=int, default=11)
    parser.add_argument("--target-layer", type=int, default=13)
    parser.add_argument("--token-position", type=int, default=-1)
    parser.add_argument("--train-count", type=int, default=10)
    parser.add_argument("--prompt-count", type=int, default=20)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    metadata = json.loads(args.direction_json.read_text(encoding="utf-8"))
    neurostate = normalize(metadata["raw_delta"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
    if device == "cpu":
        model.to(device)
    model.eval()
    layers = decoder_layers(model)

    directions = {"neurostate": neurostate}
    for name, (positive_instruction, negative_instruction) in CONTROL_SPECS.items():
        positive_vectors, negative_vectors = [], []
        for prompt in PROMPTS[: args.train_count]:
            for instruction, destination in ((positive_instruction, positive_vectors), (negative_instruction, negative_vectors)):
                text = build_chat_prompt(tokenizer, paired_text(prompt, instruction))
                inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
                with torch.inference_mode():
                    result = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
                destination.append(result.hidden_states[args.source_layer + 1][0, -1, :].float().cpu().tolist())
        directions[name] = normalize(mean_delta(positive_vectors, negative_vectors))

    bank_ids = {name: (token_ids(tokenizer, pos), token_ids(tokenizer, neg)) for name, (pos, neg) in MEASURE_BANKS.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "direction_geometry.json").write_text(
        json.dumps({"cosine_with_neurostate": {name: dot(neurostate, direction) for name, direction in directions.items()}}, indent=2) + "\n",
        encoding="utf-8",
    )
    output_path = args.output_dir / "semantic_control_scan.jsonl"
    run_id = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for direction_name, direction in directions.items():
            for prompt_id, prompt in enumerate(PROMPTS[: args.prompt_count]):
                text = build_chat_prompt(tokenizer, prompt)
                inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
                for alpha in SEMANTIC_ALPHAS:
                    registered = layers[args.target_layer].register_forward_hook(
                        PositionSteeringHook(torch, direction, alpha, args.token_position)
                    )
                    try:
                        with torch.inference_mode():
                            logits = model(**inputs, use_cache=False, return_dict=True).logits[0, -1, :].float()
                        contrasts = {
                            name: float((logits[pos_ids].mean() - logits[neg_ids].mean()).cpu())
                            for name, (pos_ids, neg_ids) in bank_ids.items()
                        }
                    finally:
                        registered.remove()
                    run_id += 1
                    row = {
                        "run_id": run_id,
                        "model": MODEL_NAME,
                        "direction_name": direction_name,
                        "source_layer": args.source_layer,
                        "target_layer": args.target_layer,
                        "token_position": args.token_position,
                        "alpha": alpha,
                        "prompt_id": prompt_id,
                        "prompt": text,
                        **{f"logit_contrast_{name}": value for name, value in contrasts.items()},
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"{run_id:04d} {direction_name:10s} prompt={prompt_id:02d} alpha={alpha:+.1f}")
    print(output_path)


if __name__ == "__main__":
    main()
