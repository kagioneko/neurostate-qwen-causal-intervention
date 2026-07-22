from __future__ import annotations

from dataclasses import dataclass

from colab_neurostate_3axis import Hook, chat, ids, layers, normalize, random_direction

MODEL = "Qwen/Qwen3-1.7B"
SOURCE_LAYER = 18
TARGET_LAYER = 20

TRAIN_PROMPTS = (
    "Write one short sentence about opening a window.",
    "Write one short sentence about preparing tea.",
    "Write one short sentence about an empty notebook.",
    "Write one short sentence about walking to a station.",
    "Write one short sentence about rain on a roof.",
    "Write one short sentence about arranging a desk.",
    "Write one short sentence about a quiet library.",
    "Write one short sentence about starting a small task.",
    "Write one short sentence about watching clouds.",
    "Write one short sentence about tying a shoelace.",
)
APPROACH_LOW = "Answer as if hesitant, paused, and waiting before continuing."
APPROACH_HIGH = "Answer as if ready to proceed and continue with practical action."
POSITIVE_BANK = (" proceed", " continue", " advance", " move", " begin", " start")
NEGATIVE_BANK = (" hesitate", " delay", " pause", " wait", " stop", " hold")
RANDOM_SEEDS = tuple(20260722 + i for i in range(5))


def mean_delta(highs, lows):
    return [
        sum(high[i] - low[i] for high, low in zip(highs, lows)) / len(highs)
        for i in range(len(highs[0]))
    ]


@dataclass
class SteeringResult:
    direction_name: str
    alpha: float
    baseline_contrast: float
    steered_contrast: float
    delta: float
    generated: str


@dataclass
class TraceStep:
    step: int
    token: str
    token_id: int
    intervention_applied: bool
    baseline_contrast: float
    steered_contrast: float
    delta: float


class ApproachSteeringDemo:
    """Research demo validated only for source 18 -> target 20."""

    def __init__(
        self,
        model_name: str = MODEL,
        source_layer: int = SOURCE_LAYER,
        target_layer: int = TARGET_LAYER,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("Use a Colab GPU runtime")
        if (source_layer, target_layer) != (SOURCE_LAYER, TARGET_LAYER):
            raise ValueError(
                "Only source 18 -> target 20 was evaluated; other layers are disabled"
            )
        self.torch = torch
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.decoder_layers = layers(self.model)
        self.positive_ids = ids(self.tokenizer, POSITIVE_BANK)
        self.negative_ids = ids(self.tokenizer, NEGATIVE_BANK)
        self.direction = self._build_direction()
        self.directions = {"semantic": self.direction}
        self.directions.update(
            {
                f"random #{i + 1}": random_direction(self.direction, seed)
                for i, seed in enumerate(RANDOM_SEEDS)
            }
        )
        print(
            "token-bank ids:",
            f"positive={len(self.positive_ids)}/{len(POSITIVE_BANK)}",
            f"negative={len(self.negative_ids)}/{len(NEGATIVE_BANK)}",
        )

    def _activation(self, text: str):
        inputs = self.tokenizer(
            chat(self.tokenizer, text),
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.model.device)
        with self.torch.inference_mode():
            output = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        return (
            output.hidden_states[self.source_layer + 1][0, -1]
            .float()
            .cpu()
            .tolist()
        )

    def _build_direction(self):
        low = [
            self._activation(f"{prompt} {APPROACH_LOW}")
            for prompt in TRAIN_PROMPTS
        ]
        high = [
            self._activation(f"{prompt} {APPROACH_HIGH}")
            for prompt in TRAIN_PROMPTS
        ]
        return normalize(mean_delta(high, low))

    def _inputs(self, prompt: str):
        return self.tokenizer(
            chat(self.tokenizer, prompt),
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.model.device)

    def _direction(self, direction_name: str):
        if direction_name not in self.directions:
            raise ValueError(f"unknown direction: {direction_name}")
        return self.directions[direction_name]

    def _next_logits(self, inputs, alpha: float, direction_name: str = "semantic"):
        handle = None
        if alpha != 0:
            handle = self.decoder_layers[self.target_layer].register_forward_hook(
                Hook(self.torch, self._direction(direction_name), alpha, -1)
            )
        try:
            with self.torch.inference_mode():
                return self.model(
                    **inputs,
                    use_cache=False,
                    return_dict=True,
                ).logits[0, -1].float()
        finally:
            if handle is not None:
                handle.remove()

    def _contrast(self, logits) -> float:
        return float(
            (
                logits[self.positive_ids].mean()
                - logits[self.negative_ids].mean()
            ).cpu()
        )

    def run(
        self,
        prompt: str,
        alpha: float,
        max_new_tokens: int = 40,
        direction_name: str = "semantic",
    ) -> SteeringResult:
        if not prompt.strip():
            raise ValueError("Prompt is empty")
        if not -2.0 <= alpha <= 2.0:
            raise ValueError("alpha must be between -2 and 2")
        max_new_tokens = max(1, min(int(max_new_tokens), 96))
        inputs = self._inputs(prompt)
        baseline_logits = self._next_logits(inputs, 0.0)
        steered_logits = self._next_logits(inputs, float(alpha), direction_name)
        baseline_contrast = self._contrast(baseline_logits)
        steered_contrast = self._contrast(steered_logits)

        first_token = steered_logits.argmax().reshape(1, 1).to(
            inputs["input_ids"].device
        )
        full_ids = self.torch.cat([inputs["input_ids"], first_token], dim=1)
        full_mask = self.torch.ones_like(full_ids)
        if max_new_tokens > 1:
            with self.torch.inference_mode():
                generated_ids = self.model.generate(
                    input_ids=full_ids,
                    attention_mask=full_mask,
                    max_new_tokens=max_new_tokens - 1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
        else:
            generated_ids = full_ids
        new_ids = generated_ids[0, inputs["input_ids"].shape[1] :]
        generated = self.tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
        ).strip()
        return SteeringResult(
            direction_name=direction_name,
            alpha=float(alpha),
            baseline_contrast=baseline_contrast,
            steered_contrast=steered_contrast,
            delta=steered_contrast - baseline_contrast,
            generated=generated,
        )

    def trace(
        self,
        prompt: str,
        alpha: float,
        mode: str = "observe",
        max_new_tokens: int = 40,
        direction_name: str = "semantic",
    ):
        if not prompt.strip():
            raise ValueError("Prompt is empty")
        if not -2.0 <= alpha <= 2.0:
            raise ValueError("alpha must be between -2 and 2")
        if mode not in {"observe", "boundary", "continuous"}:
            raise ValueError("mode must be observe, boundary, or continuous")
        max_new_tokens = max(1, min(int(max_new_tokens), 96))
        initial = self._inputs(prompt)
        full_ids = initial["input_ids"]
        trace_steps = []

        for step in range(max_new_tokens):
            current = {
                "input_ids": full_ids,
                "attention_mask": self.torch.ones_like(full_ids),
            }
            baseline_logits = self._next_logits(current, 0.0)
            apply_intervention = (
                mode == "continuous"
                or (mode == "boundary" and step == 0)
            )
            if apply_intervention:
                choice_logits = self._next_logits(
                    current, float(alpha), direction_name
                )
            else:
                choice_logits = baseline_logits
            baseline_contrast = self._contrast(baseline_logits)
            steered_contrast = self._contrast(choice_logits)
            token_id = int(choice_logits.argmax().item())
            token = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            )
            trace_steps.append(
                TraceStep(
                    step=step,
                    token=token,
                    token_id=token_id,
                    intervention_applied=apply_intervention,
                    baseline_contrast=baseline_contrast,
                    steered_contrast=steered_contrast,
                    delta=steered_contrast - baseline_contrast,
                )
            )
            next_token = self.torch.tensor(
                [[token_id]],
                device=full_ids.device,
                dtype=full_ids.dtype,
            )
            full_ids = self.torch.cat([full_ids, next_token], dim=1)
            if token_id == self.tokenizer.eos_token_id:
                break

        new_ids = full_ids[0, initial["input_ids"].shape[1] :]
        generated = self.tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
        ).strip()
        return {
            "direction_name": direction_name,
            "mode": mode,
            "alpha": float(alpha),
            "generated": generated,
            "steps": trace_steps,
        }
