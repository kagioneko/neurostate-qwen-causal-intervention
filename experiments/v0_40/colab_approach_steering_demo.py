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
    enable_thinking: bool
    alpha: float
    baseline_contrast: float
    steered_contrast: float
    delta: float
    generated: str


@dataclass
class TraceStep:
    step: int
    token: str
    raw_token: str
    cumulative_text: str
    token_id: int
    baseline_token: str
    baseline_token_id: int
    argmax_changed: bool
    intervention_applied: bool
    baseline_contrast: float
    steered_contrast: float
    delta: float
    baseline_margin: float
    steered_margin: float


@dataclass
class CandidateScore:
    candidate: str
    token_ids: list[int]
    raw_tokens: list[str]
    baseline_total_logprob: float
    steered_total_logprob: float
    delta_total_logprob: float
    baseline_mean_logprob: float
    steered_mean_logprob: float
    delta_mean_logprob: float


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
        self.direction_banks = {}
        self.directions = self._directions(enable_thinking=True)
        self.direction = self.directions["semantic"]
        print(
            "token-bank ids:",
            f"positive={len(self.positive_ids)}/{len(POSITIVE_BANK)}",
            f"negative={len(self.negative_ids)}/{len(NEGATIVE_BANK)}",
        )

    def _chat(self, text: str, enable_thinking: bool):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def _activation(self, text: str, enable_thinking: bool):
        inputs = self.tokenizer(
            self._chat(text, enable_thinking),
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

    def _build_direction(self, enable_thinking: bool):
        low = [
            self._activation(f"{prompt} {APPROACH_LOW}", enable_thinking)
            for prompt in TRAIN_PROMPTS
        ]
        high = [
            self._activation(f"{prompt} {APPROACH_HIGH}", enable_thinking)
            for prompt in TRAIN_PROMPTS
        ]
        return normalize(mean_delta(high, low))

    def _directions(self, enable_thinking: bool):
        key = bool(enable_thinking)
        if key not in self.direction_banks:
            label = "thinking" if key else "no-thinking exploratory"
            print(f"building {label} direction bank...")
            semantic = self._build_direction(key)
            bank = {"semantic": semantic}
            bank.update(
                {
                    f"random #{i + 1}": random_direction(semantic, seed)
                    for i, seed in enumerate(RANDOM_SEEDS)
                }
            )
            self.direction_banks[key] = bank
        return self.direction_banks[key]

    def _inputs(self, prompt: str, enable_thinking: bool):
        return self.tokenizer(
            self._chat(prompt, enable_thinking),
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.model.device)

    def _direction(self, direction_name: str, enable_thinking: bool):
        directions = self._directions(enable_thinking)
        if direction_name not in directions:
            raise ValueError(f"unknown direction: {direction_name}")
        return directions[direction_name]

    def _next_logits(
        self,
        inputs,
        alpha: float,
        direction_name: str = "semantic",
        enable_thinking: bool = True,
    ):
        handle = None
        if alpha != 0:
            handle = self.decoder_layers[self.target_layer].register_forward_hook(
                Hook(
                    self.torch,
                    self._direction(direction_name, enable_thinking),
                    alpha,
                    -1,
                )
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

    def score_candidates(
        self,
        prompt: str,
        candidates,
        alpha: float,
        mode: str = "boundary",
        direction_name: str = "semantic",
        enable_thinking: bool = True,
        allow_extended_alpha: bool = False,
        include_eos: bool = True,
    ):
        if not prompt.strip():
            raise ValueError("Prompt is empty")
        alpha_limit = 4.0 if allow_extended_alpha else 2.0
        if not -alpha_limit <= alpha <= alpha_limit:
            raise ValueError(
                f"alpha must be between {-alpha_limit:g} and {alpha_limit:g}"
            )
        if mode not in {"observe", "boundary", "continuous"}:
            raise ValueError("mode must be observe, boundary, or continuous")

        cleaned = [str(candidate).strip() for candidate in candidates]
        if len(cleaned) < 2 or any(not candidate for candidate in cleaned):
            raise ValueError("Provide at least two non-empty candidates")

        initial = self._inputs(prompt, enable_thinking)
        results = []
        for candidate in cleaned:
            candidate_ids = self.tokenizer.encode(
                candidate,
                add_special_tokens=False,
            )
            if include_eos:
                candidate_ids = [*candidate_ids, self.tokenizer.eos_token_id]
            full_ids = initial["input_ids"]
            baseline_logprobs = []
            steered_logprobs = []

            for step, target_id in enumerate(candidate_ids):
                current = {
                    "input_ids": full_ids,
                    "attention_mask": self.torch.ones_like(full_ids),
                }
                baseline_logits = self._next_logits(current, 0.0)
                apply_intervention = (
                    mode == "continuous"
                    or (mode == "boundary" and step == 0)
                )
                if apply_intervention and float(alpha) != 0.0:
                    steered_logits = self._next_logits(
                        current,
                        float(alpha),
                        direction_name,
                        enable_thinking,
                    )
                else:
                    steered_logits = baseline_logits
                baseline_logprobs.append(
                    float(
                        self.torch.log_softmax(baseline_logits, dim=-1)[
                            target_id
                        ].item()
                    )
                )
                steered_logprobs.append(
                    float(
                        self.torch.log_softmax(steered_logits, dim=-1)[
                            target_id
                        ].item()
                    )
                )
                next_token = self.torch.tensor(
                    [[target_id]],
                    device=full_ids.device,
                    dtype=full_ids.dtype,
                )
                full_ids = self.torch.cat([full_ids, next_token], dim=1)

            baseline_total = sum(baseline_logprobs)
            steered_total = sum(steered_logprobs)
            count = len(candidate_ids)
            results.append(
                CandidateScore(
                    candidate=candidate,
                    token_ids=candidate_ids,
                    raw_tokens=[
                        self.tokenizer.convert_ids_to_tokens(token_id)
                        for token_id in candidate_ids
                    ],
                    baseline_total_logprob=baseline_total,
                    steered_total_logprob=steered_total,
                    delta_total_logprob=steered_total - baseline_total,
                    baseline_mean_logprob=baseline_total / count,
                    steered_mean_logprob=steered_total / count,
                    delta_mean_logprob=(
                        steered_total - baseline_total
                    )
                    / count,
                )
            )
        return results

    def run(
        self,
        prompt: str,
        alpha: float,
        max_new_tokens: int = 40,
        direction_name: str = "semantic",
        enable_thinking: bool = True,
        allow_extended_alpha: bool = False,
    ) -> SteeringResult:
        if not prompt.strip():
            raise ValueError("Prompt is empty")
        alpha_limit = 4.0 if allow_extended_alpha else 2.0
        if not -alpha_limit <= alpha <= alpha_limit:
            raise ValueError(
                f"alpha must be between {-alpha_limit:g} and {alpha_limit:g}"
            )
        max_new_tokens = max(1, min(int(max_new_tokens), 96))
        inputs = self._inputs(prompt, enable_thinking)
        baseline_logits = self._next_logits(inputs, 0.0)
        steered_logits = self._next_logits(
            inputs, float(alpha), direction_name, enable_thinking
        )
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
            enable_thinking=bool(enable_thinking),
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
        enable_thinking: bool = True,
        allow_extended_alpha: bool = False,
    ):
        if not prompt.strip():
            raise ValueError("Prompt is empty")
        alpha_limit = 4.0 if allow_extended_alpha else 2.0
        if not -alpha_limit <= alpha <= alpha_limit:
            raise ValueError(
                f"alpha must be between {-alpha_limit:g} and {alpha_limit:g}"
            )
        if mode not in {"observe", "boundary", "continuous"}:
            raise ValueError("mode must be observe, boundary, or continuous")
        max_new_tokens = max(1, min(int(max_new_tokens), 96))
        initial = self._inputs(prompt, enable_thinking)
        full_ids = initial["input_ids"]
        trace_steps = []
        generated_token_ids = []

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
                    current, float(alpha), direction_name, enable_thinking
                )
            else:
                choice_logits = baseline_logits
            baseline_contrast = self._contrast(baseline_logits)
            steered_contrast = self._contrast(choice_logits)
            baseline_top = self.torch.topk(baseline_logits, k=2)
            steered_top = self.torch.topk(choice_logits, k=2)
            baseline_token_id = int(baseline_top.indices[0].item())
            token_id = int(choice_logits.argmax().item())
            baseline_token = self.tokenizer.decode(
                [baseline_token_id],
                skip_special_tokens=False,
            )
            token = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            )
            raw_token = self.tokenizer.convert_ids_to_tokens(token_id)
            generated_token_ids.append(token_id)
            cumulative_text = self.tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=False,
            )
            trace_steps.append(
                TraceStep(
                    step=step,
                    token=token,
                    raw_token=raw_token,
                    cumulative_text=cumulative_text,
                    token_id=token_id,
                    baseline_token=baseline_token,
                    baseline_token_id=baseline_token_id,
                    argmax_changed=baseline_token_id != token_id,
                    intervention_applied=(
                        apply_intervention and float(alpha) != 0.0
                    ),
                    baseline_contrast=baseline_contrast,
                    steered_contrast=steered_contrast,
                    delta=steered_contrast - baseline_contrast,
                    baseline_margin=float(
                        (baseline_top.values[0] - baseline_top.values[1]).item()
                    ),
                    steered_margin=float(
                        (steered_top.values[0] - steered_top.values[1]).item()
                    ),
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
            "enable_thinking": bool(enable_thinking),
            "extended_alpha": bool(allow_extended_alpha),
            "mode": mode,
            "alpha": float(alpha),
            "generated": generated,
            "steps": trace_steps,
        }
