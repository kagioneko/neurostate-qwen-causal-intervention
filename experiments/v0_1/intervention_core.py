from __future__ import annotations

import math
import random
from typing import Sequence

PROMPTS = (
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
    "Write one short sentence about a lamp at dusk.",
    "Write one short sentence about washing a cup.",
    "Write one short sentence about reading a map.",
    "Write one short sentence about hearing distant traffic.",
    "Write one short sentence about folding a towel.",
    "Write one short sentence about checking the time.",
    "Write one short sentence about a garden path.",
    "Write one short sentence about closing a drawer.",
    "Write one short sentence about sharpening a pencil.",
    "Write one short sentence about waiting for an elevator.",
)

ACTIVE_WORDS = (" move", " act", " forward", " begin", " go", " start")
CALM_WORDS = (" rest", " calm", " quiet", " wait", " pause", " still")


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(vector: Sequence[float]) -> float:
    return math.sqrt(dot(vector, vector))


def normalize(vector: Sequence[float]) -> list[float]:
    length = norm(vector)
    if not length:
        raise ValueError("cannot normalize zero vector")
    return [value / length for value in vector]


def orthogonal_random_direction(reference: Sequence[float], seed: int) -> list[float]:
    unit_reference = normalize(reference)
    rng = random.Random(seed)
    candidate = [rng.gauss(0.0, 1.0) for _ in unit_reference]
    projection = dot(candidate, unit_reference)
    orthogonal = [value - projection * axis for value, axis in zip(candidate, unit_reference)]
    return normalize(orthogonal)


def linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("slope requires varying x")
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def exact_sign_flip_p(values: Sequence[float]) -> float:
    observed = abs(sum(values) / len(values))
    exceed = 0
    total = 1 << len(values)
    for mask in range(total):
        permuted = sum((-value if mask & (1 << i) else value) for i, value in enumerate(values)) / len(values)
        exceed += abs(permuted) >= observed - 1e-12
    return exceed / total


def keyword_counts(text: str) -> dict[str, int]:
    lowered = " " + text.lower()
    return {
        "active_word_count": sum(lowered.count(word) for word in ACTIVE_WORDS),
        "calm_word_count": sum(lowered.count(word) for word in CALM_WORDS),
    }
