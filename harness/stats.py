from __future__ import annotations

import random
from statistics import mean
from typing import Sequence


def bootstrap_mean_ci(
    values: Sequence[float], *, confidence: float = 0.95, iterations: int = 2000, seed: int = 0
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        value = float(values[0])
        return value, value
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(mean(rng.choices(values, k=n)) for _ in range(iterations))
    tail = (1.0 - confidence) / 2
    return samples[int(tail * iterations)], samples[min(iterations - 1, int((1 - tail) * iterations))]


def paired_bootstrap_ci(
    left: Sequence[float], right: Sequence[float], *, confidence: float = 0.95, iterations: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    if len(left) != len(right):
        raise ValueError("Paired samples must have equal lengths.")
    if not left:
        return 0.0, 0.0, 0.0
    differences = [float(b) - float(a) for a, b in zip(left, right)]
    low, high = bootstrap_mean_ci(differences, confidence=confidence, iterations=iterations, seed=seed)
    return mean(differences), low, high


def paired_binary_test(left: Sequence[bool], right: Sequence[bool]) -> dict[str, float | int]:
    if len(left) != len(right):
        raise ValueError("Paired samples must have equal lengths.")
    discordant_left = sum(a and not b for a, b in zip(left, right))
    discordant_right = sum(not a and b for a, b in zip(left, right))
    discordant = discordant_left + discordant_right
    p_value = 1.0
    if discordant:
        smaller = min(discordant_left, discordant_right)
        p_value = min(1.0, 2 * sum(_binomial_probability(discordant, k) for k in range(smaller + 1)))
    return {
        "discordant_left": discordant_left,
        "discordant_right": discordant_right,
        "mcnemar_exact_p": p_value,
    }


def _binomial_probability(n: int, k: int) -> float:
    import math

    return math.comb(n, k) / (2**n)
