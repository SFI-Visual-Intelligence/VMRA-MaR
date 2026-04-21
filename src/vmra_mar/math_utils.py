"""Small math helpers for the metadata-backed model stack."""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence


def seeded_random(seed: int) -> random.Random:
    return random.Random(seed)


def zeros(length: int) -> list[float]:
    return [0.0 for _ in range(length)]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def add(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [a + b for a, b in zip(left, right)]


def subtract(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def scale(values: Sequence[float], factor: float) -> list[float]:
    return [factor * value for value in values]


def average(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    total = [0.0] * dim
    for vector in vectors:
        for idx, value in enumerate(vector):
            total[idx] += value
    count = float(len(vectors))
    return [value / count for value in total]


def stable_mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / float(len(values)) if values else 0.0


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def tanh(value: float) -> float:
    return math.tanh(value)


def relu(value: float) -> float:
    return value if value > 0.0 else 0.0


def softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def l2_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def euclidean_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) * (a - b) for a, b in zip(left, right)))


def cumulative_sum(values: Sequence[float]) -> list[float]:
    running = 0.0
    output: list[float] = []
    for value in values:
        running += value
        output.append(running)
    return output


def build_matrix(rng: random.Random, rows: int, cols: int, scale_hint: float = 1.0) -> list[list[float]]:
    limit = scale_hint / max(1.0, math.sqrt(cols))
    return [[rng.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


def build_vector(rng: random.Random, length: int, scale_hint: float = 1.0) -> list[float]:
    limit = scale_hint / max(1.0, math.sqrt(length))
    return [rng.uniform(-limit, limit) for _ in range(length)]


def matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float], bias: Sequence[float] | None = None) -> list[float]:
    output = []
    for row_index, row in enumerate(matrix):
        value = dot(row, vector)
        if bias is not None:
            value += bias[row_index]
        output.append(value)
    return output
