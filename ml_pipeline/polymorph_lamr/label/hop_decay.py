"""Hop-decay kernels for AST distance weighting.

Given the hop distance `h` from a token's leaf node to the nearest structural
scaffold node, produce `w_dep ∈ [0, 1]`. `w_sem = 1 - w_dep`.
"""

from __future__ import annotations

import math
from typing import Callable


def exp_decay(alpha: float = 0.5) -> Callable[[int], float]:
    """w(h) = exp(-α · h). Defaults to a soft half-life around h ≈ 1.4."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    def kernel(h: int) -> float:
        return math.exp(-alpha * max(0, h))

    return kernel


def linear_decay(max_hops: int = 6) -> Callable[[int], float]:
    """w(h) = max(0, 1 - h/max_hops). Sharper cutoff than exp."""
    if max_hops <= 0:
        raise ValueError("max_hops must be positive")

    def kernel(h: int) -> float:
        return max(0.0, 1.0 - h / max_hops)

    return kernel


def get_kernel(spec: dict) -> Callable[[int], float]:
    """Build a kernel from the config dict shape used in configs/default.yaml."""
    kind = spec.get("kernel", "exp")
    if kind == "exp":
        return exp_decay(alpha=float(spec.get("alpha", 0.5)))
    if kind == "linear":
        return linear_decay(max_hops=int(spec.get("max_hops", 6)))
    raise ValueError(f"unknown hop-decay kernel: {kind}")
