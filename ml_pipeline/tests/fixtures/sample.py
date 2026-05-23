"""A small Python file used as a fixture for the AST-split tests."""

from __future__ import annotations

import math


def compute_area(radius: float) -> float:
    """Return the area of a circle. This docstring is prose."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    return math.pi * radius * radius


class Circle:
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return compute_area(self.radius)
