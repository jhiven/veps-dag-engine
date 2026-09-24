"""Stable, position-balanced treatment orders for matched benchmark blocks."""

from __future__ import annotations

import hashlib
import itertools
import random
from typing import TypeVar

T = TypeVar("T")


def balanced_order(items: tuple[T, ...], seed: int, label: str, repetition: int) -> tuple[T, ...]:
    if not items or repetition < 1:
        raise ValueError("items must be non-empty and repetition must be positive")
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    if len(items) == 3:
        # Six repetitions cover every permutation once. In a 30-block GPU
        # campaign each mechanism therefore occupies each position ten times.
        orders = list(itertools.permutations(items))
        rng.shuffle(orders)
        return orders[(repetition - 1) % len(orders)]
    base = list(items)
    rng.shuffle(base)
    shift = (repetition - 1) % len(base)
    return tuple(base[shift:] + base[:shift])
