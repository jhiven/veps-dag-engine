"""Reproducible matched-block order schedules."""

from __future__ import annotations

import itertools

from benchmarks.ordering import balanced_order


def test_three_treatments_cover_all_six_orders() -> None:
    items = ("Stop", "Pause", "VEPS")
    orders = tuple(balanced_order(items, 42, "gpu", rep) for rep in range(1, 7))
    assert set(orders) == set(itertools.permutations(items))
    assert orders == tuple(balanced_order(items, 42, "gpu", rep) for rep in range(1, 7))
    thirty = tuple(balanced_order(items, 42, "gpu", rep) for rep in range(1, 31))
    for item in items:
        for position in range(3):
            assert sum(order[position] == item for order in thirty) == 10


def test_four_treatments_form_a_latin_square() -> None:
    items = ("A", "B", "C", "D")
    orders = tuple(balanced_order(items, 42, "ablation", rep) for rep in range(1, 5))
    for item in items:
        assert {order.index(item) for order in orders} == {0, 1, 2, 3}
