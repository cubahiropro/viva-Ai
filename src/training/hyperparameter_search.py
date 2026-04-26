"""Lightweight grid-style hyperparameter search.

This is a deliberately simple sweep over learning rate and dropout — enough to
satisfy section 6.4 / phase-4 instructions without heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable


@dataclass
class HPGridPoint:
    learning_rate: float
    dropout: float


def hp_grid(
    learning_rates: Iterable[float] = (1e-3, 3e-4, 1e-4),
    dropouts: Iterable[float] = (0.2, 0.3),
) -> list[HPGridPoint]:
    return [HPGridPoint(lr, d) for lr, d in product(learning_rates, dropouts)]
