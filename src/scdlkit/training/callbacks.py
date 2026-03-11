"""Callback placeholders for future extension."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EarlyStoppingState:
    best_loss: float
    best_epoch: int
    epochs_without_improvement: int = 0
