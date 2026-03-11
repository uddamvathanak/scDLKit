"""Dataclasses for prepared datasets and preprocessing metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class SplitData:
    """One dataset split with optional encoded labels and batches."""

    X: Any
    labels: np.ndarray | None = None
    batches: np.ndarray | None = None
    obs_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.X.shape[0])


@dataclass(slots=True)
class PreparedData:
    """Prepared train/validation/test splits and metadata."""

    train: SplitData
    val: SplitData | None
    test: SplitData | None
    input_dim: int
    feature_names: list[str]
    label_encoder: dict[str, int] | None
    batch_encoder: dict[str, int] | None
    preprocessing: dict[str, Any]
