"""Generic internal protocols for foundation-model adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from torch import Tensor, nn

from scdlkit.foundation.peft import AnnotationStrategy, PEFTConfig


class FoundationModelAdapter(Protocol):
    """Internal protocol for a foundation-model wrapper."""

    checkpoint_id: str

    def encode_batch(self, batch: Mapping[str, Tensor]) -> Tensor:
        """Return pooled latent embeddings for a prepared token batch."""


class FoundationAnnotationAdapter(Protocol):
    """Internal protocol for annotation-capable foundation adapters."""

    supported_strategies: tuple[AnnotationStrategy, ...]

    def build_model(
        self,
        *,
        num_classes: int,
        strategy: AnnotationStrategy,
        strategy_config: PEFTConfig | None,
        label_categories: tuple[str, ...] | None,
        classifier_dropout: float,
        device: str,
        cache_dir: str | Path | None,
    ) -> nn.Module:
        """Build a strategy-specific annotation model."""

    def serialize_strategy_config(self, config: PEFTConfig | None) -> dict[str, Any] | None:
        """Serialize a strategy config for persistence."""
