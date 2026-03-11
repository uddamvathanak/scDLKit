"""Reusable neural network building blocks."""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn


def build_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    *,
    output_dim: int | None = None,
    dropout: float = 0.0,
    final_activation: nn.Module | None = None,
) -> nn.Sequential:
    """Build an MLP with GELU activations."""

    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    if output_dim is not None:
        layers.append(nn.Linear(prev_dim, output_dim))
        if final_activation is not None:
            layers.append(final_activation)
    return nn.Sequential(*layers)
