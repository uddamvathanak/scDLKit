"""MLP classification baseline."""

from __future__ import annotations

import torch

from scdlkit.models.base import BaseModel
from scdlkit.models.blocks import build_mlp
from scdlkit.models.registry import register_model


@register_model("mlp_classifier", "classifier")
class MLPClassifier(BaseModel):
    """Simple classifier over preprocessed expression features."""

    supported_tasks = ("classification",)

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
    ):
        super().__init__(input_dim=input_dim)
        self.network = build_mlp(input_dim, hidden_dims, output_dim=num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.network(x)
        return {"logits": logits}
