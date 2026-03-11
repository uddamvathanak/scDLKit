"""MLP autoencoder baseline."""

from __future__ import annotations

import torch
from torch import nn

from scdlkit.models.base import BaseModel
from scdlkit.models.blocks import build_mlp
from scdlkit.models.registry import register_model


@register_model("autoencoder", "ae")
class AutoEncoder(BaseModel):
    """Simple MLP autoencoder."""

    supported_tasks = ("representation", "reconstruction")

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
    ):
        super().__init__(input_dim=input_dim)
        self.encoder = build_mlp(input_dim, hidden_dims, dropout=dropout)
        self.to_latent = nn.Linear(hidden_dims[-1], latent_dim)
        self.decoder = build_mlp(
            latent_dim, reversed(hidden_dims), output_dim=input_dim, dropout=dropout
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)
        return self.to_latent(hidden)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return {"latent": latent, "reconstruction": reconstruction}
