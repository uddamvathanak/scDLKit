"""Variational autoencoder baseline."""

from __future__ import annotations

import torch
from torch import nn

from scdlkit.models.base import BaseModel
from scdlkit.models.blocks import build_mlp
from scdlkit.models.registry import register_model


@register_model("vae")
class VariationalAutoEncoder(BaseModel):
    """Variational autoencoder with an MLP encoder/decoder."""

    supported_tasks = ("representation", "reconstruction")

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
        kl_weight: float = 1.0,
    ):
        super().__init__(input_dim=input_dim)
        self.encoder = build_mlp(input_dim, hidden_dims, dropout=dropout)
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar = nn.Linear(hidden_dims[-1], latent_dim)
        self.decoder = build_mlp(
            latent_dim, reversed(hidden_dims), output_dim=input_dim, dropout=dropout
        )
        self.kl_weight = kl_weight

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)
        return self.mu(hidden)

    def _encode_distribution(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.mu(hidden), self.logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self._encode_distribution(x)
        latent = self.reparameterize(mu, logvar)
        reconstruction = self.decode(latent)
        return {
            "latent": latent,
            "mu": mu,
            "logvar": logvar,
            "reconstruction": reconstruction,
        }
