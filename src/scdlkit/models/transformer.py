"""Patch-based transformer autoencoder for tabular single-cell inputs."""

from __future__ import annotations

import math

import torch
from torch import nn

from scdlkit.models.base import BaseModel
from scdlkit.models.blocks import build_mlp
from scdlkit.models.registry import register_model


@register_model("transformer_ae", "transformer_autoencoder")
class TransformerAutoEncoder(BaseModel):
    """Patch-based transformer autoencoder."""

    supported_tasks = ("representation", "reconstruction")

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, ...] | None = None,
        patch_size: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        decoder_hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ):
        super().__init__(input_dim=input_dim)
        del hidden_dims
        self.patch_size = patch_size
        self.num_patches = math.ceil(input_dim / patch_size)
        self.padded_dim = self.num_patches * patch_size
        self.patch_projection = nn.Linear(patch_size, d_model)
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.to_latent = nn.Linear(d_model, latent_dim)
        self.decoder = build_mlp(
            latent_dim, decoder_hidden_dims, output_dim=input_dim, dropout=dropout
        )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        if self.padded_dim > self.input_dim:
            pad_width = self.padded_dim - self.input_dim
            x = torch.nn.functional.pad(x, (0, pad_width))
        return x.view(x.size(0), self.num_patches, self.patch_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        patches = self._patchify(x)
        tokens = self.patch_projection(patches) + self.positional_embedding
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.to_latent(pooled)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return {"latent": latent, "reconstruction": reconstruction}
