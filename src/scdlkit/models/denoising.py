"""Denoising autoencoder baseline."""

from __future__ import annotations

import torch

from scdlkit.models.autoencoder import AutoEncoder
from scdlkit.models.registry import register_model


@register_model("denoising_autoencoder", "dae")
class DenoisingAutoEncoder(AutoEncoder):
    """Autoencoder with input masking noise during training."""

    supported_tasks = ("representation", "reconstruction")

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
        noise_probability: float = 0.15,
    ):
        super().__init__(
            input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims, dropout=dropout
        )
        self.noise_probability = noise_probability

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        noisy_x = x
        if self.training and self.noise_probability > 0:
            mask = torch.rand_like(x) > self.noise_probability
            noisy_x = x * mask
        latent = self.encode(noisy_x)
        reconstruction = self.decode(latent)
        return {"latent": latent, "reconstruction": reconstruction}
