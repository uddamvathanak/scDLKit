from __future__ import annotations

import torch

from scdlkit import create_model


def test_model_registry_resolves_aliases() -> None:
    model = create_model("ae", input_dim=20, latent_dim=8)
    assert model.__class__.__name__ == "AutoEncoder"


def test_autoencoder_family_shapes() -> None:
    batch = torch.randn(4, 20)
    for name in ("autoencoder", "vae", "denoising_autoencoder", "transformer_ae"):
        model = create_model(name, input_dim=20, latent_dim=8, hidden_dims=(16, 12))
        outputs = model(batch)
        assert outputs["reconstruction"].shape == (4, 20)
        assert outputs["latent"].shape[0] == 4


def test_classifier_shape() -> None:
    batch = torch.randn(4, 20)
    model = create_model("mlp_classifier", input_dim=20, num_classes=3)
    outputs = model(batch)
    assert outputs["logits"].shape == (4, 3)
