from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from scdlkit.adapters import (
    wrap_classification_module,
    wrap_reconstruction_module,
)
from scdlkit.evaluation import evaluate_predictions
from scdlkit.training import Trainer


class TinyAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))


class ForwardOnlyAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)


class TinyClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def test_reconstruction_adapter_supports_auto_detected_encode(dense_adata) -> None:
    from scdlkit import prepare_data

    prepared = prepare_data(dense_adata, label_key="cell_type", batch_key="batch")
    module = TinyAutoencoder(prepared.input_dim, latent_dim=8)
    model = wrap_reconstruction_module(
        module,
        input_dim=prepared.input_dim,
        supported_tasks=("representation", "reconstruction"),
    )
    trainer = Trainer(model=model, task="representation", epochs=2, batch_size=8)
    trainer.fit(prepared.train, prepared.val)
    split = prepared.test or prepared.val or prepared.train
    predictions = trainer.predict_dataset(split)
    assert "reconstruction" in predictions
    assert "latent" in predictions
    metrics = evaluate_predictions("representation", predictions)
    assert "pearson" in metrics
    assert "silhouette" in metrics


def test_reconstruction_adapter_supports_explicit_encode_fn(dense_adata) -> None:
    from scdlkit import prepare_data

    prepared = prepare_data(dense_adata, label_key="cell_type")
    module = ForwardOnlyAutoencoder(prepared.input_dim, latent_dim=8)
    model = wrap_reconstruction_module(
        module,
        input_dim=prepared.input_dim,
        supported_tasks=("representation", "reconstruction"),
        encode_fn=lambda wrapped_module, x: wrapped_module.encoder(x),
    )
    trainer = Trainer(model=model, task="representation", epochs=1, batch_size=8)
    trainer.fit(prepared.train, prepared.val)
    split = prepared.test or prepared.val or prepared.train
    predictions = trainer.predict_dataset(split)
    assert predictions["latent"].shape[1] == 8


def test_reconstruction_adapter_custom_loss_surfaces_stats(dense_adata) -> None:
    from scdlkit import prepare_data

    prepared = prepare_data(dense_adata, label_key="cell_type")
    module = TinyAutoencoder(prepared.input_dim, latent_dim=8)
    model = wrap_reconstruction_module(
        module,
        input_dim=prepared.input_dim,
        supported_tasks=("reconstruction",),
        loss_fn=lambda _module, batch, outputs: (
            torch.mean(torch.abs(outputs["reconstruction"] - batch["x"])),
            {
                "mae_loss": float(
                    torch.mean(torch.abs(outputs["reconstruction"] - batch["x"]))
                    .detach()
                    .cpu()
                )
            },
        ),
    )
    trainer = Trainer(model=model, task="reconstruction", epochs=1, batch_size=8)
    trainer.fit(prepared.train, prepared.val)
    history_columns = set(trainer.history_frame.columns)
    assert "train_mae_loss" in history_columns


def test_classification_adapter_default_and_custom_loss(dense_adata) -> None:
    from scdlkit import prepare_data

    prepared = prepare_data(dense_adata, label_key="cell_type")
    num_classes = len(prepared.label_encoder or {})
    module = TinyClassifier(prepared.input_dim, num_classes=num_classes)
    model = wrap_classification_module(module, input_dim=prepared.input_dim)
    trainer = Trainer(model=model, task="classification", epochs=2, batch_size=8)
    trainer.fit(prepared.train, prepared.val)
    split = prepared.test or prepared.val or prepared.train
    predictions = trainer.predict_dataset(split)
    assert "logits" in predictions
    metrics = evaluate_predictions("classification", predictions)
    assert "accuracy" in metrics

    custom_model = wrap_classification_module(
        TinyClassifier(prepared.input_dim, num_classes=num_classes),
        input_dim=prepared.input_dim,
        loss_fn=lambda _module, batch, outputs: (
            torch.nn.functional.cross_entropy(outputs["logits"], batch["y"]),
            {"custom_loss": 1.0},
        ),
    )
    custom_trainer = Trainer(model=custom_model, task="classification", epochs=1, batch_size=8)
    custom_trainer.fit(prepared.train, prepared.val)
    assert "train_custom_loss" in set(custom_trainer.history_frame.columns)


def test_reconstruction_adapter_rejects_invalid_supported_tasks() -> None:
    with pytest.raises(ValueError, match="only supports"):
        wrap_reconstruction_module(
            TinyAutoencoder(20, latent_dim=4),
            input_dim=20,
            supported_tasks=("classification",),
        )


def test_representation_adapter_requires_latent_path() -> None:
    model = wrap_reconstruction_module(
        ForwardOnlyAutoencoder(20, latent_dim=4),
        input_dim=20,
        supported_tasks=("representation", "reconstruction"),
    )
    with pytest.raises(ValueError, match="latent path"):
        model(torch.randn(4, 20))


def test_classification_adapter_requires_logits_tensor() -> None:
    module = TinyClassifier(20, num_classes=2)
    model = wrap_classification_module(
        module,
        input_dim=20,
        forward_fn=lambda _module, x: {"scores": x},
    )
    with pytest.raises(ValueError, match="'logits'"):
        model(torch.randn(4, 20))


def test_reconstruction_adapter_requires_reconstruction_tensor() -> None:
    module = TinyAutoencoder(20, latent_dim=4)
    model = wrap_reconstruction_module(
        module,
        input_dim=20,
        forward_fn=lambda _module, x: {"decoded": x},
    )
    with pytest.raises(ValueError, match="'reconstruction'"):
        model(torch.randn(4, 20))


def test_trainer_validates_supported_tasks_early(dense_adata) -> None:
    from scdlkit import prepare_data

    prepared = prepare_data(dense_adata, label_key="cell_type")
    num_classes = len(prepared.label_encoder or {})
    classifier = wrap_classification_module(
        TinyClassifier(prepared.input_dim, num_classes=num_classes),
        input_dim=prepared.input_dim,
    )
    with pytest.raises(ValueError, match="does not support task 'representation'"):
        Trainer(model=classifier, task="representation", epochs=1, batch_size=8)

    recon = wrap_reconstruction_module(
        TinyAutoencoder(prepared.input_dim, latent_dim=8),
        input_dim=prepared.input_dim,
        supported_tasks=("representation", "reconstruction"),
    )
    with pytest.raises(ValueError, match="does not support task 'classification'"):
        Trainer(model=recon, task="classification", epochs=1, batch_size=8)


def test_evaluate_predictions_works_with_adapter_outputs() -> None:
    predictions = {
        "x": np.random.default_rng(1).normal(size=(8, 6)).astype("float32"),
        "reconstruction": np.random.default_rng(2).normal(size=(8, 6)).astype("float32"),
        "latent": np.random.default_rng(3).normal(size=(8, 3)).astype("float32"),
        "y": np.array([0, 1, 0, 1, 0, 1, 0, 1]),
        "batch": np.array([0, 0, 1, 1, 0, 0, 1, 1]),
    }
    metrics = evaluate_predictions("representation", predictions)
    assert "pearson" in metrics
    assert "silhouette" in metrics
