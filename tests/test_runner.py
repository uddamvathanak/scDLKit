from __future__ import annotations

from pathlib import Path

from scdlkit import TaskRunner, compare_models
from scdlkit.training import Trainer


def test_trainer_records_history(dense_adata) -> None:
    from scdlkit import create_model, prepare_data

    prepared = prepare_data(dense_adata, label_key="cell_type")
    model = create_model(
        "autoencoder", input_dim=prepared.input_dim, latent_dim=8, hidden_dims=(16, 12)
    )
    trainer = Trainer(model=model, task="representation", epochs=2, batch_size=8)
    trainer.fit(prepared.train, prepared.val)
    assert len(trainer.history_) >= 1
    assert trainer.best_state_dict_ is not None


def test_taskrunner_representation_workflow(dense_adata, tmp_path: Path) -> None:
    runner = TaskRunner(
        model="vae",
        task="representation",
        epochs=2,
        batch_size=8,
        latent_dim=8,
        hidden_dims=(16, 12),
        label_key="cell_type",
        batch_key="batch",
        output_dir=str(tmp_path / "vae"),
    )
    runner.fit(dense_adata)
    metrics = runner.evaluate()
    assert "pearson" in metrics
    assert "silhouette" in metrics
    runner.plot_losses()
    runner.plot_latent(method="pca", color="label")
    runner.plot_reconstruction()
    reconstructed = runner.reconstruct(dense_adata)
    assert reconstructed.shape[0] == dense_adata.n_obs
    report_path = runner.save_report(tmp_path / "report.md")
    assert report_path.exists()
    assert report_path.with_suffix(".csv").exists()


def test_taskrunner_classification_workflow(dense_adata) -> None:
    runner = TaskRunner(
        model="mlp_classifier",
        task="classification",
        epochs=2,
        batch_size=8,
        hidden_dims=(16, 8),
        label_key="cell_type",
    )
    runner.fit(dense_adata)
    metrics = runner.evaluate()
    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics
    runner.plot_confusion_matrix()
    try:
        runner.reconstruct(dense_adata)
    except ValueError as exc:
        assert "reconstructed expression" in str(exc)
    else:
        raise AssertionError("Classification runner should not support reconstruct().")


def test_compare_models_writes_outputs(dense_adata, tmp_path: Path) -> None:
    result = compare_models(
        dense_adata,
        models=["autoencoder", "transformer_ae"],
        task="representation",
        shared_kwargs={
            "epochs": 1,
            "batch_size": 8,
            "latent_dim": 8,
            "hidden_dims": (16, 12),
            "label_key": "cell_type",
            "batch_key": "batch",
        },
        output_dir=str(tmp_path / "compare"),
    )
    assert set(result.metrics_frame["model"]) == {"autoencoder", "transformer_ae"}
    assert "runtime_sec" in result.metrics_frame.columns
    assert (tmp_path / "compare" / "benchmark_metrics.csv").exists()
    assert (tmp_path / "compare" / "benchmark_report.md").exists()
    assert (tmp_path / "compare" / "benchmark_comparison.png").exists()
