from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

external_evidence = importlib.import_module("run_external_annotation_evidence")


def _artifact_dir(root: Path, dataset: str, model: str) -> Path:
    path = root / dataset / model
    path.mkdir(parents=True, exist_ok=True)
    for filename in (
        "report.md",
        "report.csv",
        "batch_metrics.csv",
        "confusion_matrix.png",
        "latent_umap.png",
    ):
        (path / filename).write_text("ok", encoding="utf-8")
    pd.DataFrame([{"batch": "batch_0", "n_cells": 8, "accuracy": 0.80, "macro_f1": 0.75}]).to_csv(
        path / "batch_metrics.csv", index=False
    )
    if model != "pca_logistic_annotation" and model != "scgpt_frozen_probe":
        best_model_dir = path / "best_model"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        (best_model_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (best_model_dir / "model_state.pt").write_text("weights", encoding="utf-8")
    return path


def _rows_for_dataset(root: Path, dataset: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (model, macro_f1, accuracy, trainable_parameters) in enumerate(
        (
            ("pca_logistic_annotation", 0.70, 0.72, 0),
            ("scgpt_frozen_probe", 0.73, 0.74, 0),
            ("scgpt_head", 0.76, 0.77, 220),
            ("scgpt_full_finetune", 0.77, 0.78, 1200),
            ("scgpt_lora", 0.75, 0.76, 760),
            ("scgpt_adapter", 0.755, 0.765, 640),
            ("scgpt_prefix_tuning", 0.752, 0.762, 520),
            ("scgpt_ia3", 0.751, 0.761, 480),
        )
    ):
        artifact_dir = _artifact_dir(root, dataset, model)
        rows.append(
            {
                "dataset": dataset,
                "regime": "full_label",
                "model": model,
                "seed": 42,
                "runtime_sec": float(5 + index),
                "artifact_dir": str(artifact_dir),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "balanced_accuracy": macro_f1 - 0.01,
                "trainable_parameters": trainable_parameters,
                "batch_accuracy_mean": 0.80,
                "batch_accuracy_min": 0.78,
                "batch_macro_f1_mean": 0.75,
                "batch_macro_f1_min": 0.72,
                "strategy": external_evidence.MODEL_DISPLAY_NAMES[model],
                "best_model_artifact": (
                    str(artifact_dir / "best_model")
                    if model not in {"pca_logistic_annotation", "scgpt_frozen_probe"}
                    else None
                ),
                "batch_metrics_artifact": str(artifact_dir / "batch_metrics.csv"),
                "confusion_matrix_artifact": str(artifact_dir / "confusion_matrix.png"),
                "latent_umap_artifact": str(artifact_dir / "latent_umap.png"),
            }
        )
    return rows


def test_run_external_annotation_evidence_writes_required_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_root = tmp_path / "fake_benchmarks"
    fake_output = fake_root / "full_label"
    fake_output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        _rows_for_dataset(fake_root, "pbmc68k_reduced")
        + _rows_for_dataset(fake_root, "openproblems_human_pancreas")
    )
    frame.to_csv(fake_output / "strategy_metrics.csv", index=False)

    monkeypatch.setattr(
        external_evidence,
        "run_annotation_benchmark",
        lambda **kwargs: {"full_label_dir": fake_output},
    )

    output_dir = tmp_path / "evidence"
    outputs = external_evidence.run_external_annotation_evidence(output_dir=output_dir)

    assert outputs["pancreas_dir"] == output_dir / "pancreas"
    assert (output_dir / "pancreas" / "report.md").exists()
    assert (output_dir / "pancreas" / "report.csv").exists()
    assert (output_dir / "pancreas" / "strategy_metrics.csv").exists()
    assert (output_dir / "pancreas" / "batch_metrics.csv").exists()
    assert (output_dir / "pancreas" / "best_strategy_confusion_matrix.png").exists()
    assert (output_dir / "pancreas" / "frozen_embedding_umap.png").exists()
    assert (output_dir / "pancreas" / "best_strategy_embedding_umap.png").exists()
    assert (output_dir / "pancreas" / "best_model" / "manifest.json").exists()
    assert (output_dir / "pancreas" / "best_model" / "model_state.pt").exists()
    assert (output_dir / "cross_dataset" / "strategy_summary.csv").exists()
    assert (output_dir / "cross_dataset" / "strategy_summary.md").exists()
