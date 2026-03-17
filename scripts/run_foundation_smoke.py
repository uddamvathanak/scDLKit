"""Run a real-checkpoint scGPT embedding smoke test on a small PBMC subset."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scdlkit.evaluation import evaluate_predictions  # noqa: E402
from scdlkit.evaluation.metrics import classification_metrics  # noqa: E402
from scdlkit.foundation import load_scgpt_model, prepare_scgpt_data  # noqa: E402
from scdlkit.training import Trainer  # noqa: E402


def _linear_probe_metrics(latent: np.ndarray, labels: np.ndarray, *, seed: int) -> dict[str, float]:
    _, counts = np.unique(labels, return_counts=True)
    stratify = labels if int(counts.min()) >= 2 else None
    train_x, test_x, train_y, test_y = train_test_split(
        latent,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=stratify,
    )
    classifier = LogisticRegression(max_iter=1000, random_state=seed)
    classifier.fit(train_x, train_y)
    logits = classifier.predict_proba(test_x)
    metrics = classification_metrics(test_y, logits)
    return {
        "probe_accuracy": float(metrics["accuracy"]),
        "probe_macro_f1": float(metrics["macro_f1"]),
    }


def main() -> None:
    output_dir = ROOT / "artifacts" / "foundation_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.datasets.pbmc3k_processed()
    rng = np.random.default_rng(42)
    subset_indices = np.sort(rng.choice(adata.n_obs, size=min(128, adata.n_obs), replace=False))
    adata = adata[subset_indices].copy()

    prepared = prepare_scgpt_data(
        adata,
        checkpoint="whole-human",
        label_key="louvain",
        batch_size=64,
        use_raw=True,
    )
    model = load_scgpt_model("whole-human", device="auto")
    trainer = Trainer(
        model=model,
        task="representation",
        batch_size=prepared.batch_size,
        device="auto",
        epochs=1,
    )

    started_at = perf_counter()
    predictions = trainer.predict_dataset(prepared.dataset)
    runtime_sec = perf_counter() - started_at

    metrics = evaluate_predictions("representation", predictions)
    metrics.update(_linear_probe_metrics(predictions["latent"], predictions["y"], seed=42))
    summary = {
        "dataset": "pbmc3k_processed",
        "subset_cells": int(adata.n_obs),
        "checkpoint": "whole-human",
        "runtime_sec": runtime_sec,
        "num_genes_matched": prepared.num_genes_matched,
        "metrics": metrics,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# scDLKit foundation smoke summary",
        "",
        "- Dataset: `pbmc3k_processed`",
        f"- Cells: `{adata.n_obs}`",
        "- Checkpoint: `whole-human`",
        f"- Runtime: `{runtime_sec:.2f}s`",
        f"- Matched genes: `{prepared.num_genes_matched}`",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(f"- `{key}`: `{value:.4f}`" for key, value in metrics.items())
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
