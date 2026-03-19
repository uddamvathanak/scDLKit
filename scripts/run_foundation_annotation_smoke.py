"""Run a real-checkpoint scGPT annotation fine-tuning smoke test."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scanpy as sc
from scipy import sparse
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scdlkit.evaluation.metrics import classification_metrics  # noqa: E402
from scdlkit.foundation import (  # noqa: E402
    ScGPTLoRAConfig,
    load_scgpt_annotation_model,
    load_scgpt_model,
    prepare_scgpt_data,
    split_scgpt_data,
)
from scdlkit.training import Trainer  # noqa: E402


def _count_trainable_parameters(model: object) -> int:
    parameters = getattr(model, "parameters", None)
    if parameters is None:
        return 0
    return int(sum(parameter.numel() for parameter in parameters() if parameter.requires_grad))


def _expand_probabilities(
    probabilities: np.ndarray,
    classes: np.ndarray | list[int],
    *,
    num_classes: int,
) -> np.ndarray:
    expanded = np.zeros((probabilities.shape[0], num_classes), dtype=np.float32)
    expanded[:, np.asarray(classes, dtype=int)] = np.asarray(probabilities, dtype=np.float32)
    return expanded


def _subset_pbmc(max_cells: int = 8) -> sc.AnnData:
    adata = sc.datasets.pbmc3k_processed()
    if adata.n_obs <= max_cells:
        return adata.copy()
    labels = adata.obs["louvain"].astype(str).to_numpy()
    rng = np.random.default_rng(42)
    indices = np.arange(adata.n_obs)
    sampled = []
    for label in np.unique(labels):
        label_indices = indices[labels == label]
        take = max(1, int(round(max_cells * (len(label_indices) / len(indices)))))
        sampled.extend(rng.choice(label_indices, size=min(take, len(label_indices)), replace=False))
    sampled = np.array(sorted({int(index) for index in sampled}), dtype=int)
    if len(sampled) > max_cells:
        sampled = np.sort(rng.choice(sampled, size=max_cells, replace=False))
    return adata[sampled].copy()


def _subset_genes(adata: sc.AnnData, max_genes: int = 32) -> sc.AnnData:
    if adata.n_vars <= max_genes:
        return adata.copy()
    source = adata.raw.to_adata() if adata.raw is not None else adata.copy()
    matrix = source.X.toarray() if sparse.issparse(source.X) else np.asarray(source.X)
    variances = np.var(matrix, axis=0)
    keep_indices = np.argsort(variances)[-max_genes:]
    subset = source[:, np.sort(keep_indices)].copy()
    subset.raw = subset.copy()
    return subset


def _build_split():
    adata = _subset_genes(_subset_pbmc(), max_genes=32)
    prepared = prepare_scgpt_data(
        adata,
        checkpoint="whole-human",
        label_key="louvain",
        batch_size=64,
        use_raw=True,
        min_gene_overlap=16,
    )
    split = split_scgpt_data(prepared, val_size=0.15, test_size=0.15, random_state=42)
    return adata, prepared, split


def _run_frozen_probe(prepared, split) -> tuple[dict[str, float], float]:
    model = load_scgpt_model("whole-human", device="auto")
    trainer = Trainer(
        model=model,
        task="representation",
        batch_size=prepared.batch_size,
        device="auto",
        epochs=1,
    )
    started = perf_counter()
    train_predictions = trainer.predict_dataset(split.train)
    test_predictions = trainer.predict_dataset(split.test)
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(train_predictions["latent"], train_predictions["y"])
    logits = _expand_probabilities(
        classifier.predict_proba(test_predictions["latent"]),
        classifier.classes_,
        num_classes=len(prepared.label_categories or ()),
    )
    metrics = classification_metrics(test_predictions["y"], logits)
    runtime_sec = perf_counter() - started
    return {
        "probe_accuracy": float(metrics["accuracy"]),
        "probe_macro_f1": float(metrics["macro_f1"]),
    }, runtime_sec


def _run_tuned_strategy(
    prepared,
    split,
    *,
    tuning_strategy: str,
) -> tuple[dict[str, float], int, float]:
    model = load_scgpt_annotation_model(
        num_classes=len(prepared.label_categories or ()),
        checkpoint="whole-human",
        tuning_strategy=tuning_strategy,
        label_categories=prepared.label_categories,
        device="auto",
        lora_config=(
            ScGPTLoRAConfig(
                rank=4,
                alpha=8.0,
                dropout=0.05,
                target_modules=("linear1", "linear2"),
            )
            if tuning_strategy == "lora"
            else None
        ),
    )
    trainer = Trainer(
        model=model,
        task="classification",
        batch_size=prepared.batch_size,
        epochs=1,
        lr=5e-3 if tuning_strategy == "head" else 2e-3,
        device="auto",
        early_stopping_patience=3,
        seed=42,
    )
    started = perf_counter()
    trainer.fit(split.train, split.val)
    predictions = trainer.predict_dataset(split.test)
    metrics = classification_metrics(predictions["y"], predictions["logits"])
    runtime_sec = perf_counter() - started
    return (
        {
            f"{tuning_strategy}_accuracy": float(metrics["accuracy"]),
            f"{tuning_strategy}_macro_f1": float(metrics["macro_f1"]),
        },
        _count_trainable_parameters(model),
        runtime_sec,
    )


def main() -> None:
    output_dir = ROOT / "artifacts" / "foundation_annotation_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    adata, prepared, split = _build_split()
    probe_metrics, probe_runtime = _run_frozen_probe(prepared, split)
    head_metrics, head_trainable, head_runtime = _run_tuned_strategy(
        prepared,
        split,
        tuning_strategy="head",
    )
    lora_metrics, lora_trainable, lora_runtime = _run_tuned_strategy(
        prepared,
        split,
        tuning_strategy="lora",
    )

    summary = {
        "dataset": "pbmc3k_processed",
        "subset_cells": int(adata.n_obs),
        "checkpoint": "whole-human",
        "num_genes_matched": int(prepared.num_genes_matched),
        "probe_accuracy": probe_metrics["probe_accuracy"],
        "probe_macro_f1": probe_metrics["probe_macro_f1"],
        "head_accuracy": head_metrics["head_accuracy"],
        "head_macro_f1": head_metrics["head_macro_f1"],
        "lora_accuracy": lora_metrics["lora_accuracy"],
        "lora_macro_f1": lora_metrics["lora_macro_f1"],
        "trainable_parameters_head": head_trainable,
        "trainable_parameters_lora": lora_trainable,
        "probe_runtime_sec": probe_runtime,
        "head_runtime_sec": head_runtime,
        "lora_runtime_sec": lora_runtime,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# scDLKit foundation annotation smoke summary",
        "",
        "- Dataset: `pbmc3k_processed`",
        f"- Cells: `{adata.n_obs}`",
        "- Checkpoint: `whole-human`",
        f"- Matched genes: `{prepared.num_genes_matched}`",
        "",
        "## Metrics",
        "",
    ]
    metric_keys = (
        "probe_accuracy",
        "probe_macro_f1",
        "head_accuracy",
        "head_macro_f1",
        "lora_accuracy",
        "lora_macro_f1",
        "trainable_parameters_head",
        "trainable_parameters_lora",
        "probe_runtime_sec",
        "head_runtime_sec",
        "lora_runtime_sec",
    )
    for key in metric_keys:
        value = summary[key]
        if isinstance(value, float):
            lines.append(f"- `{key}`: `{value:.4f}`")
        else:
            lines.append(f"- `{key}`: `{value}`")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
