"""Internal quality suite for benchmark and tutorial release-gate checks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import matplotlib
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

matplotlib.use("Agg")

QUALITY_GATES: dict[str, float] = {
    "pbmc_vae_silhouette_min": 0.12,
    "pbmc_vae_knn_label_consistency_min": 0.85,
    "pbmc_vae_pearson_min": 0.15,
    "pbmc_vae_silhouette_std_max": 0.05,
    "pbmc_classifier_accuracy_min": 0.85,
    "pbmc_classifier_macro_f1_min": 0.80,
    "scgpt_vs_pca_max_drop": 0.02,
    "scgpt_silhouette_win": 0.01,
    "scgpt_annotation_head_accuracy_drop_max": 0.02,
    "scgpt_annotation_head_macro_f1_drop_max": 0.02,
    "scgpt_annotation_lora_accuracy_drop_max": 0.02,
    "scgpt_annotation_lora_macro_f1_drop_max": 0.02,
    "scgpt_annotation_tuned_accuracy_min": 0.80,
    "scgpt_annotation_tuned_macro_f1_min": 0.40,
}

RUNTIME_BUDGETS: dict[str, dict[str, float]] = {
    "ci": {
        "quality_suite_total_sec": 300.0,
        "tutorial_total_sec": 480.0,
        "transformer_ae_warn_sec": 15.0,
        "transformer_ae_fail_sec": 25.0,
        "scgpt_head_warn_sec": 20.0,
        "scgpt_head_fail_sec": 35.0,
        "scgpt_lora_warn_sec": 90.0,
        "scgpt_lora_fail_sec": 120.0,
    },
    "full": {
        "quality_suite_total_sec": 1200.0,
        "tutorial_total_sec": 1200.0,
        "transformer_ae_warn_sec": 30.0,
        "transformer_ae_fail_sec": 60.0,
        "scgpt_head_warn_sec": 40.0,
        "scgpt_head_fail_sec": 70.0,
        "scgpt_lora_warn_sec": 180.0,
        "scgpt_lora_fail_sec": 240.0,
    },
}

PROFILE_DEFAULTS: dict[str, dict[str, dict[str, dict[str, tuple[int, ...]]]]] = {
    "ci": {
        "representation": {
            "pbmc3k_processed": {
                "pca": (42,),
                "autoencoder": (42,),
                "vae": (42, 52, 62),
                "transformer_ae": (42,),
            },
            "paul15": {
                "pca": (42,),
                "vae": (42,),
            },
        },
        "classification": {
            "pbmc3k_processed": {
                "mlp_classifier": (42,),
                "logistic_regression_pca": (42,),
            },
        },
        "foundation": {
            "pbmc3k_processed": {
                "pca_foundation": (42,),
                "scgpt_whole_human": (42,),
            },
            "pbmc68k_reduced": {
                "pca_foundation": (42,),
                "scgpt_whole_human": (42,),
            },
        },
        "foundation_annotation": {
            "pbmc3k_processed": {
                "pca_logistic_annotation": (42,),
                "scgpt_frozen_probe": (42,),
                "scgpt_head": (42,),
            },
        },
    },
    "full": {
        "representation": {
            "pbmc3k_processed": {
                "pca": (42, 52, 62),
                "autoencoder": (42, 52, 62),
                "vae": (42, 52, 62),
                "transformer_ae": (42, 52, 62),
            },
            "paul15": {
                "pca": (42, 52, 62),
                "autoencoder": (42, 52, 62),
                "vae": (42, 52, 62),
            },
            "moignard15": {
                "pca": (42,),
                "vae": (42,),
            },
        },
        "classification": {
            "pbmc3k_processed": {
                "mlp_classifier": (42, 52, 62),
                "logistic_regression_pca": (42, 52, 62),
            },
        },
        "foundation": {
            "pbmc3k_processed": {
                "pca_foundation": (42,),
                "scgpt_whole_human": (42,),
            },
            "pbmc68k_reduced": {
                "pca_foundation": (42,),
                "scgpt_whole_human": (42,),
            },
        },
        "foundation_annotation": {
            "pbmc3k_processed": {
                "pca_logistic_annotation": (42,),
                "scgpt_frozen_probe": (42,),
                "scgpt_head": (42,),
                "scgpt_lora": (42,),
            },
            "pbmc68k_reduced": {
                "pca_logistic_annotation": (42,),
                "scgpt_frozen_probe": (42,),
                "scgpt_head": (42,),
                "scgpt_lora": (42,),
            },
            "openproblems_human_pancreas": {
                "pca_logistic_annotation": (42,),
                "scgpt_frozen_probe": (42,),
                "scgpt_head": (42,),
                "scgpt_lora": (42,),
            },
        },
    },
}

REQUIRED_TUTORIAL_NAMES = (
    "scanpy_pbmc_quickstart",
    "downstream_scanpy_after_scdlkit",
    "pbmc_model_comparison",
    "reconstruction_sanity_pbmc",
    "pbmc_classification",
    "custom_model_extension",
    "scgpt_pbmc_embeddings",
    "scgpt_cell_type_annotation",
    "scgpt_dataset_specific_annotation",
    "synthetic_smoke",
)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    label_key: str
    batch_key: str | None = None


DATASET_SPECS: dict[str, DatasetSpec] = {
    "pbmc3k_processed": DatasetSpec(name="pbmc3k_processed", label_key="louvain"),
    "pbmc68k_reduced": DatasetSpec(name="pbmc68k_reduced", label_key="bulk_labels"),
    "openproblems_human_pancreas": DatasetSpec(
        name="openproblems_human_pancreas",
        label_key="cell_type",
        batch_key="batch",
    ),
    "paul15": DatasetSpec(name="paul15", label_key="paul15_clusters"),
    "moignard15": DatasetSpec(name="moignard15", label_key="exp_groups"),
}

COMPACT_TRANSFORMER_MODEL_KWARGS: dict[str, Any] = {
    "patch_size": 48,
    "d_model": 64,
    "n_heads": 2,
    "n_layers": 1,
    "decoder_hidden_dims": (128,),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_DEFAULTS),
        default="full",
        help="Quality-suite profile to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to a dated folder under artifacts/quality/.",
    )
    parser.add_argument(
        "--tutorial-summary",
        type=Path,
        default=None,
        help="Optional tutorial validation summary JSON for RC readiness checks.",
    )
    parser.add_argument(
        "--foundation-annotation-metrics",
        type=Path,
        default=None,
        help=(
            "Optional CSV with precomputed scGPT annotation benchmark rows. "
            "Used in CI to reuse the dedicated annotation smoke results."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail with a non-zero exit code when benchmark gates fail.",
    )
    parser.add_argument(
        "--require-rc",
        action="store_true",
        help="Require full release-candidate readiness, including tutorial checks.",
    )
    return parser.parse_args()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _default_output_dir(profile: str) -> Path:
    return ROOT / "artifacts" / "quality" / f"{_utc_timestamp()}-{profile}"


def _load_dataset(name: str, *, profile: str = "full") -> tuple[AnnData, DatasetSpec]:
    import scanpy as sc

    if name == "pbmc3k_processed":
        data_path = ROOT / "examples" / "data" / "pbmc3k_processed.h5ad"
        adata = sc.read_h5ad(data_path) if data_path.exists() else sc.datasets.pbmc3k_processed()
    elif name == "pbmc68k_reduced":
        adata = sc.datasets.pbmc68k_reduced()
    elif name == "openproblems_human_pancreas":
        from scdlkit._datasets.openproblems import (
            load_openproblems_pancreas_annotation_dataset,
        )

        adata = load_openproblems_pancreas_annotation_dataset(profile=profile)
    elif name == "paul15":
        adata = sc.datasets.paul15()
    elif name == "moignard15":
        adata = sc.datasets.moignard15()
    else:
        msg = f"Unsupported dataset '{name}'."
        raise ValueError(msg)
    return adata, DATASET_SPECS[name]


def _to_dense(x_matrix: Any) -> np.ndarray:
    dense = x_matrix.toarray() if sparse.issparse(x_matrix) else np.asarray(x_matrix)
    return dense.astype("float32", copy=False)


def _process_peak_memory_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    raw_value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(raw_value) / (1024.0 * 1024.0)
    return float(raw_value) / 1024.0


def _save_scanpy_umap(
    adata: AnnData,
    latent: np.ndarray,
    label_key: str,
    path: Path,
    *,
    seed: int,
) -> None:
    import scanpy as sc
    from matplotlib import pyplot as plt

    plot_adata = adata.copy()
    plot_adata.obsm["X_quality_latent"] = latent
    n_obs = int(plot_adata.n_obs)
    if n_obs < 4:
        umap_fig, axis = plt.subplots(figsize=(5, 4))
        x_values = latent[:, 0]
        y_values = latent[:, 1] if latent.shape[1] > 1 else np.zeros(n_obs, dtype="float32")
        labels = plot_adata.obs[label_key].astype(str).to_numpy()
        for label in sorted(set(labels)):
            mask = labels == label
            axis.scatter(x_values[mask], y_values[mask], label=label, s=48, alpha=0.9)
        axis.set_title("Latent view (UMAP fallback)")
        axis.set_xlabel("latent_1")
        axis.set_ylabel("latent_2")
        axis.legend(loc="best", fontsize=8, frameon=False)
        umap_fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(umap_fig)
        return

    n_neighbors = max(2, min(5, n_obs - 1))
    sc.pp.neighbors(plot_adata, use_rep="X_quality_latent", n_neighbors=n_neighbors)
    try:
        sc.tl.umap(plot_adata, random_state=seed, init_pos="random")
        umap_fig = sc.pl.umap(plot_adata, color=label_key, return_fig=True, frameon=False)
    except (TypeError, ValueError):
        umap_fig, axis = plt.subplots(figsize=(5, 4))
        x_values = latent[:, 0]
        y_values = latent[:, 1] if latent.shape[1] > 1 else np.zeros(n_obs, dtype="float32")
        labels = plot_adata.obs[label_key].astype(str).to_numpy()
        for label in sorted(set(labels)):
            mask = labels == label
            axis.scatter(x_values[mask], y_values[mask], label=label, s=48, alpha=0.9)
        axis.set_title("Latent view (UMAP fallback)")
        axis.set_xlabel("latent_1")
        axis.set_ylabel("latent_2")
        axis.legend(loc="best", fontsize=8, frameon=False)
    umap_fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(umap_fig)


def _encode_obs(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    encoded = pd.Categorical(values.astype(str))
    return encoded.codes.astype(int), list(encoded.categories)


def _linear_probe_metrics(
    latent: np.ndarray,
    labels: np.ndarray | None,
    *,
    seed: int,
) -> dict[str, float | list[list[int]]]:
    from scdlkit.evaluation.metrics import classification_metrics

    if labels is None or len(np.unique(labels)) < 2:
        return {}
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
        "probe_confusion_matrix": metrics["confusion_matrix"],
    }


def _write_report(output_dir: Path, title: str, bullets: dict[str, Any]) -> None:
    report_lines = [f"# {title}", ""]
    report_lines.extend(f"- `{key}`: `{value}`" for key, value in bullets.items())
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: value for key, value in metrics.items() if isinstance(value, (int, float))}


def _iter_profile_runs(profile: str) -> list[tuple[str, str, str, int]]:
    runs: list[tuple[str, str, str, int]] = []
    profile_config = PROFILE_DEFAULTS[profile]
    for task, datasets in profile_config.items():
        for dataset_name, model_seeds in datasets.items():
            for model_name, seeds in model_seeds.items():
                runs.extend((dataset_name, task, model_name, seed) for seed in seeds)
    return runs


def _runner_kwargs(model_name: str, profile: str) -> dict[str, Any]:
    if model_name == "autoencoder":
        return {"epochs": 10 if profile == "ci" else 25, "batch_size": 128}
    if model_name == "vae":
        return {
            "epochs": 20 if profile == "ci" else 50,
            "batch_size": 128,
            "model_kwargs": {"kl_weight": 1e-3},
        }
    if model_name == "transformer_ae":
        return {
            "epochs": 10 if profile == "ci" else 25,
            "batch_size": 128,
            "model_kwargs": COMPACT_TRANSFORMER_MODEL_KWARGS,
        }
    if model_name == "mlp_classifier":
        return {"epochs": 15 if profile == "ci" else 30, "batch_size": 128}
    msg = f"Unsupported runner model '{model_name}'."
    raise ValueError(msg)


def _subset_adata_for_foundation(
    adata: AnnData,
    *,
    label_key: str,
    seed: int,
    max_cells: int = 128,
) -> AnnData:
    if adata.n_obs <= max_cells:
        return adata.copy()
    indices = np.arange(adata.n_obs)
    labels, _ = _encode_obs(adata.obs[label_key])
    keep_indices, _ = train_test_split(
        indices,
        train_size=max_cells,
        random_state=seed,
        stratify=labels,
    )
    return adata[np.sort(keep_indices)].copy()


def _subset_foundation_genes(
    adata: AnnData,
    *,
    max_genes: int | None,
) -> AnnData:
    if max_genes is None or adata.n_vars <= max_genes:
        return adata.copy()
    source = adata.raw.to_adata() if adata.raw is not None else adata.copy()
    dense = _to_dense(source.X)
    variances = np.var(dense, axis=0)
    keep_indices = np.argsort(variances)[-max_genes:]
    subset = source[:, np.sort(keep_indices)].copy()
    subset.raw = subset.copy()
    return subset


def _save_loss_curve(runner: Any, output_dir: Path) -> None:
    from matplotlib import pyplot as plt

    loss_fig, _ = runner.plot_losses()
    loss_fig.savefig(output_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close(loss_fig)


def _save_confusion_plot(
    confusion: list[list[int]],
    class_names: list[str],
    output_dir: Path,
) -> None:
    from matplotlib import pyplot as plt

    from scdlkit.visualization.classification import plot_confusion_matrix

    confusion_fig, _ = plot_confusion_matrix(confusion, class_names=class_names)
    confusion_fig.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(confusion_fig)


def _count_trainable_parameters(model: Any) -> int:
    return int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )


def _batch_metrics_frame(
    *,
    obs: pd.DataFrame,
    batch_key: str | None,
    y_true: np.ndarray,
    logits: np.ndarray,
) -> pd.DataFrame:
    from scdlkit.evaluation.metrics import classification_metrics

    columns = ["batch", "n_cells", "accuracy", "macro_f1", "balanced_accuracy", "auroc_ovr"]
    if batch_key is None or batch_key not in obs:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    batches = obs[batch_key].astype(str).to_numpy()
    for batch_value in sorted(pd.unique(batches)):
        mask = batches == batch_value
        if int(mask.sum()) == 0:
            continue
        metrics = classification_metrics(y_true[mask], logits[mask])
        rows.append(
            {
                "batch": batch_value,
                "n_cells": int(mask.sum()),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "auroc_ovr": float(metrics["auroc_ovr"])
                if isinstance(metrics.get("auroc_ovr"), (int, float))
                else float("nan"),
            }
        )
    return pd.DataFrame.from_records(rows, columns=columns)


def _batch_metric_summary(batch_metrics: pd.DataFrame) -> dict[str, float]:
    if batch_metrics.empty:
        return {}
    return {
        "batch_accuracy_mean": float(batch_metrics["accuracy"].mean()),
        "batch_accuracy_min": float(batch_metrics["accuracy"].min()),
        "batch_macro_f1_mean": float(batch_metrics["macro_f1"].mean()),
        "batch_macro_f1_min": float(batch_metrics["macro_f1"].min()),
        "batch_balanced_accuracy_mean": float(batch_metrics["balanced_accuracy"].mean()),
        "batch_balanced_accuracy_min": float(batch_metrics["balanced_accuracy"].min()),
    }


def _save_trainable_annotation_checkpoint(
    *,
    output_dir: Path,
    model: Any,
    label_key: str,
    label_categories: list[str],
    best_strategy: str,
    batch_size: int,
    random_state: int,
    trainable_parameters: int,
    metrics: dict[str, Any],
    strategy_config: dict[str, Any] | None,
) -> Path:
    import torch

    checkpoint_dir = output_dir / "best_model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    strategy_metrics = {
        "strategy": best_strategy,
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "balanced_accuracy": float(metrics.get("balanced_accuracy", float("nan"))),
        "runtime_sec": float(metrics["runtime_sec"]),
        "trainable_parameters": int(trainable_parameters),
    }
    manifest = {
        "checkpoint_id": "whole-human",
        "label_key": label_key,
        "label_categories": list(label_categories),
        "best_strategy": best_strategy,
        "strategies": [best_strategy],
        "batch_size": int(batch_size),
        "val_size": 0.15,
        "test_size": 0.15,
        "random_state": int(random_state),
        "classifier_dropout": 0.1,
        "strategy_configs": (
            {best_strategy: strategy_config} if strategy_config is not None else None
        ),
        "lora_config": strategy_config if best_strategy == "lora" else None,
        "metrics": strategy_metrics,
        "strategy_metrics": [strategy_metrics],
        "output_dir": str(output_dir),
    }
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    torch.save(
        {"kind": "torch_state_dict", "state_dict": model.state_dict()},
        checkpoint_dir / "model_state.pt",
    )
    return checkpoint_dir


def _prepare_annotation_benchmark_adata(
    dataset_name: str,
    adata: AnnData,
    *,
    label_key: str,
    seed: int,
    profile: str,
) -> AnnData:
    if dataset_name == "openproblems_human_pancreas":
        return adata.copy()
    profile_config = _foundation_annotation_profile(profile)
    annotation_adata = _subset_adata_for_foundation(
        adata,
        label_key=label_key,
        seed=seed,
        max_cells=profile_config["max_cells"],
    )
    return _subset_foundation_genes(
        annotation_adata,
        max_genes=profile_config["max_genes"],
    )


def _expand_probabilities(
    probabilities: np.ndarray,
    classes: np.ndarray | list[int],
    *,
    num_classes: int,
) -> np.ndarray:
    expanded = np.zeros((probabilities.shape[0], num_classes), dtype=np.float32)
    class_indices = np.asarray(classes, dtype=int)
    expanded[:, class_indices] = np.asarray(probabilities, dtype=np.float32)
    return expanded


def _subset_adata_from_dataset(
    adata: AnnData,
    dataset: Any,
) -> AnnData:
    from torch.utils.data import Subset

    if isinstance(dataset, Subset):
        return adata[np.asarray(dataset.indices, dtype=int)].copy()
    return adata.copy()


def _first_nonempty_split(*splits: Any) -> Any:
    for split in splits:
        if split is not None and len(split) > 0:
            return split
    return None


def _classification_split_seed(profile: str) -> int:
    return 42 if profile == "ci" else 52


def _foundation_annotation_profile(profile: str) -> dict[str, int]:
    if profile == "ci":
        return {
            "max_cells": 32,
            "max_genes": 32,
            "min_gene_overlap": 16,
            "head_epochs": 1,
            "full_finetune_epochs": 1,
            "lora_epochs": 1,
            "adapter_epochs": 1,
            "prefix_tuning_epochs": 1,
            "ia3_epochs": 1,
        }
    return {
        "max_cells": 64,
        "max_genes": 128,
        "min_gene_overlap": 64,
        "head_epochs": 15,
        "full_finetune_epochs": 10,
        "lora_epochs": 15,
        "adapter_epochs": 15,
        "prefix_tuning_epochs": 15,
        "ia3_epochs": 15,
    }


def run_representation_runner(
    *,
    dataset_name: str,
    adata: AnnData,
    label_key: str,
    batch_key: str | None,
    model_name: str,
    seed: int,
    profile: str,
    output_root: Path,
) -> dict[str, Any]:
    from scdlkit import TaskRunner

    output_dir = output_root / dataset_name / "representation" / model_name / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = _runner_kwargs(model_name, profile)
    runner = TaskRunner(
        model=model_name,
        task="representation",
        label_key=label_key,
        batch_key=batch_key,
        device="auto",
        seed=seed,
        random_state=seed,
        output_dir=str(output_dir),
        **settings,
    )
    started_at = perf_counter()
    runner.fit(adata)
    metrics = runner.evaluate()
    runtime_sec = perf_counter() - started_at
    runner.save_report(output_dir / "report.md")
    _save_loss_curve(runner, output_dir)
    latent = runner.encode(adata)
    _save_scanpy_umap(adata, latent, label_key, output_dir / "latent_umap.png", seed=seed)
    scalar_metrics = {
        key: value for key, value in metrics.items() if isinstance(value, (int, float))
    }
    return {
        "dataset": dataset_name,
        "task": "representation",
        "model": model_name,
        "seed": seed,
        "profile": profile,
        "runtime_sec": runtime_sec,
        "peak_memory_mb": _process_peak_memory_mb(),
        "artifact_dir": str(output_dir),
        **scalar_metrics,
    }


def run_pca_baseline(
    *,
    dataset_name: str,
    adata: AnnData,
    label_key: str,
    batch_key: str | None,
    seed: int,
    output_root: Path,
    n_components: int = 32,
) -> dict[str, Any]:
    from scdlkit.evaluation.metrics import reconstruction_metrics, representation_metrics

    output_dir = output_root / dataset_name / "representation" / "pca" / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    x_matrix = _to_dense(adata.X)
    labels, _ = _encode_obs(adata.obs[label_key])
    batches = None
    if batch_key is not None and batch_key in adata.obs:
        batches, _ = _encode_obs(adata.obs[batch_key])
    n_pcs = min(n_components, x_matrix.shape[0] - 1, x_matrix.shape[1])
    started_at = perf_counter()
    pca = PCA(n_components=n_pcs, random_state=seed)
    latent = pca.fit_transform(x_matrix)
    reconstruction = pca.inverse_transform(latent)
    runtime_sec = perf_counter() - started_at
    metrics = reconstruction_metrics(x_matrix, reconstruction)
    metrics.update(representation_metrics(latent, labels, batches))
    metrics.update(_linear_probe_metrics(latent, labels, seed=seed))
    report_metrics = {
        "Dataset": dataset_name,
        "Components": n_pcs,
        **{
            key: f"{value:.4f}" if isinstance(value, float) else value
            for key, value in metrics.items()
            if key != "probe_confusion_matrix"
        },
    }
    _write_report(output_dir, "PCA baseline report", report_metrics)
    pd.DataFrame([metrics]).to_csv(output_dir / "report.csv", index=False)
    _save_scanpy_umap(adata, latent, label_key, output_dir / "latent_umap.png", seed=seed)
    return {
        "dataset": dataset_name,
        "task": "representation",
        "model": "pca",
        "seed": seed,
        "profile": "baseline",
        "runtime_sec": runtime_sec,
        "peak_memory_mb": _process_peak_memory_mb(),
        "artifact_dir": str(output_dir),
        **_scalar_metrics(metrics),
    }


def run_foundation_pca_reference(
    *,
    dataset_name: str,
    adata: AnnData,
    label_key: str,
    batch_key: str | None,
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    from scdlkit.evaluation.metrics import representation_metrics

    output_dir = output_root / dataset_name / "foundation" / "pca_foundation" / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    subset = _subset_adata_for_foundation(adata, label_key=label_key, seed=seed)
    x_matrix = _to_dense(subset.X)
    labels, _ = _encode_obs(subset.obs[label_key])
    batches = None
    if batch_key is not None and batch_key in subset.obs:
        batches, _ = _encode_obs(subset.obs[batch_key])
    n_components = min(32, x_matrix.shape[0] - 1, x_matrix.shape[1])
    started_at = perf_counter()
    pca = PCA(n_components=n_components, random_state=seed)
    latent = pca.fit_transform(x_matrix)
    runtime_sec = perf_counter() - started_at
    metrics = representation_metrics(latent, labels, batches)
    metrics.update(_linear_probe_metrics(latent, labels, seed=seed))
    report_metrics = {
        "Dataset": dataset_name,
        "Reference model": "PCA",
        "Cells": subset.n_obs,
        **{
            key: f"{value:.4f}" if isinstance(value, float) else value
            for key, value in metrics.items()
            if key != "probe_confusion_matrix"
        },
    }
    _write_report(output_dir, "Foundation reference PCA report", report_metrics)
    pd.DataFrame([metrics]).to_csv(output_dir / "report.csv", index=False)
    _save_scanpy_umap(subset, latent, label_key, output_dir / "latent_umap.png", seed=seed)
    return {
        "dataset": dataset_name,
        "task": "foundation",
        "model": "pca_foundation",
        "seed": seed,
        "profile": "foundation",
        "runtime_sec": runtime_sec,
        "peak_memory_mb": _process_peak_memory_mb(),
        "artifact_dir": str(output_dir),
        **_scalar_metrics(metrics),
    }


def run_scgpt_embedding_baseline(
    *,
    dataset_name: str,
    adata: AnnData,
    label_key: str,
    batch_key: str | None,
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    from scdlkit.evaluation.metrics import representation_metrics
    from scdlkit.foundation import load_scgpt_model, prepare_scgpt_data
    from scdlkit.training import Trainer

    output_dir = output_root / dataset_name / "foundation" / "scgpt_whole_human" / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    subset = _subset_adata_for_foundation(adata, label_key=label_key, seed=seed)
    prepared = prepare_scgpt_data(
        subset,
        checkpoint="whole-human",
        label_key=label_key,
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
    latent = np.asarray(predictions["latent"], dtype="float32")
    labels = predictions.get("y")
    batches = None
    if batch_key is not None and batch_key in subset.obs:
        batches, _ = _encode_obs(subset.obs[batch_key])
    metrics = representation_metrics(latent, labels, batches)
    metrics.update(_linear_probe_metrics(latent, labels, seed=seed))
    report_metrics = {
        "Dataset": dataset_name,
        "Checkpoint": "whole-human",
        "Matched genes": prepared.num_genes_matched,
        "Cells": subset.n_obs,
        **{
            key: f"{value:.4f}" if isinstance(value, float) else value
            for key, value in metrics.items()
            if key != "probe_confusion_matrix"
        },
    }
    _write_report(output_dir, "scGPT frozen embedding report", report_metrics)
    pd.DataFrame([metrics]).to_csv(output_dir / "report.csv", index=False)
    _save_scanpy_umap(subset, latent, label_key, output_dir / "latent_umap.png", seed=seed)
    return {
        "dataset": dataset_name,
        "task": "foundation",
        "model": "scgpt_whole_human",
        "seed": seed,
        "profile": "foundation",
        "runtime_sec": runtime_sec,
        "peak_memory_mb": _process_peak_memory_mb(),
        "artifact_dir": str(output_dir),
        "num_genes_matched": prepared.num_genes_matched,
        **_scalar_metrics(metrics),
    }


def run_classification_runner(
    *,
    dataset_name: str,
    adata: AnnData,
    label_key: str,
    seed: int,
    profile: str,
    output_root: Path,
) -> dict[str, Any]:
    from scdlkit import TaskRunner

    output_dir = output_root / dataset_name / "classification" / "mlp_classifier" / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = _runner_kwargs("mlp_classifier", profile)
    runner = TaskRunner(
        model="mlp_classifier",
        task="classification",
        label_key=label_key,
        device="auto",
        seed=seed,
        random_state=seed,
        output_dir=str(output_dir),
        **settings,
    )
    started_at = perf_counter()
    runner.fit(adata)
    metrics = runner.evaluate()
    runtime_sec = perf_counter() - started_at
    runner.save_report(output_dir / "report.md")
    _save_loss_curve(runner, output_dir)
    confusion = metrics.get("confusion_matrix")
    if confusion is None:
        msg = "Classification runner did not return a confusion matrix."
        raise RuntimeError(msg)
    _, class_names = _encode_obs(adata.obs[label_key])
    _save_confusion_plot(confusion, class_names, output_dir)
    scalar_metrics = {
        key: value for key, value in metrics.items() if isinstance(value, (int, float))
    }
    return {
        "dataset": dataset_name,
        "task": "classification",
        "model": "mlp_classifier",
        "seed": seed,
        "profile": profile,
        "runtime_sec": runtime_sec,
        "peak_memory_mb": _process_peak_memory_mb(),
        "artifact_dir": str(output_dir),
        **scalar_metrics,
    }


def run_logistic_regression_pca(
    *,
    dataset_name: str,
    adata: AnnData,
    label_key: str,
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    from scdlkit import prepare_data
    from scdlkit.evaluation.metrics import classification_metrics

    output_dir = (
        output_root / dataset_name / "classification" / "logistic_regression_pca" / f"seed_{seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_data(adata, label_key=label_key, random_state=seed)
    train_split = prepared.train
    test_split = _first_nonempty_split(prepared.test, prepared.val, prepared.train)
    if test_split is None or train_split.labels is None or test_split.labels is None:
        msg = "Classification benchmark requires train/test labels."
        raise RuntimeError(msg)
    x_train = _to_dense(train_split.X)
    x_test = _to_dense(test_split.X)
    n_components = min(32, x_train.shape[0] - 1, x_train.shape[1])
    started_at = perf_counter()
    pca = PCA(n_components=n_components, random_state=seed)
    train_latent = pca.fit_transform(x_train)
    test_latent = pca.transform(x_test)
    classifier = LogisticRegression(max_iter=1000, random_state=seed)
    classifier.fit(train_latent, train_split.labels)
    logits = _expand_probabilities(
        classifier.predict_proba(test_latent),
        classifier.classes_,
        num_classes=len(prepared.label_encoder or []),
    )
    runtime_sec = perf_counter() - started_at
    metrics = classification_metrics(test_split.labels, logits)
    class_names = list(prepared.label_encoder) if prepared.label_encoder is not None else []
    confusion = metrics.get("confusion_matrix")
    if isinstance(confusion, list):
        _save_confusion_plot(confusion, class_names, output_dir)
    report_lines = [
        "# Logistic regression on PCA baseline",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- PCA components: `{n_components}`",
        "",
    ]
    report_lines.extend(
        f"- `{key}`: `{value:.4f}`" for key, value in metrics.items() if isinstance(value, float)
    )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(output_dir / "report.csv", index=False)
    scalar_metrics = {
        key: value for key, value in metrics.items() if isinstance(value, (int, float))
    }
    return {
        "dataset": dataset_name,
        "task": "classification",
        "model": "logistic_regression_pca",
        "seed": seed,
        "profile": "baseline",
        "runtime_sec": runtime_sec,
        "peak_memory_mb": _process_peak_memory_mb(),
        "artifact_dir": str(output_dir),
        **scalar_metrics,
    }


def run_foundation_annotation_pca_logistic(
    *,
    dataset_name: str,
    adata: AnnData,
    label_key: str,
    batch_key: str | None,
    seed: int,
    output_root: Path,
    profile: str,
) -> dict[str, Any]:
    from scdlkit import prepare_data
    from scdlkit.evaluation.metrics import classification_metrics, representation_metrics

    output_dir = (
        output_root
        / dataset_name
        / "foundation_annotation"
        / "pca_logistic_annotation"
        / f"seed_{seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_adata = _prepare_annotation_benchmark_adata(
        dataset_name,
        adata,
        label_key=label_key,
        seed=seed,
        profile=profile,
    )
    prepared = prepare_data(annotation_adata, label_key=label_key, random_state=seed)
    train_split = prepared.train
    test_split = _first_nonempty_split(prepared.test, prepared.val, prepared.train)
    if test_split is None or train_split.labels is None or test_split.labels is None:
        msg = "Foundation annotation baseline requires train/test labels."
        raise RuntimeError(msg)

    x_train = _to_dense(train_split.X)
    x_test = _to_dense(test_split.X)
    n_components = min(32, x_train.shape[0] - 1, x_train.shape[1])
    started_at = perf_counter()
    pca = PCA(n_components=n_components, random_state=seed)
    train_latent = pca.fit_transform(x_train)
    test_latent = pca.transform(x_test)
    classifier = LogisticRegression(max_iter=1000, random_state=seed)
    classifier.fit(train_latent, train_split.labels)
    logits = _expand_probabilities(
        classifier.predict_proba(test_latent),
        classifier.classes_,
        num_classes=len(prepared.label_encoder or []),
    )
    runtime_sec = perf_counter() - started_at

    metrics = classification_metrics(test_split.labels, logits)
    metrics.update(representation_metrics(test_latent, test_split.labels, None))
    confusion = metrics.get("confusion_matrix")
    class_names = list(prepared.label_encoder) if prepared.label_encoder is not None else []
    if isinstance(confusion, list):
        _save_confusion_plot(confusion, class_names, output_dir)

    obs_names = test_split.obs_names
    test_adata = annotation_adata[obs_names].copy() if obs_names else annotation_adata.copy()
    batch_metrics = _batch_metrics_frame(
        obs=test_adata.obs,
        batch_key=batch_key,
        y_true=np.asarray(test_split.labels, dtype=int),
        logits=np.asarray(logits, dtype=np.float32),
    )
    batch_metrics.to_csv(output_dir / "batch_metrics.csv", index=False)
    _save_scanpy_umap(test_adata, test_latent, label_key, output_dir / "latent_umap.png", seed=seed)

    report_metrics = {
        "Dataset": dataset_name,
        "Strategy": "PCA + logistic regression",
        "Trainable parameters": 0,
        **{
            key: f"{value:.4f}" if isinstance(value, float) else value
            for key, value in metrics.items()
            if key != "confusion_matrix"
        },
    }
    _write_report(output_dir, "Foundation annotation PCA baseline", report_metrics)
    pd.DataFrame(
        [
            {
                **metrics,
                **_batch_metric_summary(batch_metrics),
                "trainable_parameters": 0,
                "batch_metrics_artifact": str(output_dir / "batch_metrics.csv"),
                "confusion_matrix_artifact": str(output_dir / "confusion_matrix.png"),
                "latent_umap_artifact": str(output_dir / "latent_umap.png"),
            }
        ]
    ).to_csv(output_dir / "report.csv", index=False)
    return {
        "dataset": dataset_name,
        "task": "foundation_annotation",
        "model": "pca_logistic_annotation",
        "seed": seed,
        "profile": "foundation_annotation",
        "runtime_sec": runtime_sec,
        "peak_memory_mb": _process_peak_memory_mb(),
        "artifact_dir": str(output_dir),
        "trainable_parameters": 0,
        "batch_metrics_artifact": str(output_dir / "batch_metrics.csv"),
        "confusion_matrix_artifact": str(output_dir / "confusion_matrix.png"),
        "latent_umap_artifact": str(output_dir / "latent_umap.png"),
        **_scalar_metrics(metrics),
        **_batch_metric_summary(batch_metrics),
    }


def run_scgpt_annotation_strategy(
    *,
    dataset_name: str,
    adata: AnnData,
    label_key: str,
    batch_key: str | None,
    seed: int,
    output_root: Path,
    model_name: str,
    profile: str,
) -> dict[str, Any]:
    from scdlkit.evaluation.metrics import classification_metrics, representation_metrics
    from scdlkit.foundation import (
        AdapterConfig,
        IA3Config,
        LoRAConfig,
        PrefixTuningConfig,
        count_trainable_parameters,
        load_scgpt_annotation_model,
        load_scgpt_model,
        prepare_scgpt_data,
        split_scgpt_data,
    )
    from scdlkit.training import Trainer

    output_dir = output_root / dataset_name / "foundation_annotation" / model_name / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_config = _foundation_annotation_profile(profile)
    annotation_adata = _prepare_annotation_benchmark_adata(
        dataset_name,
        adata,
        label_key=label_key,
        seed=seed,
        profile=profile,
    )
    prepared = prepare_scgpt_data(
        annotation_adata,
        checkpoint="whole-human",
        label_key=label_key,
        batch_size=64,
        use_raw=True,
        min_gene_overlap=profile_config["min_gene_overlap"],
    )
    split = split_scgpt_data(prepared, val_size=0.15, test_size=0.15, random_state=seed)
    test_dataset = split.test or split.val or split.train
    test_adata = _subset_adata_from_dataset(annotation_adata, test_dataset)
    label_categories = list(prepared.label_categories or [])

    strategy_specs: dict[str, dict[str, Any]] = {
        "scgpt_head": {
            "tuning_strategy": "head",
            "epochs": profile_config["head_epochs"],
            "lr": 5e-3,
            "strategy_config": None,
        },
        "scgpt_full_finetune": {
            "tuning_strategy": "full_finetune",
            "epochs": profile_config["full_finetune_epochs"],
            "lr": 5e-4,
            "strategy_config": None,
        },
        "scgpt_lora": {
            "tuning_strategy": "lora",
            "epochs": profile_config["lora_epochs"],
            "lr": 2e-3,
            "strategy_config": LoRAConfig(rank=4, alpha=8.0, dropout=0.05),
        },
        "scgpt_adapter": {
            "tuning_strategy": "adapter",
            "epochs": profile_config["adapter_epochs"],
            "lr": 2e-3,
            "strategy_config": AdapterConfig(bottleneck_dim=64, dropout=0.05),
        },
        "scgpt_prefix_tuning": {
            "tuning_strategy": "prefix_tuning",
            "epochs": profile_config["prefix_tuning_epochs"],
            "lr": 2e-3,
            "strategy_config": PrefixTuningConfig(prefix_length=20, dropout=0.05),
        },
        "scgpt_ia3": {
            "tuning_strategy": "ia3",
            "epochs": profile_config["ia3_epochs"],
            "lr": 2e-3,
            "strategy_config": IA3Config(init_scale=1.0),
        },
    }

    started_at = perf_counter()
    if model_name == "scgpt_frozen_probe":
        model = load_scgpt_model("whole-human", device="auto")
        strategy_config_payload = None
        trainer = Trainer(
            model=model,
            task="representation",
            batch_size=prepared.batch_size,
            device="auto",
            epochs=1,
        )
        train_predictions = trainer.predict_dataset(split.train)
        test_predictions = trainer.predict_dataset(test_dataset)
        classifier = LogisticRegression(max_iter=1000, random_state=seed)
        classifier.fit(train_predictions["latent"], train_predictions["y"])
        logits = _expand_probabilities(
            classifier.predict_proba(test_predictions["latent"]),
            classifier.classes_,
            num_classes=len(label_categories),
        )
        predictions = {
            "logits": logits,
            "latent": test_predictions["latent"],
            "y": test_predictions["y"],
        }
        trainable_parameters = 0
    else:
        if model_name not in strategy_specs:
            msg = f"Unsupported scGPT annotation benchmark model '{model_name}'."
            raise ValueError(msg)
        strategy_spec = strategy_specs[model_name]
        tuning_strategy = str(strategy_spec["tuning_strategy"])
        strategy_config = strategy_spec["strategy_config"]
        strategy_config_payload = (
            strategy_config.to_payload() if strategy_config is not None else None
        )
        model = load_scgpt_annotation_model(
            num_classes=len(label_categories),
            checkpoint="whole-human",
            tuning_strategy=tuning_strategy,  # type: ignore[arg-type]
            label_categories=prepared.label_categories,
            device="auto",
            strategy_config=strategy_config,
        )
        trainer = Trainer(
            model=model,
            task="classification",
            batch_size=prepared.batch_size,
            epochs=int(strategy_spec["epochs"]),
            lr=float(strategy_spec["lr"]),
            device="auto",
            early_stopping_patience=5,
            lr_schedule_gamma=0.9,
            seed=seed,
        )
        trainer.fit(split.train, split.val)
        predictions = trainer.predict_dataset(test_dataset)
        trainable_parameters = count_trainable_parameters(model)
    runtime_sec = perf_counter() - started_at

    metrics = classification_metrics(predictions["y"], predictions["logits"])
    metrics.update(representation_metrics(predictions["latent"], predictions["y"], None))
    confusion = metrics.get("confusion_matrix")
    if isinstance(confusion, list):
        _save_confusion_plot(confusion, label_categories, output_dir)
    batch_metrics = _batch_metrics_frame(
        obs=test_adata.obs,
        batch_key=batch_key,
        y_true=np.asarray(predictions["y"], dtype=int),
        logits=np.asarray(predictions["logits"], dtype=np.float32),
    )
    batch_metrics.to_csv(output_dir / "batch_metrics.csv", index=False)
    _save_scanpy_umap(
        test_adata,
        np.asarray(predictions["latent"], dtype="float32"),
        label_key,
        output_dir / "latent_umap.png",
        seed=seed,
    )

    report_metrics = {
        "Dataset": dataset_name,
        "Strategy": model_name,
        "Trainable parameters": trainable_parameters,
        **{
            key: f"{value:.4f}" if isinstance(value, float) else value
            for key, value in metrics.items()
            if key != "confusion_matrix"
        },
    }
    _write_report(output_dir, "scGPT annotation strategy report", report_metrics)
    pd.DataFrame(
        [
            {
                **metrics,
                **_batch_metric_summary(batch_metrics),
                "trainable_parameters": trainable_parameters,
                "batch_metrics_artifact": str(output_dir / "batch_metrics.csv"),
                "confusion_matrix_artifact": str(output_dir / "confusion_matrix.png"),
                "latent_umap_artifact": str(output_dir / "latent_umap.png"),
            }
        ]
    ).to_csv(output_dir / "report.csv", index=False)

    best_model_artifact = None
    if model_name != "scgpt_frozen_probe":
        best_model_artifact = str(
            _save_trainable_annotation_checkpoint(
                output_dir=output_dir,
                model=model,
                label_key=label_key,
                label_categories=label_categories,
                best_strategy=tuning_strategy,
                batch_size=prepared.batch_size,
                random_state=seed,
                trainable_parameters=trainable_parameters,
                metrics={
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "runtime_sec": runtime_sec,
                },
                strategy_config=strategy_config_payload,
            )
        )

    checkpoint_size_bytes = None
    if best_model_artifact is not None:
        checkpoint_path = Path(best_model_artifact) / "model_state.pt"
        if checkpoint_path.exists():
            checkpoint_size_bytes = int(checkpoint_path.stat().st_size)

    return {
        "dataset": dataset_name,
        "task": "foundation_annotation",
        "model": model_name,
        "seed": seed,
        "profile": "foundation_annotation",
        "runtime_sec": runtime_sec,
        "peak_memory_mb": _process_peak_memory_mb(),
        "artifact_dir": str(output_dir),
        "trainable_parameters": trainable_parameters,
        "best_model_artifact": best_model_artifact,
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "batch_metrics_artifact": str(output_dir / "batch_metrics.csv"),
        "confusion_matrix_artifact": str(output_dir / "confusion_matrix.png"),
        "latent_umap_artifact": str(output_dir / "latent_umap.png"),
        **_scalar_metrics(metrics),
        **_batch_metric_summary(batch_metrics),
    }


def aggregate_metrics(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = metrics_frame.select_dtypes(include=("number",)).columns.tolist()
    value_columns = [column for column in numeric_columns if column != "seed"]
    if not value_columns:
        return pd.DataFrame()
    aggregated = (
        metrics_frame.groupby(["dataset", "task", "model"])[value_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregated.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in aggregated.columns.to_flat_index()
    ]
    return aggregated


def _required_benchmark_artifacts(task: str, model: str) -> tuple[str, ...]:
    if task == "foundation":
        return ("report.md", "report.csv", "latent_umap.png")
    if task == "foundation_annotation":
        return (
            "report.md",
            "report.csv",
            "batch_metrics.csv",
            "confusion_matrix.png",
            "latent_umap.png",
        )
    if task == "representation" and model in {"pca", "scgpt_whole_human"}:
        return ("report.md", "report.csv", "latent_umap.png")
    if task == "representation":
        return ("report.md", "report.csv", "loss_curve.png", "latent_umap.png")
    if model == "mlp_classifier":
        return ("report.md", "report.csv", "loss_curve.png", "confusion_matrix.png")
    return ("report.md", "report.csv", "confusion_matrix.png")


def _find_missing_runs(metrics_frame: pd.DataFrame, profile: str) -> list[str]:
    present = {
        (str(row.dataset), str(row.task), str(row.model), int(row.seed))
        for row in metrics_frame.itertuples()
    }
    missing: list[str] = []
    for dataset_name, task, model_name, seed in _iter_profile_runs(profile):
        run_spec = (dataset_name, task, model_name, seed)
        if run_spec not in present:
            missing.append(f"{dataset_name}/{task}/{model_name}/seed_{seed}")
    return missing


def _collect_benchmark_artifact_checks(metrics_frame: pd.DataFrame) -> dict[str, Any]:
    missing_files: list[str] = []
    checked_files = 0
    for row in metrics_frame.itertuples():
        artifact_dir = Path(str(row.artifact_dir))
        for filename in _required_benchmark_artifacts(str(row.task), str(row.model)):
            checked_files += 1
            if not (artifact_dir / filename).exists():
                missing_files.append(str(artifact_dir / filename))
    return {
        "passed": not missing_files,
        "checked_files": checked_files,
        "missing_files": missing_files,
    }


def _normalize_tutorial_summary(tutorial_summary: dict[str, Any] | None) -> dict[str, Any]:
    if tutorial_summary is None:
        return {
            "validated": False,
            "passed": False,
            "issues": ["Tutorial summary not provided."],
            "runtime": {
                "total_sec": None,
                "budget_sec": None,
                "passed": False,
                "notebook_count": 0,
            },
            "artifact_checks": {
                "passed": False,
                "missing_files": [],
            },
            "missing_notebooks": list(REQUIRED_TUTORIAL_NAMES),
        }
    runtime = tutorial_summary.get("runtime", {})
    artifact_checks = tutorial_summary.get("artifact_checks", {})
    notebook_names = {
        str(notebook.get("name"))
        for notebook in tutorial_summary.get("notebooks", [])
        if notebook.get("name")
    }
    missing_notebooks = [
        notebook_name
        for notebook_name in REQUIRED_TUTORIAL_NAMES
        if notebook_name not in notebook_names
    ]
    issues = list(tutorial_summary.get("issues", []))
    if missing_notebooks:
        issues.extend(
            f"Missing tutorial summary entry for notebook '{notebook_name}'."
            for notebook_name in missing_notebooks
        )
    passed = bool(tutorial_summary.get("passed", False)) and not missing_notebooks
    return {
        "validated": True,
        "passed": passed,
        "issues": issues,
        "runtime": {
            "total_sec": runtime.get("total_sec"),
            "budget_sec": runtime.get("budget_sec"),
            "passed": bool(runtime.get("passed", False)),
            "notebook_count": int(runtime.get("notebook_count", len(notebook_names))),
        },
        "artifact_checks": {
            "passed": bool(artifact_checks.get("passed", False)),
            "missing_files": list(artifact_checks.get("missing_files", [])),
        },
        "missing_notebooks": missing_notebooks,
    }


def evaluate_quality_gates(metrics_frame: pd.DataFrame, *, profile: str) -> list[str]:
    issues: list[str] = []
    pbmc_vae = metrics_frame[
        (metrics_frame["dataset"] == "pbmc3k_processed")
        & (metrics_frame["task"] == "representation")
        & (metrics_frame["model"] == "vae")
    ]
    if len(pbmc_vae) < 3:
        issues.append("PBMC VAE benchmark did not run across three seeds.")
    else:
        silhouette_mean = float(pbmc_vae["silhouette"].mean())
        knn_mean = float(pbmc_vae["knn_label_consistency"].mean())
        pearson_mean = float(pbmc_vae["pearson"].mean())
        silhouette_std = float(pbmc_vae["silhouette"].std(ddof=0))
        if silhouette_mean < QUALITY_GATES["pbmc_vae_silhouette_min"]:
            issues.append(
                "PBMC VAE silhouette fell below "
                f"{QUALITY_GATES['pbmc_vae_silhouette_min']:.2f} "
                f"(observed {silhouette_mean:.3f})."
            )
        if knn_mean < QUALITY_GATES["pbmc_vae_knn_label_consistency_min"]:
            issues.append(
                "PBMC VAE kNN label consistency fell below "
                f"{QUALITY_GATES['pbmc_vae_knn_label_consistency_min']:.2f} "
                f"(observed {knn_mean:.3f})."
            )
        if pearson_mean < QUALITY_GATES["pbmc_vae_pearson_min"]:
            issues.append(
                "PBMC VAE Pearson correlation fell below "
                f"{QUALITY_GATES['pbmc_vae_pearson_min']:.2f} "
                f"(observed {pearson_mean:.3f})."
            )
        if silhouette_std > QUALITY_GATES["pbmc_vae_silhouette_std_max"]:
            issues.append(
                "PBMC VAE silhouette variance exceeded "
                f"{QUALITY_GATES['pbmc_vae_silhouette_std_max']:.2f} "
                f"(observed {silhouette_std:.3f})."
            )

    pbmc_classifier = metrics_frame[
        (metrics_frame["dataset"] == "pbmc3k_processed")
        & (metrics_frame["task"] == "classification")
        & (metrics_frame["model"] == "mlp_classifier")
    ]
    if pbmc_classifier.empty:
        issues.append("PBMC classifier benchmark did not run.")
    else:
        accuracy = float(pbmc_classifier["accuracy"].mean())
        macro_f1 = float(pbmc_classifier["macro_f1"].mean())
        if accuracy < QUALITY_GATES["pbmc_classifier_accuracy_min"]:
            issues.append(
                "PBMC classifier accuracy fell below "
                f"{QUALITY_GATES['pbmc_classifier_accuracy_min']:.2f} "
                f"(observed {accuracy:.3f})."
            )
        if macro_f1 < QUALITY_GATES["pbmc_classifier_macro_f1_min"]:
            issues.append(
                "PBMC classifier macro F1 fell below "
                f"{QUALITY_GATES['pbmc_classifier_macro_f1_min']:.2f} "
                f"(observed {macro_f1:.3f})."
            )

    foundation_datasets = ("pbmc3k_processed", "pbmc68k_reduced")
    winning_datasets = 0
    for dataset_name in foundation_datasets:
        pca_rows = metrics_frame[
            (metrics_frame["dataset"] == dataset_name)
            & (metrics_frame["task"] == "foundation")
            & (metrics_frame["model"] == "pca_foundation")
        ]
        scgpt_rows = metrics_frame[
            (metrics_frame["dataset"] == dataset_name)
            & (metrics_frame["task"] == "foundation")
            & (metrics_frame["model"] == "scgpt_whole_human")
        ]
        if pca_rows.empty or scgpt_rows.empty:
            issues.append(f"Foundation benchmark for '{dataset_name}' is incomplete.")
            continue
        for metric_name in ("silhouette",):
            pca_value = float(pca_rows[metric_name].mean())
            scgpt_value = float(scgpt_rows[metric_name].mean())
            if scgpt_value < pca_value - QUALITY_GATES["scgpt_vs_pca_max_drop"]:
                issues.append(
                    f"scGPT underperformed PCA on {dataset_name} {metric_name} by more than "
                    f"{QUALITY_GATES['scgpt_vs_pca_max_drop']:.2f} "
                    f"(PCA {pca_value:.3f}, scGPT {scgpt_value:.3f})."
                )
        if (
            float(scgpt_rows["silhouette"].mean())
            >= float(pca_rows["silhouette"].mean()) + QUALITY_GATES["scgpt_silhouette_win"]
        ):
            winning_datasets += 1
    if winning_datasets == 0:
        issues.append(
            "scGPT did not show a silhouette improvement over PCA on either PBMC dataset."
        )

    annotation_rows = metrics_frame[
        (metrics_frame["dataset"] == "pbmc3k_processed")
        & (metrics_frame["task"] == "foundation_annotation")
    ]
    tuned_model_names = (
        "scgpt_head",
        "scgpt_full_finetune",
        "scgpt_lora",
        "scgpt_adapter",
        "scgpt_prefix_tuning",
        "scgpt_ia3",
    )
    frozen_probe = annotation_rows[annotation_rows["model"] == "scgpt_frozen_probe"]
    head_rows = annotation_rows[annotation_rows["model"] == "scgpt_head"]
    lora_rows = annotation_rows[annotation_rows["model"] == "scgpt_lora"]
    if frozen_probe.empty or head_rows.empty or (profile != "ci" and lora_rows.empty):
        issues.append("Foundation annotation benchmark for 'pbmc3k_processed' is incomplete.")
        return issues

    frozen_accuracy = float(frozen_probe["accuracy"].mean())
    frozen_macro_f1 = float(frozen_probe["macro_f1"].mean())
    head_accuracy = float(head_rows["accuracy"].mean())
    head_macro_f1 = float(head_rows["macro_f1"].mean())
    lora_accuracy = float(lora_rows["accuracy"].mean()) if not lora_rows.empty else float("nan")
    lora_macro_f1 = float(lora_rows["macro_f1"].mean()) if not lora_rows.empty else float("nan")

    if head_accuracy < frozen_accuracy - QUALITY_GATES["scgpt_annotation_head_accuracy_drop_max"]:
        issues.append(
            "scGPT head-only tuning regressed too far from the frozen probe baseline on "
            "annotation accuracy "
            f"(frozen {frozen_accuracy:.3f}, head {head_accuracy:.3f})."
        )
    if head_macro_f1 < frozen_macro_f1 - QUALITY_GATES["scgpt_annotation_head_macro_f1_drop_max"]:
        issues.append(
            "scGPT head-only tuning regressed too far from the frozen probe baseline on "
            "annotation macro F1 "
            f"(frozen {frozen_macro_f1:.3f}, head {head_macro_f1:.3f})."
        )
    if profile != "ci":
        if lora_accuracy < head_accuracy - QUALITY_GATES["scgpt_annotation_lora_accuracy_drop_max"]:
            issues.append(
                "scGPT LoRA annotation accuracy regressed too far from head-only tuning "
                f"(head {head_accuracy:.3f}, LoRA {lora_accuracy:.3f})."
            )
        if lora_macro_f1 < head_macro_f1 - QUALITY_GATES["scgpt_annotation_lora_macro_f1_drop_max"]:
            issues.append(
                "scGPT LoRA annotation macro F1 regressed too far from head-only tuning "
                f"(head {head_macro_f1:.3f}, LoRA {lora_macro_f1:.3f})."
            )

    tuned_rows = annotation_rows[annotation_rows["model"].isin(tuned_model_names)]
    best_accuracy = (
        float(tuned_rows["accuracy"].max())
        if not tuned_rows.empty
        else max(head_accuracy, lora_accuracy)
    )
    best_macro_f1 = (
        float(tuned_rows["macro_f1"].max())
        if not tuned_rows.empty
        else max(head_macro_f1, lora_macro_f1)
    )
    if best_accuracy < QUALITY_GATES["scgpt_annotation_tuned_accuracy_min"]:
        issues.append(
            "No tuned scGPT annotation strategy reached the minimum accuracy target "
            f"of {QUALITY_GATES['scgpt_annotation_tuned_accuracy_min']:.2f} "
            f"(best {best_accuracy:.3f})."
        )
    if best_macro_f1 < QUALITY_GATES["scgpt_annotation_tuned_macro_f1_min"]:
        issues.append(
            "No tuned scGPT annotation strategy reached the minimum macro F1 target "
            f"of {QUALITY_GATES['scgpt_annotation_tuned_macro_f1_min']:.2f} "
            f"(best {best_macro_f1:.3f})."
        )

    pancreas_rows = metrics_frame[
        (metrics_frame["dataset"] == "openproblems_human_pancreas")
        & (metrics_frame["task"] == "foundation_annotation")
    ]
    if not pancreas_rows.empty:
        tuned_rows = pancreas_rows[pancreas_rows["model"].isin(tuned_model_names)]
        if tuned_rows.empty:
            issues.append(
                "OpenProblems human pancreas annotation benchmark is missing tuned strategies."
            )
        else:
            best_pancreas_macro_f1 = float(pancreas_rows["macro_f1"].max())
            best_tuned_macro_f1 = float(tuned_rows["macro_f1"].max())
            if best_tuned_macro_f1 < best_pancreas_macro_f1 - 0.03:
                issues.append(
                    "No tuned scGPT strategy stayed within 0.03 macro F1 of the best "
                    "OpenProblems human pancreas annotation result "
                    + (
                        f"(best overall {best_pancreas_macro_f1:.3f}, "
                        f"best tuned {best_tuned_macro_f1:.3f})."
                    )
                )
    return issues


def _evaluate_runtime(
    metrics_frame: pd.DataFrame,
    *,
    profile: str,
    suite_runtime_sec: float,
    tutorial_checks: dict[str, Any],
) -> dict[str, Any]:
    budgets = RUNTIME_BUDGETS[profile]
    benchmark_issues: list[str] = []
    tutorial_issues: list[str] = []
    warnings: list[str] = []
    suite_passed = suite_runtime_sec <= budgets["quality_suite_total_sec"]
    if not suite_passed:
        benchmark_issues.append(
            "Quality-suite runtime exceeded "
            f"{budgets['quality_suite_total_sec']:.0f}s "
            f"(observed {suite_runtime_sec:.1f}s)."
        )

    transformer_rows = metrics_frame[
        (metrics_frame["dataset"] == "pbmc3k_processed")
        & (metrics_frame["task"] == "representation")
        & (metrics_frame["model"] == "transformer_ae")
    ]
    transformer_mean = (
        float(transformer_rows["runtime_sec"].mean()) if not transformer_rows.empty else None
    )
    if transformer_mean is not None:
        if transformer_mean > budgets["transformer_ae_fail_sec"]:
            benchmark_issues.append(
                "PBMC Transformer AE runtime exceeded the hard threshold "
                f"of {budgets['transformer_ae_fail_sec']:.0f}s "
                f"(observed {transformer_mean:.1f}s)."
            )
        elif transformer_mean > budgets["transformer_ae_warn_sec"]:
            warnings.append(
                "PBMC Transformer AE runtime exceeded the warning threshold "
                f"of {budgets['transformer_ae_warn_sec']:.0f}s "
                f"(observed {transformer_mean:.1f}s)."
            )

    head_rows = metrics_frame[
        (metrics_frame["dataset"] == "pbmc3k_processed")
        & (metrics_frame["task"] == "foundation_annotation")
        & (metrics_frame["model"] == "scgpt_head")
    ]
    head_mean = float(head_rows["runtime_sec"].mean()) if not head_rows.empty else None
    if head_mean is not None:
        if head_mean > budgets["scgpt_head_fail_sec"]:
            benchmark_issues.append(
                "scGPT head-only annotation runtime exceeded the hard threshold "
                f"of {budgets['scgpt_head_fail_sec']:.0f}s "
                f"(observed {head_mean:.1f}s)."
            )
        elif head_mean > budgets["scgpt_head_warn_sec"]:
            warnings.append(
                "scGPT head-only annotation runtime exceeded the warning threshold "
                f"of {budgets['scgpt_head_warn_sec']:.0f}s "
                f"(observed {head_mean:.1f}s)."
            )

    lora_rows = metrics_frame[
        (metrics_frame["dataset"] == "pbmc3k_processed")
        & (metrics_frame["task"] == "foundation_annotation")
        & (metrics_frame["model"] == "scgpt_lora")
    ]
    lora_mean = float(lora_rows["runtime_sec"].mean()) if not lora_rows.empty else None
    if lora_mean is not None:
        if lora_mean > budgets["scgpt_lora_fail_sec"]:
            benchmark_issues.append(
                "scGPT LoRA annotation runtime exceeded the hard threshold "
                f"of {budgets['scgpt_lora_fail_sec']:.0f}s "
                f"(observed {lora_mean:.1f}s)."
            )
        elif lora_mean > budgets["scgpt_lora_warn_sec"]:
            warnings.append(
                "scGPT LoRA annotation runtime exceeded the warning threshold "
                f"of {budgets['scgpt_lora_warn_sec']:.0f}s "
                f"(observed {lora_mean:.1f}s)."
            )

    tutorial_runtime = tutorial_checks["runtime"]["total_sec"]
    tutorial_budget = tutorial_checks["runtime"]["budget_sec"]
    tutorial_passed = bool(tutorial_checks["runtime"]["passed"])
    if tutorial_checks["validated"] and not tutorial_passed:
        tutorial_issues.append(
            "Tutorial suite runtime exceeded the configured budget "
            f"(observed {tutorial_runtime:.1f}s, budget {tutorial_budget:.0f}s)."
        )

    per_model_means = (
        metrics_frame.groupby(["dataset", "task", "model"])["runtime_sec"]
        .mean()
        .reset_index()
        .rename(columns={"runtime_sec": "runtime_sec_mean"})
        .to_dict(orient="records")
    )
    return {
        "quality_suite": {
            "total_sec": suite_runtime_sec,
            "budget_sec": budgets["quality_suite_total_sec"],
            "passed": suite_passed,
        },
        "tutorials": {
            "validated": tutorial_checks["validated"],
            "total_sec": tutorial_runtime,
            "budget_sec": tutorial_budget,
            "passed": tutorial_passed,
        },
        "transformer_ae": {
            "mean_runtime_sec": transformer_mean,
            "warn_sec": budgets["transformer_ae_warn_sec"],
            "fail_sec": budgets["transformer_ae_fail_sec"],
        },
        "scgpt_head": {
            "mean_runtime_sec": head_mean,
            "warn_sec": budgets["scgpt_head_warn_sec"],
            "fail_sec": budgets["scgpt_head_fail_sec"],
        },
        "scgpt_lora": {
            "mean_runtime_sec": lora_mean,
            "warn_sec": budgets["scgpt_lora_warn_sec"],
            "fail_sec": budgets["scgpt_lora_fail_sec"],
        },
        "per_model_means": per_model_means,
        "warnings": warnings,
        "benchmark_issues": benchmark_issues,
        "tutorial_issues": tutorial_issues,
        "issues": [*benchmark_issues, *tutorial_issues],
    }


def build_summary(
    metrics_frame: pd.DataFrame,
    *,
    profile: str,
    suite_runtime_sec: float | None = None,
    tutorial_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    suite_runtime = (
        float(metrics_frame["runtime_sec"].sum())
        if suite_runtime_sec is None
        else suite_runtime_sec
    )
    aggregated = aggregate_metrics(metrics_frame)
    gate_issues = evaluate_quality_gates(metrics_frame, profile=profile)
    missing_runs = _find_missing_runs(metrics_frame, profile)
    benchmark_artifact_checks = _collect_benchmark_artifact_checks(metrics_frame)
    tutorial_checks = _normalize_tutorial_summary(tutorial_summary)
    runtime = _evaluate_runtime(
        metrics_frame,
        profile=profile,
        suite_runtime_sec=suite_runtime,
        tutorial_checks=tutorial_checks,
    )
    benchmark_ready = (
        not gate_issues
        and not missing_runs
        and benchmark_artifact_checks["passed"]
        and not runtime["benchmark_issues"]
    )
    release_rc_ready = (
        benchmark_ready
        and tutorial_checks["validated"]
        and tutorial_checks["passed"]
        and tutorial_checks["artifact_checks"]["passed"]
        and not tutorial_checks["missing_notebooks"]
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "num_runs": int(len(metrics_frame)),
        "gates": {
            "passed": not gate_issues,
            "issues": gate_issues,
            "thresholds": QUALITY_GATES,
        },
        "runtime": runtime,
        "artifact_checks": {
            "benchmark": benchmark_artifact_checks,
            "tutorials": tutorial_checks,
        },
        "missing_runs": missing_runs,
        "benchmark_ready": benchmark_ready,
        "release_rc_ready": release_rc_ready,
        "aggregates": aggregated.to_dict(orient="records"),
    }


def render_summary_markdown(summary: dict[str, Any]) -> str:
    gates = summary["gates"]
    runtime = summary["runtime"]
    benchmark_artifacts = summary["artifact_checks"]["benchmark"]
    tutorial_checks = summary["artifact_checks"]["tutorials"]
    lines = [
        "# scDLKit quality-suite summary",
        "",
        f"- Profile: `{summary['profile']}`",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Total runs: `{summary['num_runs']}`",
        f"- Benchmark ready: `{summary['benchmark_ready']}`",
        f"- Release RC ready: `{summary['release_rc_ready']}`",
        "",
        "## Quality gates",
        "",
    ]
    if gates["issues"]:
        lines.extend(f"- {issue}" for issue in gates["issues"])
    else:
        lines.append("- All configured scientific quality gates passed.")

    lines.extend(["", "## Missing runs", ""])
    if summary["missing_runs"]:
        lines.extend(f"- `{missing}`" for missing in summary["missing_runs"])
    else:
        lines.append("- All required dataset/task/model/seed runs are present.")

    lines.extend(
        [
            "",
            "## Runtime",
            "",
            "- Quality suite: "
            f"`{runtime['quality_suite']['total_sec']:.1f}s` / "
            f"`{runtime['quality_suite']['budget_sec']:.0f}s` "
            f"(passed: `{runtime['quality_suite']['passed']}`)",
        ]
    )
    if runtime["tutorials"]["validated"]:
        lines.append(
            "- Tutorials: "
            f"`{runtime['tutorials']['total_sec']:.1f}s` / "
            f"`{runtime['tutorials']['budget_sec']:.0f}s` "
            f"(passed: `{runtime['tutorials']['passed']}`)"
        )
    else:
        lines.append("- Tutorials: not validated in this summary.")
    if runtime["transformer_ae"]["mean_runtime_sec"] is not None:
        lines.append(
            "- PBMC Transformer AE mean runtime: "
            f"`{runtime['transformer_ae']['mean_runtime_sec']:.1f}s` "
            f"(warn: `{runtime['transformer_ae']['warn_sec']:.0f}s`, "
            f"fail: `{runtime['transformer_ae']['fail_sec']:.0f}s`)"
        )
    if runtime["scgpt_head"]["mean_runtime_sec"] is not None:
        lines.append(
            "- scGPT head-only annotation mean runtime: "
            f"`{runtime['scgpt_head']['mean_runtime_sec']:.1f}s` "
            f"(warn: `{runtime['scgpt_head']['warn_sec']:.0f}s`, "
            f"fail: `{runtime['scgpt_head']['fail_sec']:.0f}s`)"
        )
    if runtime["scgpt_lora"]["mean_runtime_sec"] is not None:
        lines.append(
            "- scGPT LoRA annotation mean runtime: "
            f"`{runtime['scgpt_lora']['mean_runtime_sec']:.1f}s` "
            f"(warn: `{runtime['scgpt_lora']['warn_sec']:.0f}s`, "
            f"fail: `{runtime['scgpt_lora']['fail_sec']:.0f}s`)"
        )
    if runtime["warnings"]:
        lines.extend(["", "### Runtime warnings", ""])
        lines.extend(f"- {warning}" for warning in runtime["warnings"])
    if runtime["issues"]:
        lines.extend(["", "### Runtime issues", ""])
        lines.extend(f"- {issue}" for issue in runtime["issues"])

    lines.extend(
        [
            "",
            "## Artifact checks",
            "",
            "- Benchmark artifacts passed: "
            f"`{benchmark_artifacts['passed']}` "
            f"({benchmark_artifacts['checked_files']} files checked)",
        ]
    )
    if benchmark_artifacts["missing_files"]:
        lines.extend(
            f"- Missing benchmark artifact: `{path}`"
            for path in benchmark_artifacts["missing_files"]
        )
    lines.append(f"- Tutorial artifacts validated: `{tutorial_checks['validated']}`")
    if tutorial_checks["validated"]:
        lines.append(
            f"- Tutorial artifacts passed: `{tutorial_checks['artifact_checks']['passed']}`"
        )
        if tutorial_checks["missing_notebooks"]:
            lines.extend(
                f"- Missing tutorial summary entry: `{name}`"
                for name in tutorial_checks["missing_notebooks"]
            )
        if tutorial_checks["artifact_checks"]["missing_files"]:
            lines.extend(
                f"- Missing tutorial artifact: `{path}`"
                for path in tutorial_checks["artifact_checks"]["missing_files"]
            )
        if tutorial_checks["issues"]:
            lines.extend(f"- Tutorial issue: {issue}" for issue in tutorial_checks["issues"])

    lines.extend(
        [
            "",
            "## Aggregated metrics",
            "",
            "| Dataset | Task | Model | Key means |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in summary["aggregates"]:
        key_means = []
        metric_names = (
            "silhouette_mean",
            "knn_label_consistency_mean",
            "pearson_mean",
            "accuracy_mean",
            "macro_f1_mean",
            "probe_accuracy_mean",
            "probe_macro_f1_mean",
            "trainable_parameters_mean",
            "runtime_sec_mean",
        )
        for metric in metric_names:
            if metric in row and not math.isnan(float(row[metric])):
                key_means.append(f"{metric}={float(row[metric]):.3f}")
        lines.append(
            "| "
            f"{row['dataset']} | {row['task']} | {row['model']} | "
            f"{', '.join(key_means) or 'n/a'} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `pbmc3k_processed` is the primary release-quality representation benchmark.",
            "- `pbmc68k_reduced` is the experimental foundation-model comparison benchmark.",
            "- `openproblems_human_pancreas` is the external human annotation evidence benchmark.",
            "- `foundation_annotation` benchmarks compare frozen scGPT, head-only tuning, and "
            "LoRA tuning for cell-type annotation.",
            "- `paul15` is the secondary built-in benchmark for checking dataset sensitivity.",
            "- `PCA` is the classical reference baseline for asking whether deep learning helps.",
            "- `pca_foundation` is the subset-matched PCA reference for scGPT comparisons.",
            "- `scgpt_whole_human` is the experimental frozen embedding baseline for "
            "human PBMC data.",
            "- `scgpt_head` and `scgpt_lora` are experimental annotation fine-tuning "
            "strategies built on the official `whole-human` checkpoint.",
            "- `transformer_ae` uses a compact CPU-friendly configuration in the quality suite: "
            "`patch_size=48`, `d_model=64`, `n_heads=2`, `n_layers=1`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _save_comparison_plot(metrics_frame: pd.DataFrame, output_dir: Path) -> None:
    from matplotlib import pyplot as plt

    from scdlkit.visualization.compare import plot_model_comparison

    pbmc_representation = metrics_frame[
        (metrics_frame["dataset"] == "pbmc3k_processed")
        & (metrics_frame["task"] == "representation")
    ].copy()
    if pbmc_representation.empty:
        return
    summary = (
        pbmc_representation.groupby("model", as_index=False)[["silhouette", "runtime_sec"]]
        .mean(numeric_only=True)
        .sort_values("model")
    )
    comparison_fig, _ = plot_model_comparison(summary, metric="silhouette")
    comparison_fig.savefig(
        output_dir / "pbmc_representation_silhouette.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(comparison_fig)


def _write_summary_files(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")


def run_quality_suite(
    *,
    profile: str,
    output_dir: Path,
    foundation_annotation_metrics: dict[tuple[str, str, int], dict[str, Any]] | None = None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    profile_config = PROFILE_DEFAULTS[profile]
    precomputed_annotation_metrics = foundation_annotation_metrics or {}

    for dataset_name, model_seeds in profile_config["representation"].items():
        adata, spec = _load_dataset(dataset_name, profile=profile)
        for model_name, seeds in model_seeds.items():
            for seed in seeds:
                if model_name == "pca":
                    records.append(
                        run_pca_baseline(
                            dataset_name=dataset_name,
                            adata=adata,
                            label_key=spec.label_key,
                            batch_key=spec.batch_key,
                            seed=seed,
                            output_root=output_dir,
                        )
                    )
                elif model_name == "scgpt_whole_human":
                    records.append(
                        run_scgpt_embedding_baseline(
                            dataset_name=dataset_name,
                            adata=adata,
                            label_key=spec.label_key,
                            batch_key=spec.batch_key,
                            seed=seed,
                            output_root=output_dir,
                        )
                    )
                else:
                    records.append(
                        run_representation_runner(
                            dataset_name=dataset_name,
                            adata=adata,
                            label_key=spec.label_key,
                            batch_key=spec.batch_key,
                            model_name=model_name,
                            seed=seed,
                            profile=profile,
                            output_root=output_dir,
                        )
                    )

    for dataset_name, model_seeds in profile_config["classification"].items():
        adata, spec = _load_dataset(dataset_name, profile=profile)
        for model_name, seeds in model_seeds.items():
            for seed in seeds:
                if model_name == "mlp_classifier":
                    records.append(
                        run_classification_runner(
                            dataset_name=dataset_name,
                            adata=adata,
                            label_key=spec.label_key,
                            seed=seed,
                            profile=profile,
                            output_root=output_dir,
                        )
                    )
                else:
                    records.append(
                        run_logistic_regression_pca(
                            dataset_name=dataset_name,
                            adata=adata,
                            label_key=spec.label_key,
                            seed=seed,
                            output_root=output_dir,
                        )
                    )

    for dataset_name, model_seeds in profile_config.get("foundation", {}).items():
        adata, spec = _load_dataset(dataset_name, profile=profile)
        for model_name, seeds in model_seeds.items():
            for seed in seeds:
                if model_name == "pca_foundation":
                    records.append(
                        run_foundation_pca_reference(
                            dataset_name=dataset_name,
                            adata=adata,
                            label_key=spec.label_key,
                            batch_key=spec.batch_key,
                            seed=seed,
                            output_root=output_dir,
                        )
                    )
                else:
                    records.append(
                        run_scgpt_embedding_baseline(
                            dataset_name=dataset_name,
                            adata=adata,
                            label_key=spec.label_key,
                            batch_key=spec.batch_key,
                            seed=seed,
                            output_root=output_dir,
                        )
                    )

    for dataset_name, model_seeds in profile_config.get("foundation_annotation", {}).items():
        adata, spec = _load_dataset(dataset_name, profile=profile)
        for model_name, seeds in model_seeds.items():
            for seed in seeds:
                if model_name == "pca_logistic_annotation":
                    records.append(
                        run_foundation_annotation_pca_logistic(
                            dataset_name=dataset_name,
                            adata=adata,
                            label_key=spec.label_key,
                            batch_key=spec.batch_key,
                            seed=seed,
                            output_root=output_dir,
                            profile=profile,
                        )
                    )
                else:
                    precomputed_key = (dataset_name, model_name, seed)
                    if precomputed_key in precomputed_annotation_metrics:
                        records.append(dict(precomputed_annotation_metrics[precomputed_key]))
                    else:
                        records.append(
                            run_scgpt_annotation_strategy(
                                dataset_name=dataset_name,
                                adata=adata,
                                label_key=spec.label_key,
                                batch_key=spec.batch_key,
                                seed=seed,
                                output_root=output_dir,
                                model_name=model_name,
                                profile=profile,
                            )
                        )

    metrics_frame = pd.DataFrame.from_records(records).sort_values(
        ["dataset", "task", "model", "seed"]
    )
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)
    _save_comparison_plot(metrics_frame, output_dir)
    return metrics_frame


def load_tutorial_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_foundation_annotation_metrics(
    path: Path | None,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    frame = pd.read_csv(path)
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for record in frame.to_dict(orient="records"):
        artifact_dir = Path(str(record["artifact_dir"]))
        if not artifact_dir.is_absolute():
            artifact_dir = (path.parent / artifact_dir).resolve()
        record["artifact_dir"] = str(artifact_dir)
        rows[(str(record["dataset"]), str(record["model"]), int(record["seed"]))] = record
    return rows


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or _default_output_dir(args.profile)
    suite_started_at = perf_counter()
    metrics_frame = run_quality_suite(
        profile=args.profile,
        output_dir=output_dir,
        foundation_annotation_metrics=load_foundation_annotation_metrics(
            args.foundation_annotation_metrics
        ),
    )
    suite_runtime_sec = perf_counter() - suite_started_at
    tutorial_summary = load_tutorial_summary(args.tutorial_summary)
    summary = build_summary(
        metrics_frame,
        profile=args.profile,
        suite_runtime_sec=suite_runtime_sec,
        tutorial_summary=tutorial_summary,
    )
    _write_summary_files(output_dir, summary)
    print(render_summary_markdown(summary))
    if args.check and not summary["benchmark_ready"]:
        raise SystemExit(1)
    if args.require_rc and not summary["release_rc_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
