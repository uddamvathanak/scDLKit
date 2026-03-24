"""Run the Milestone 1 annotation-pillar benchmark matrix."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
from torch.utils.data import Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

matplotlib.use("Agg")

from run_quality_suite import (  # noqa: E402
    _batch_metric_summary,
    _batch_metrics_frame,
    _foundation_annotation_profile,
    _load_dataset,
    _process_peak_memory_mb,
    _save_confusion_plot,
    _save_scanpy_umap,
    _save_trainable_annotation_checkpoint,
    _write_report,
)

from scdlkit.benchmarks.annotation_spec import (  # noqa: E402
    ANNOTATION_DATASET_SPECS,
    ANNOTATION_REGIME_SPECS,
    ANNOTATION_TASK_SPEC,
    CrossStudyFoldSpec,
)
from scdlkit.evaluation.metrics import classification_metrics  # noqa: E402
from scdlkit.foundation import (  # noqa: E402
    AdapterConfig,
    IA3Config,
    LoRAConfig,
    PrefixTuningConfig,
    count_trainable_parameters,
    load_scgpt_annotation_model,
    load_scgpt_model,
    prepare_scgpt_data,
)
from scdlkit.training import Trainer  # noqa: E402

BENCHMARK_MODELS: tuple[str, ...] = (
    "pca_logistic_annotation",
    "scgpt_frozen_probe",
    "scgpt_head",
    "scgpt_full_finetune",
    "scgpt_lora",
    "scgpt_adapter",
    "scgpt_prefix_tuning",
    "scgpt_ia3",
)

MODEL_DISPLAY_NAMES = {
    "pca_logistic_annotation": "PCA + logistic regression",
    "scgpt_frozen_probe": "Frozen scGPT probe",
    "scgpt_head": "scGPT head-only tuning",
    "scgpt_full_finetune": "scGPT full fine-tuning",
    "scgpt_lora": "scGPT LoRA",
    "scgpt_adapter": "scGPT adapters",
    "scgpt_prefix_tuning": "scGPT prefix tuning",
    "scgpt_ia3": "scGPT IA3",
}

TRAINABLE_MODELS = {
    "scgpt_head",
    "scgpt_full_finetune",
    "scgpt_lora",
    "scgpt_adapter",
    "scgpt_prefix_tuning",
    "scgpt_ia3",
}


@dataclass(frozen=True, slots=True)
class SplitPlan:
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    label_fraction: float | None = None
    cross_study_fold: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        nargs="*",
        choices=tuple(ANNOTATION_DATASET_SPECS),
        default=tuple(ANNOTATION_DATASET_SPECS),
        help="Datasets to include in the benchmark.",
    )
    parser.add_argument(
        "--regime",
        nargs="*",
        choices=tuple(ANNOTATION_REGIME_SPECS),
        default=tuple(ANNOTATION_REGIME_SPECS),
        help="Regimes to include in the benchmark.",
    )
    parser.add_argument(
        "--profile",
        choices=("quickstart", "full"),
        default="full",
        help="Benchmark execution profile.",
    )
    parser.add_argument(
        "--strategies",
        nargs="*",
        choices=BENCHMARK_MODELS,
        default=BENCHMARK_MODELS,
        help="Strategy matrix to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "annotation_pillar",
        help="Artifact output directory.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=ANNOTATION_REGIME_SPECS["full_label"].seeds,
        help="Override the default benchmark seeds.",
    )
    parser.add_argument(
        "--label-fractions",
        nargs="*",
        type=float,
        default=ANNOTATION_REGIME_SPECS["low_label"].label_fractions,
        help="Override the default low-label fractions.",
    )
    parser.add_argument(
        "--cross-study-folds",
        nargs="*",
        default=tuple(
            fold.name for fold in ANNOTATION_REGIME_SPECS["cross_study"].cross_study_folds
        ),
        help="Override the default cross-study folds by name.",
    )
    return parser.parse_args()


def _benchmark_profile(profile: str) -> tuple[str, dict[str, int]]:
    if profile == "quickstart":
        return "quickstart", _foundation_annotation_profile("ci")
    return "full", _foundation_annotation_profile("full")


def _to_dense(matrix: Any) -> np.ndarray:
    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    return dense.astype(np.float32, copy=False)


def _expression_adata(adata: AnnData) -> AnnData:
    if adata.raw is not None:
        return adata.raw.to_adata()
    return adata


def _encode_labels(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    categorical = pd.Categorical(values.astype(str))
    categories = [str(value) for value in categorical.categories]
    return categorical.codes.astype(np.int64), categories


def _expand_probabilities(
    probabilities: np.ndarray,
    classes: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    if probabilities.shape[1] == num_classes:
        return probabilities.astype(np.float32, copy=False)
    expanded = np.zeros((probabilities.shape[0], num_classes), dtype=np.float32)
    class_indices = np.asarray(classes, dtype=np.int64)
    expanded[:, class_indices] = np.asarray(probabilities, dtype=np.float32)
    return expanded


def _safe_split(
    indices: np.ndarray,
    labels: np.ndarray,
    *,
    train_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    stratify = labels if len(np.unique(labels)) > 1 else None
    try:
        train_idx, held_out_idx = train_test_split(
            indices,
            train_size=train_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        train_idx, held_out_idx = train_test_split(
            indices,
            train_size=train_size,
            random_state=seed,
            stratify=None,
        )
    return np.sort(np.asarray(train_idx, dtype=np.int64)), np.sort(
        np.asarray(held_out_idx, dtype=np.int64)
    )


def build_full_label_split(
    adata: AnnData,
    *,
    label_key: str,
    seed: int,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> SplitPlan:
    labels, _ = _encode_labels(adata.obs[label_key])
    indices = np.arange(adata.n_obs, dtype=np.int64)
    train_idx, remaining_idx = _safe_split(
        indices,
        labels,
        train_size=1.0 - val_size - test_size,
        seed=seed,
    )
    remaining_labels = labels[remaining_idx]
    val_fraction = val_size / (val_size + test_size)
    val_idx, test_idx = _safe_split(
        remaining_idx,
        remaining_labels,
        train_size=val_fraction,
        seed=seed + 1,
    )
    return SplitPlan(train_indices=train_idx, val_indices=val_idx, test_indices=test_idx)


def build_low_label_split(
    full_split: SplitPlan,
    adata: AnnData,
    *,
    label_key: str,
    label_fraction: float,
    seed: int,
) -> SplitPlan:
    labels, _ = _encode_labels(adata.obs[label_key])
    train_idx = np.asarray(full_split.train_indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    train_labels = labels[train_idx]
    for class_value in np.unique(train_labels):
        class_indices = train_idx[train_labels == class_value].copy()
        rng.shuffle(class_indices)
        keep = max(1, int(round(len(class_indices) * label_fraction)))
        selected.extend(int(index) for index in class_indices[:keep])
    subsampled_train = np.sort(np.asarray(selected, dtype=np.int64))
    return SplitPlan(
        train_indices=subsampled_train,
        val_indices=np.asarray(full_split.val_indices, dtype=np.int64),
        test_indices=np.asarray(full_split.test_indices, dtype=np.int64),
        label_fraction=label_fraction,
    )


def build_cross_study_split(
    adata: AnnData,
    *,
    label_key: str,
    batch_key: str,
    fold: CrossStudyFoldSpec,
    seed: int,
    val_size: float = 0.15,
) -> SplitPlan:
    if batch_key not in adata.obs:
        raise ValueError(f"Batch key '{batch_key}' is not present in adata.obs.")
    batch_values = adata.obs[batch_key].astype(str).to_numpy()
    test_mask = np.isin(batch_values, np.asarray(fold.held_out_batches, dtype=object))
    if not bool(test_mask.any()):
        raise ValueError(f"Cross-study fold '{fold.name}' matched no batches in '{batch_key}'.")
    test_idx = np.flatnonzero(test_mask).astype(np.int64)
    train_pool = np.flatnonzero(~test_mask).astype(np.int64)
    labels, _ = _encode_labels(adata.obs[label_key])
    train_idx, val_idx = _safe_split(
        train_pool,
        labels[train_pool],
        train_size=1.0 - val_size,
        seed=seed,
    )
    return SplitPlan(
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=np.sort(test_idx),
        cross_study_fold=fold.name,
    )


def _strategy_sort_value(model_name: str) -> int:
    try:
        return BENCHMARK_MODELS.index(model_name)
    except ValueError:
        return len(BENCHMARK_MODELS)


def _strategy_config_for_model(model_name: str) -> tuple[str, Any, float]:
    if model_name == "scgpt_head":
        return "head", None, 5e-3
    if model_name == "scgpt_full_finetune":
        return "full_finetune", None, 5e-4
    if model_name == "scgpt_lora":
        return "lora", LoRAConfig(rank=4, alpha=8.0, dropout=0.05), 2e-3
    if model_name == "scgpt_adapter":
        return "adapter", AdapterConfig(bottleneck_dim=64, dropout=0.05), 2e-3
    if model_name == "scgpt_prefix_tuning":
        return "prefix_tuning", PrefixTuningConfig(prefix_length=20, dropout=0.05), 2e-3
    if model_name == "scgpt_ia3":
        return "ia3", IA3Config(init_scale=1.0), 2e-3
    raise ValueError(f"Unsupported trainable annotation model '{model_name}'.")


def _run_output_dir(
    output_dir: Path,
    *,
    regime: str,
    dataset_name: str,
    model_name: str,
    seed: int,
    label_fraction: float | None,
    cross_study_fold: str | None,
) -> Path:
    parts = [output_dir, "runs", regime, dataset_name, model_name, f"seed_{seed}"]
    if label_fraction is not None:
        parts.append(f"fraction_{label_fraction:.2f}".replace(".", "p"))
    if cross_study_fold is not None:
        parts.append(f"fold_{cross_study_fold}")
    path = Path(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _subset_dataset(dataset: Dataset[dict[str, Any]], indices: np.ndarray) -> Subset[Any]:
    return Subset(dataset, [int(index) for index in np.asarray(indices, dtype=np.int64)])


def _checkpoint_size_bytes(best_model_artifact: str | None) -> int | None:
    if not best_model_artifact:
        return None
    checkpoint_path = Path(best_model_artifact) / "model_state.pt"
    if not checkpoint_path.exists():
        return None
    return int(checkpoint_path.stat().st_size)


def _scalar_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return float("nan")


def _report_payload(
    *,
    dataset_name: str,
    model_name: str,
    metrics: dict[str, Any],
    trainable_parameters: int,
    runtime_sec: float,
    checkpoint_size_bytes: int | None,
    split_plan: SplitPlan,
) -> dict[str, Any]:
    return {
        "Dataset": dataset_name,
        "Strategy": MODEL_DISPLAY_NAMES[model_name],
        "Trainable parameters": trainable_parameters,
        "Runtime (sec)": runtime_sec,
        "Checkpoint size (bytes)": (
            checkpoint_size_bytes if checkpoint_size_bytes is not None else 0
        ),
        "Macro F1": _scalar_value(metrics, "macro_f1"),
        "Accuracy": _scalar_value(metrics, "accuracy"),
        "Balanced accuracy": _scalar_value(metrics, "balanced_accuracy"),
        "AUROC OVR": _scalar_value(metrics, "auroc_ovr"),
        "Low-label fraction": (
            split_plan.label_fraction if split_plan.label_fraction is not None else ""
        ),
        "Cross-study fold": split_plan.cross_study_fold or "",
    }


def _run_pca_logistic_strategy(
    *,
    adata: AnnData,
    label_key: str,
    batch_key: str | None,
    split_plan: SplitPlan,
    dataset_name: str,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    source_adata = _expression_adata(adata)
    x_matrix = _to_dense(source_adata.X)
    labels, label_categories = _encode_labels(adata.obs[label_key])
    train_x = x_matrix[split_plan.train_indices]
    test_x = x_matrix[split_plan.test_indices]
    train_y = labels[split_plan.train_indices]
    test_y = labels[split_plan.test_indices]
    n_components = max(1, min(32, train_x.shape[0] - 1, train_x.shape[1]))

    started_at = perf_counter()
    pca = PCA(n_components=n_components, random_state=seed)
    train_latent = pca.fit_transform(train_x)
    test_latent = pca.transform(test_x)
    classifier = LogisticRegression(max_iter=1000, random_state=seed)
    classifier.fit(train_latent, train_y)
    logits = _expand_probabilities(
        classifier.predict_proba(test_latent),
        np.asarray(classifier.classes_, dtype=np.int64),
        num_classes=len(label_categories),
    )
    runtime_sec = perf_counter() - started_at

    metrics = classification_metrics(test_y, logits)
    test_adata = adata[np.asarray(split_plan.test_indices, dtype=np.int64)].copy()
    batch_metrics = _batch_metrics_frame(
        obs=test_adata.obs,
        batch_key=batch_key,
        y_true=test_y,
        logits=logits,
    )
    batch_metrics.to_csv(output_dir / "batch_metrics.csv", index=False)
    _save_confusion_plot(metrics["confusion_matrix"], label_categories, output_dir)
    _save_scanpy_umap(test_adata, test_latent, label_key, output_dir / "latent_umap.png", seed=seed)

    pd.DataFrame(
        [
            {
                **metrics,
                **_batch_metric_summary(batch_metrics),
                "trainable_parameters": 0,
                "checkpoint_size_bytes": 0,
            }
        ]
    ).to_csv(output_dir / "report.csv", index=False)
    _write_report(
        output_dir,
        "Annotation benchmark PCA + logistic regression",
        _report_payload(
            dataset_name=dataset_name,
            model_name="pca_logistic_annotation",
            metrics=metrics,
            trainable_parameters=0,
            runtime_sec=runtime_sec,
            checkpoint_size_bytes=0,
            split_plan=split_plan,
        ),
    )
    return {
        "dataset": dataset_name,
        "regime": "",
        "model": "pca_logistic_annotation",
        "strategy": MODEL_DISPLAY_NAMES["pca_logistic_annotation"],
        "seed": seed,
        "label_fraction": split_plan.label_fraction,
        "cross_study_fold": split_plan.cross_study_fold,
        "runtime_sec": runtime_sec,
        "peak_memory_mb": _process_peak_memory_mb(),
        "trainable_parameters": 0,
        "checkpoint_size_bytes": 0,
        "artifact_dir": str(output_dir),
        "best_model_artifact": None,
        "batch_metrics_artifact": str(output_dir / "batch_metrics.csv"),
        "confusion_matrix_artifact": str(output_dir / "confusion_matrix.png"),
        "latent_umap_artifact": str(output_dir / "latent_umap.png"),
        **{key: value for key, value in metrics.items() if isinstance(value, (int, float))},
        **_batch_metric_summary(batch_metrics),
    }


def _run_scgpt_strategy(
    *,
    adata: AnnData,
    label_key: str,
    batch_key: str | None,
    split_plan: SplitPlan,
    dataset_name: str,
    model_name: str,
    seed: int,
    profile_settings: dict[str, int],
    output_dir: Path,
) -> dict[str, Any]:
    prepared = prepare_scgpt_data(
        adata,
        checkpoint="whole-human",
        label_key=label_key,
        batch_size=64,
        use_raw=True,
        min_gene_overlap=profile_settings["min_gene_overlap"],
    )
    train_dataset = _subset_dataset(prepared.dataset, split_plan.train_indices)
    val_dataset = _subset_dataset(prepared.dataset, split_plan.val_indices)
    test_dataset = _subset_dataset(prepared.dataset, split_plan.test_indices)
    test_adata = adata[np.asarray(split_plan.test_indices, dtype=np.int64)].copy()
    label_categories = list(prepared.label_categories or [])

    started_at = perf_counter()
    if model_name == "scgpt_frozen_probe":
        model = load_scgpt_model("whole-human", device="auto")
        trainer = Trainer(
            model=model,
            task="representation",
            batch_size=prepared.batch_size,
            device="auto",
            epochs=1,
        )
        train_predictions = trainer.predict_dataset(train_dataset)
        test_predictions = trainer.predict_dataset(test_dataset)
        classifier = LogisticRegression(max_iter=1000, random_state=seed)
        classifier.fit(train_predictions["latent"], train_predictions["y"])
        logits = _expand_probabilities(
            classifier.predict_proba(test_predictions["latent"]),
            np.asarray(classifier.classes_, dtype=np.int64),
            num_classes=len(label_categories),
        )
        predictions = {
            "y": test_predictions["y"],
            "logits": logits,
            "latent": test_predictions["latent"],
        }
        trainable_parameters = 0
        best_model_artifact = None
        checkpoint_size_bytes = 0
    else:
        tuning_strategy, strategy_config, lr = _strategy_config_for_model(model_name)
        epoch_key = {
            "head": "head_epochs",
            "full_finetune": "full_finetune_epochs",
            "lora": "lora_epochs",
            "adapter": "adapter_epochs",
            "prefix_tuning": "prefix_tuning_epochs",
            "ia3": "ia3_epochs",
        }[tuning_strategy]
        model = load_scgpt_annotation_model(
            num_classes=len(label_categories),
            checkpoint="whole-human",
            tuning_strategy=tuning_strategy,  # type: ignore[arg-type]
            label_categories=prepared.label_categories,
            strategy_config=strategy_config,
            device="auto",
        )
        trainer = Trainer(
            model=model,
            task="classification",
            batch_size=prepared.batch_size,
            epochs=profile_settings[epoch_key],
            lr=lr,
            device="auto",
            early_stopping_patience=3,
            seed=seed,
        )
        trainer.fit(train_dataset, val_dataset)
        predictions = trainer.predict_dataset(test_dataset)
        trainable_parameters = count_trainable_parameters(model)
        metrics_for_save = classification_metrics(predictions["y"], predictions["logits"])
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
                    "accuracy": metrics_for_save["accuracy"],
                    "macro_f1": metrics_for_save["macro_f1"],
                    "balanced_accuracy": metrics_for_save["balanced_accuracy"],
                    "runtime_sec": 0.0,
                },
                strategy_config=(
                    strategy_config.to_payload() if strategy_config is not None else None
                ),
            )
        )
        checkpoint_size_bytes = _checkpoint_size_bytes(best_model_artifact) or 0
    runtime_sec = perf_counter() - started_at

    metrics = classification_metrics(predictions["y"], predictions["logits"])
    batch_metrics = _batch_metrics_frame(
        obs=test_adata.obs,
        batch_key=batch_key,
        y_true=np.asarray(predictions["y"], dtype=np.int64),
        logits=np.asarray(predictions["logits"], dtype=np.float32),
    )
    batch_metrics.to_csv(output_dir / "batch_metrics.csv", index=False)
    _save_confusion_plot(metrics["confusion_matrix"], label_categories, output_dir)
    _save_scanpy_umap(
        test_adata,
        np.asarray(predictions["latent"], dtype=np.float32),
        label_key,
        output_dir / "latent_umap.png",
        seed=seed,
    )
    pd.DataFrame(
        [
            {
                **metrics,
                **_batch_metric_summary(batch_metrics),
                "trainable_parameters": trainable_parameters,
                "checkpoint_size_bytes": checkpoint_size_bytes,
            }
        ]
    ).to_csv(output_dir / "report.csv", index=False)
    _write_report(
        output_dir,
        "Annotation benchmark scGPT strategy",
        _report_payload(
            dataset_name=dataset_name,
            model_name=model_name,
            metrics=metrics,
            trainable_parameters=trainable_parameters,
            runtime_sec=runtime_sec,
            checkpoint_size_bytes=checkpoint_size_bytes,
            split_plan=split_plan,
        ),
    )
    return {
        "dataset": dataset_name,
        "regime": "",
        "model": model_name,
        "strategy": MODEL_DISPLAY_NAMES[model_name],
        "seed": seed,
        "label_fraction": split_plan.label_fraction,
        "cross_study_fold": split_plan.cross_study_fold,
        "runtime_sec": runtime_sec,
        "peak_memory_mb": _process_peak_memory_mb(),
        "trainable_parameters": trainable_parameters,
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "artifact_dir": str(output_dir),
        "best_model_artifact": best_model_artifact,
        "batch_metrics_artifact": str(output_dir / "batch_metrics.csv"),
        "confusion_matrix_artifact": str(output_dir / "confusion_matrix.png"),
        "latent_umap_artifact": str(output_dir / "latent_umap.png"),
        **{key: value for key, value in metrics.items() if isinstance(value, (int, float))},
        **_batch_metric_summary(batch_metrics),
    }


def _run_single_benchmark(
    *,
    dataset_name: str,
    adata: AnnData,
    label_key: str,
    batch_key: str | None,
    regime: str,
    split_plan: SplitPlan,
    model_name: str,
    seed: int,
    profile_settings: dict[str, int],
    output_dir: Path,
) -> dict[str, Any]:
    run_dir = _run_output_dir(
        output_dir,
        regime=regime,
        dataset_name=dataset_name,
        model_name=model_name,
        seed=seed,
        label_fraction=split_plan.label_fraction,
        cross_study_fold=split_plan.cross_study_fold,
    )
    if model_name == "pca_logistic_annotation":
        row = _run_pca_logistic_strategy(
            adata=adata,
            label_key=label_key,
            batch_key=batch_key,
            split_plan=split_plan,
            dataset_name=dataset_name,
            seed=seed,
            output_dir=run_dir,
        )
    else:
        row = _run_scgpt_strategy(
            adata=adata,
            label_key=label_key,
            batch_key=batch_key,
            split_plan=split_plan,
            dataset_name=dataset_name,
            model_name=model_name,
            seed=seed,
            profile_settings=profile_settings,
            output_dir=run_dir,
        )
    row["regime"] = regime
    return row


def _collect_full_label_rows(
    *,
    dataset_name: str,
    adata: AnnData,
    label_key: str,
    batch_key: str | None,
    seeds: tuple[int, ...],
    strategies: tuple[str, ...],
    output_dir: Path,
    profile_settings: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        split_plan = build_full_label_split(adata, label_key=label_key, seed=seed)
        for model_name in strategies:
            rows.append(
                _run_single_benchmark(
                    dataset_name=dataset_name,
                    adata=adata,
                    label_key=label_key,
                    batch_key=batch_key,
                    regime="full_label",
                    split_plan=split_plan,
                    model_name=model_name,
                    seed=seed,
                    profile_settings=profile_settings,
                    output_dir=output_dir,
                )
            )
    return rows


def _collect_low_label_rows(
    *,
    dataset_name: str,
    adata: AnnData,
    label_key: str,
    batch_key: str | None,
    seeds: tuple[int, ...],
    label_fractions: tuple[float, ...],
    strategies: tuple[str, ...],
    output_dir: Path,
    profile_settings: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        full_split = build_full_label_split(adata, label_key=label_key, seed=seed)
        for label_fraction in label_fractions:
            split_plan = build_low_label_split(
                full_split,
                adata,
                label_key=label_key,
                label_fraction=label_fraction,
                seed=seed,
            )
            for model_name in strategies:
                rows.append(
                    _run_single_benchmark(
                        dataset_name=dataset_name,
                        adata=adata,
                        label_key=label_key,
                        batch_key=batch_key,
                        regime="low_label",
                        split_plan=split_plan,
                        model_name=model_name,
                        seed=seed,
                        profile_settings=profile_settings,
                        output_dir=output_dir,
                    )
                )
    return rows


def _collect_cross_study_rows(
    *,
    dataset_name: str,
    adata: AnnData,
    label_key: str,
    batch_key: str,
    seeds: tuple[int, ...],
    fold_names: tuple[str, ...],
    strategies: tuple[str, ...],
    output_dir: Path,
    profile_settings: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    available_folds = {
        fold.name: fold for fold in ANNOTATION_REGIME_SPECS["cross_study"].cross_study_folds
    }
    for fold_name in fold_names:
        fold = available_folds[fold_name]
        for seed in seeds:
            split_plan = build_cross_study_split(
                adata,
                label_key=label_key,
                batch_key=batch_key,
                fold=fold,
                seed=seed,
            )
            for model_name in strategies:
                rows.append(
                    _run_single_benchmark(
                        dataset_name=dataset_name,
                        adata=adata,
                        label_key=label_key,
                        batch_key=batch_key,
                        regime="cross_study",
                        split_plan=split_plan,
                        model_name=model_name,
                        seed=seed,
                        profile_settings=profile_settings,
                        output_dir=output_dir,
                    )
                )
    return rows


def _ordered_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(rows)
    return frame.sort_values(
        ["dataset", "regime", "cross_study_fold", "label_fraction", "model", "seed"],
        key=lambda column: column.map(_strategy_sort_value) if column.name == "model" else column,
        kind="mergesort",
    ).reset_index(drop=True)


def _save_performance_figure(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    from matplotlib import pyplot as plt

    if frame.empty:
        summary = pd.DataFrame(
            columns=["dataset", "model", "strategy", "macro_f1", "balanced_accuracy"]
        )
    else:
        summary = (
            frame.groupby(["dataset", "model", "strategy"], as_index=False)[
                ["macro_f1", "balanced_accuracy"]
            ]
            .mean(numeric_only=True)
            .sort_values(
                ["dataset", "model"],
                key=lambda column: (
                    column.map(_strategy_sort_value) if column.name == "model" else column
                ),
            )
        )
    summary.to_csv(output_dir / "annotation_performance.csv", index=False)

    datasets = list(summary["dataset"].drop_duplicates()) if not summary.empty else []
    figure, axes = plt.subplots(
        nrows=max(1, len(datasets)),
        ncols=1,
        figsize=(12, max(4, 4 * max(1, len(datasets)))),
        squeeze=False,
    )
    if not datasets:
        axes[0, 0].text(0.5, 0.5, "No full-label rows available", ha="center", va="center")
        axes[0, 0].axis("off")
    else:
        for axis, dataset_name in zip(axes[:, 0], datasets, strict=False):
            dataset_frame = summary[summary["dataset"] == dataset_name].copy()
            dataset_frame = dataset_frame.sort_values(
                "model",
                key=lambda column: column.map(_strategy_sort_value),
            )
            x_positions = np.arange(len(dataset_frame))
            axis.bar(x_positions - 0.2, dataset_frame["macro_f1"], width=0.4, label="macro_f1")
            axis.bar(
                x_positions + 0.2,
                dataset_frame["balanced_accuracy"],
                width=0.4,
                label="balanced_accuracy",
            )
            axis.set_title(dataset_name)
            axis.set_ylim(0.0, 1.0)
            axis.set_xticks(x_positions)
            axis.set_xticklabels(dataset_frame["strategy"], rotation=35, ha="right")
            axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "annotation_performance.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return summary


def _save_low_label_figure(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    from matplotlib import pyplot as plt

    if frame.empty:
        summary = pd.DataFrame(
            columns=[
                "dataset",
                "label_fraction",
                "model",
                "strategy",
                "macro_f1",
                "balanced_accuracy",
            ]
        )
    else:
        summary = (
            frame.groupby(["dataset", "label_fraction", "model", "strategy"], as_index=False)[
                ["macro_f1", "balanced_accuracy"]
            ]
            .mean(numeric_only=True)
            .sort_values(
                ["dataset", "label_fraction", "model"],
                key=lambda column: (
                    column.map(_strategy_sort_value) if column.name == "model" else column
                ),
            )
        )
    summary.to_csv(output_dir / "annotation_low_label_curves.csv", index=False)

    datasets = list(summary["dataset"].drop_duplicates()) if not summary.empty else []
    figure, axes = plt.subplots(
        nrows=max(1, len(datasets)),
        ncols=1,
        figsize=(12, max(4, 4 * max(1, len(datasets)))),
        squeeze=False,
    )
    if not datasets:
        axes[0, 0].text(0.5, 0.5, "No low-label rows available", ha="center", va="center")
        axes[0, 0].axis("off")
    else:
        for axis, dataset_name in zip(axes[:, 0], datasets, strict=False):
            dataset_frame = summary[summary["dataset"] == dataset_name]
            for model_name in dataset_frame["model"].drop_duplicates():
                model_frame = dataset_frame[dataset_frame["model"] == model_name].sort_values(
                    "label_fraction"
                )
                axis.plot(
                    model_frame["label_fraction"],
                    model_frame["macro_f1"],
                    marker="o",
                    label=MODEL_DISPLAY_NAMES.get(model_name, model_name),
                )
            axis.set_title(dataset_name)
            axis.set_xlabel("label fraction")
            axis.set_ylabel("macro_f1")
            axis.set_ylim(0.0, 1.0)
            axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "annotation_low_label_curves.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return summary


def _save_cross_study_figure(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    from matplotlib import pyplot as plt

    if frame.empty:
        summary = pd.DataFrame(
            columns=["cross_study_fold", "model", "strategy", "macro_f1", "balanced_accuracy"]
        )
    else:
        summary = (
            frame.groupby(["cross_study_fold", "model", "strategy"], as_index=False)[
                ["macro_f1", "balanced_accuracy"]
            ]
            .mean(numeric_only=True)
            .sort_values(
                ["cross_study_fold", "model"],
                key=lambda column: (
                    column.map(_strategy_sort_value) if column.name == "model" else column
                ),
            )
        )
    summary.to_csv(output_dir / "annotation_cross_study.csv", index=False)

    folds = list(summary["cross_study_fold"].drop_duplicates()) if not summary.empty else []
    figure, axes = plt.subplots(
        nrows=max(1, len(folds)),
        ncols=1,
        figsize=(12, max(4, 4 * max(1, len(folds)))),
        squeeze=False,
    )
    if not folds:
        axes[0, 0].text(0.5, 0.5, "No cross-study rows available", ha="center", va="center")
        axes[0, 0].axis("off")
    else:
        for axis, fold_name in zip(axes[:, 0], folds, strict=False):
            fold_frame = summary[summary["cross_study_fold"] == fold_name].copy()
            fold_frame = fold_frame.sort_values(
                "model",
                key=lambda column: column.map(_strategy_sort_value),
            )
            x_positions = np.arange(len(fold_frame))
            axis.bar(x_positions - 0.2, fold_frame["macro_f1"], width=0.4, label="macro_f1")
            axis.bar(
                x_positions + 0.2,
                fold_frame["balanced_accuracy"],
                width=0.4,
                label="balanced_accuracy",
            )
            axis.set_title(fold_name)
            axis.set_ylim(0.0, 1.0)
            axis.set_xticks(x_positions)
            axis.set_xticklabels(fold_frame["strategy"], rotation=35, ha="right")
            axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "annotation_cross_study.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return summary


def _save_pareto_figure(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    from matplotlib import pyplot as plt

    if frame.empty:
        summary = pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "strategy",
                "macro_f1",
                "trainable_parameters",
                "runtime_sec",
            ]
        )
    else:
        summary = (
            frame.groupby(["dataset", "model", "strategy"], as_index=False)[
                ["macro_f1", "trainable_parameters", "runtime_sec"]
            ]
            .mean(numeric_only=True)
            .sort_values(
                ["dataset", "model"],
                key=lambda column: (
                    column.map(_strategy_sort_value) if column.name == "model" else column
                ),
            )
        )
    summary.to_csv(output_dir / "annotation_pareto.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    if summary.empty:
        for axis in axes:
            axis.text(0.5, 0.5, "No full-label rows available", ha="center", va="center")
            axis.axis("off")
    else:
        for dataset_name in summary["dataset"].drop_duplicates():
            dataset_frame = summary[summary["dataset"] == dataset_name]
            axes[0].scatter(
                dataset_frame["trainable_parameters"],
                dataset_frame["macro_f1"],
                label=dataset_name,
            )
            axes[1].scatter(
                dataset_frame["runtime_sec"],
                dataset_frame["macro_f1"],
                label=dataset_name,
            )
        axes[0].set_xlabel("trainable_parameters")
        axes[0].set_ylabel("macro_f1")
        axes[0].set_title("Efficiency-performance Pareto")
        axes[1].set_xlabel("runtime_sec")
        axes[1].set_ylabel("macro_f1")
        axes[1].set_title("Runtime-performance Pareto")
        axes[0].legend(frameon=False)
        axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "annotation_pareto.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return summary


def _ranking_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["macro_f1", "balanced_accuracy", "accuracy", "trainable_parameters", "runtime_sec"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    )


def _write_summary(
    *,
    full_label_frame: pd.DataFrame,
    low_label_frame: pd.DataFrame,
    cross_study_frame: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    full_bests: dict[str, dict[str, Any]] = {}
    if not full_label_frame.empty:
        for dataset_name in full_label_frame["dataset"].drop_duplicates():
            dataset_frame = _ranking_columns(
                full_label_frame[full_label_frame["dataset"] == dataset_name]
            )
            if dataset_frame.empty:
                continue
            best = dataset_frame.iloc[0]
            full_bests[dataset_name] = {
                "model": str(best["model"]),
                "strategy": str(best["strategy"]),
                "macro_f1": float(best["macro_f1"]),
                "balanced_accuracy": float(best["balanced_accuracy"]),
            }

    datasets = sorted(
        set(full_label_frame.get("dataset", pd.Series(dtype=object)))
        .union(low_label_frame.get("dataset", pd.Series(dtype=object)))
        .union(cross_study_frame.get("dataset", pd.Series(dtype=object)))
    )
    summary = {
        "task": ANNOTATION_TASK_SPEC.task_name,
        "datasets": datasets,
        "strategies": list(BENCHMARK_MODELS),
        "full_label_rows": int(len(full_label_frame)),
        "low_label_rows": int(len(low_label_frame)),
        "cross_study_rows": int(len(cross_study_frame)),
        "best_full_label_by_dataset": full_bests,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Annotation pillar benchmark summary",
        "",
        f"- Full-label rows: `{summary['full_label_rows']}`",
        f"- Low-label rows: `{summary['low_label_rows']}`",
        f"- Cross-study rows: `{summary['cross_study_rows']}`",
        "",
        "## Best full-label strategies",
        "",
    ]
    if full_bests:
        for dataset_name, best in full_bests.items():
            lines.append(
                "- "
                f"`{dataset_name}`: `{best['strategy']}` "
                f"(macro_f1 `{best['macro_f1']:.4f}`, "
                f"balanced_accuracy `{best['balanced_accuracy']:.4f}`)"
            )
    else:
        lines.append("- No full-label rows were produced.")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _sync_tutorial_bundle(output_dir: Path) -> None:
    tutorial_dir = output_dir / "tutorial"
    tutorial_dir.mkdir(parents=True, exist_ok=True)
    source_root = ROOT / "artifacts" / "scgpt_human_pancreas_annotation"
    copied: list[str] = []
    for relative_path in (
        "report.md",
        "report.csv",
        "strategy_metrics.csv",
        "best_strategy_confusion_matrix.png",
        "frozen_embedding_umap.png",
        "best_strategy_embedding_umap.png",
        "best_model/manifest.json",
        "best_model/model_state.pt",
    ):
        source_path = source_root / relative_path
        destination_path = tutorial_dir / relative_path
        if not source_path.exists():
            continue
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(source_path.read_bytes())
        copied.append(relative_path)
    manifest = {
        "main_tutorial_notebook": "examples/scgpt_human_pancreas_annotation.ipynb",
        "source_artifact_dir": str(source_root),
        "copied_files": copied,
    }
    (tutorial_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _load_annotation_dataset(dataset_name: str, *, profile: str) -> tuple[AnnData, str, str | None]:
    adata, spec = _load_dataset(dataset_name, profile=profile)
    return adata, spec.label_key, spec.batch_key


def run_annotation_benchmark(
    *,
    datasets: tuple[str, ...],
    regimes: tuple[str, ...],
    profile: str,
    strategies: tuple[str, ...],
    output_dir: Path,
    seeds: tuple[int, ...],
    label_fractions: tuple[float, ...],
    cross_study_folds: tuple[str, ...],
) -> dict[str, Path]:
    dataset_profile, profile_settings = _benchmark_profile(profile)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for dataset_name in datasets:
        dataset_spec = ANNOTATION_DATASET_SPECS[dataset_name]
        adata, label_key, batch_key = _load_annotation_dataset(
            dataset_name,
            profile=dataset_profile,
        )
        if "full_label" in regimes and "full_label" in dataset_spec.regimes:
            rows.extend(
                _collect_full_label_rows(
                    dataset_name=dataset_name,
                    adata=adata,
                    label_key=label_key,
                    batch_key=batch_key,
                    seeds=seeds,
                    strategies=strategies,
                    output_dir=output_dir,
                    profile_settings=profile_settings,
                )
            )
        if "low_label" in regimes and "low_label" in dataset_spec.regimes:
            rows.extend(
                _collect_low_label_rows(
                    dataset_name=dataset_name,
                    adata=adata,
                    label_key=label_key,
                    batch_key=batch_key,
                    seeds=seeds,
                    label_fractions=label_fractions,
                    strategies=strategies,
                    output_dir=output_dir,
                    profile_settings=profile_settings,
                )
            )
        if (
            "cross_study" in regimes
            and "cross_study" in dataset_spec.regimes
            and batch_key is not None
        ):
            rows.extend(
                _collect_cross_study_rows(
                    dataset_name=dataset_name,
                    adata=adata,
                    label_key=label_key,
                    batch_key=batch_key,
                    seeds=seeds,
                    fold_names=cross_study_folds,
                    strategies=strategies,
                    output_dir=output_dir,
                    profile_settings=profile_settings,
                )
            )

    frame = _ordered_frame(rows)
    if frame.empty:
        full_label_frame = pd.DataFrame()
        low_label_frame = pd.DataFrame()
        cross_study_frame = pd.DataFrame()
    else:
        full_label_frame = frame[frame["regime"] == "full_label"].copy()
        low_label_frame = frame[frame["regime"] == "low_label"].copy()
        cross_study_frame = frame[frame["regime"] == "cross_study"].copy()

    full_dir = output_dir / "full_label"
    low_dir = output_dir / "low_label"
    cross_dir = output_dir / "cross_study"
    figures_dir = output_dir / "figures"
    tutorial_dir = output_dir / "tutorial"
    for directory in (full_dir, low_dir, cross_dir, figures_dir, tutorial_dir):
        directory.mkdir(parents=True, exist_ok=True)

    full_label_frame.to_csv(full_dir / "strategy_metrics.csv", index=False)
    low_label_frame.to_csv(low_dir / "strategy_metrics.csv", index=False)
    cross_study_frame.to_csv(cross_dir / "strategy_metrics.csv", index=False)

    _save_performance_figure(full_label_frame, figures_dir)
    _save_low_label_figure(low_label_frame, figures_dir)
    _save_cross_study_figure(cross_study_frame, figures_dir)
    _save_pareto_figure(full_label_frame, figures_dir)
    _write_summary(
        full_label_frame=full_label_frame,
        low_label_frame=low_label_frame,
        cross_study_frame=cross_study_frame,
        output_dir=output_dir,
    )
    _sync_tutorial_bundle(output_dir)
    return {
        "output_dir": output_dir,
        "full_label_dir": full_dir,
        "low_label_dir": low_dir,
        "cross_study_dir": cross_dir,
        "figures_dir": figures_dir,
        "tutorial_dir": tutorial_dir,
    }


def main() -> None:
    args = parse_args()
    outputs = run_annotation_benchmark(
        datasets=tuple(args.dataset),
        regimes=tuple(args.regime),
        profile=str(args.profile),
        strategies=tuple(args.strategies),
        output_dir=args.output_dir,
        seeds=tuple(int(seed) for seed in args.seeds),
        label_fractions=tuple(float(value) for value in args.label_fractions),
        cross_study_folds=tuple(str(value) for value in args.cross_study_folds),
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
