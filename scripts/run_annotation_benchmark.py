"""Run the Milestone 1 annotation-pillar benchmark matrix."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
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
    _prepare_annotation_benchmark_adata,
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
    count_total_parameters,
    count_trainable_parameters,
    load_scgpt_annotation_model,
    load_scgpt_checkpoint_state_dict,
    load_scgpt_model,
    prepare_scgpt_data,
)
from scdlkit.foundation.data import ScGPTPreparedData  # noqa: E402
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
    fold: int | None = None
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
        "--n-folds",
        type=int,
        default=ANNOTATION_REGIME_SPECS["full_label"].n_folds,
        help="Number of cross-validation folds (default 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=ANNOTATION_REGIME_SPECS["full_label"].seed,
        help="Base random seed for reproducibility (default 42).",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed per-run rows already present under the output directory.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip execution and rebuild the top-level bundle from existing per-run rows.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate figures from the saved all_results.csv (no row walk or model loading).",
    )
    return parser.parse_args()


def _benchmark_profile(profile: str) -> tuple[str, dict[str, int]]:
    if profile == "quickstart":
        settings = dict(_foundation_annotation_profile("ci"))
        settings["token_max_length"] = 192
        return "quickstart", settings
    settings = dict(_foundation_annotation_profile("full"))
    # Override quality-suite data limits for publication benchmarking.
    # The quality-suite "full" profile uses max_cells=64 for gate checks;
    # a real benchmark needs the full dataset to produce meaningful results.
    settings["max_cells"] = 999_999  # effectively no subsampling
    settings["max_genes"] = 999_999  # effectively no gene filtering
    settings["min_gene_overlap"] = 50
    settings["token_max_length"] = 512
    return "full", settings


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


def build_kfold_splits(
    adata: AnnData,
    *,
    label_key: str,
    n_folds: int = 5,
    seed: int = 42,
    val_size: float = 0.15,
) -> list[SplitPlan]:
    """Build stratified k-fold cross-validation splits.

    Each fold holds out ~1/k of the data as the test set, then splits the
    remaining data into train and validation (val_size fraction of remaining).
    """
    from sklearn.model_selection import StratifiedKFold

    labels, _ = _encode_labels(adata.obs[label_key])
    indices = np.arange(adata.n_obs, dtype=np.int64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits: list[SplitPlan] = []
    for fold_index, (train_val_idx, test_idx) in enumerate(skf.split(indices, labels)):
        train_val_idx = np.sort(np.asarray(train_val_idx, dtype=np.int64))
        test_idx = np.sort(np.asarray(test_idx, dtype=np.int64))
        train_val_labels = labels[train_val_idx]
        train_idx, val_idx = _safe_split(
            train_val_idx,
            train_val_labels,
            train_size=1.0 - val_size,
            seed=seed + fold_index,
        )
        splits.append(
            SplitPlan(
                train_indices=train_idx,
                val_indices=val_idx,
                test_indices=test_idx,
                fold=fold_index,
            )
        )
    return splits


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
        fold=full_split.fold,
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


def _expected_row_count(
    *,
    datasets: tuple[str, ...],
    regimes: tuple[str, ...],
    strategies: tuple[str, ...],
    n_folds: int,
    label_fractions: tuple[float, ...],
    cross_study_folds: tuple[str, ...],
) -> int:
    total = 0
    for dataset_name in datasets:
        spec = ANNOTATION_DATASET_SPECS[dataset_name]
        if "full_label" in regimes and "full_label" in spec.regimes:
            total += n_folds * len(strategies)
        if "low_label" in regimes and "low_label" in spec.regimes:
            total += n_folds * len(label_fractions) * len(strategies)
        if "cross_study" in regimes and "cross_study" in spec.regimes and spec.batch_key is not None:
            total += len(cross_study_folds) * len(strategies)
    return total


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
    fold: int,
    label_fraction: float | None,
    cross_study_fold: str | None,
) -> Path:
    parts = [output_dir, "runs", regime, dataset_name, model_name, f"fold_{fold}"]
    if label_fraction is not None:
        parts.append(f"fraction_{label_fraction:.2f}".replace(".", "p"))
    if cross_study_fold is not None:
        parts.append(f"fold_{cross_study_fold}")
    path = Path(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_row_path(run_dir: Path) -> Path:
    return run_dir / "row.json"


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_run_row(run_dir: Path, row: dict[str, Any]) -> None:
    payload = {key: _jsonable_value(value) for key, value in row.items()}
    _run_row_path(run_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_run_row(run_dir: Path) -> dict[str, Any] | None:
    row_path = _run_row_path(run_dir)
    if not row_path.exists():
        return None
    payload = json.loads(row_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def _subset_dataset(dataset: Dataset[dict[str, Any]], indices: np.ndarray) -> Subset[Any]:
    return Subset(dataset, [int(index) for index in np.asarray(indices, dtype=np.int64)])


def _truncate_prepared_data(
    prepared: ScGPTPreparedData,
    *,
    max_token_length: int,
) -> ScGPTPreparedData:
    dataset = prepared.dataset
    gene_ids = getattr(dataset, "gene_ids", None)
    values = getattr(dataset, "values", None)
    padding_mask = getattr(dataset, "padding_mask", None)
    labels = getattr(dataset, "labels", None)
    if not isinstance(gene_ids, torch.Tensor):
        return prepared
    if int(gene_ids.shape[1]) <= max_token_length:
        return prepared
    if not isinstance(values, torch.Tensor) or not isinstance(padding_mask, torch.Tensor):
        return prepared

    class _TruncatedTensorDataset(Dataset[dict[str, torch.Tensor]]):
        def __init__(
            self,
            *,
            gene_ids: torch.Tensor,
            values: torch.Tensor,
            padding_mask: torch.Tensor,
            labels: torch.Tensor | None,
        ) -> None:
            self.gene_ids = gene_ids
            self.values = values
            self.padding_mask = padding_mask
            self.labels = labels

        def __len__(self) -> int:
            return int(self.gene_ids.shape[0])

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            sample: dict[str, torch.Tensor] = {
                "gene_ids": self.gene_ids[index],
                "values": self.values[index],
                "padding_mask": self.padding_mask[index],
            }
            if self.labels is not None:
                sample["y"] = self.labels[index]
            return sample

    truncated_dataset = _TruncatedTensorDataset(
        gene_ids=gene_ids[:, :max_token_length].clone(),
        values=values[:, :max_token_length].clone(),
        padding_mask=padding_mask[:, :max_token_length].clone(),
        labels=labels.clone() if isinstance(labels, torch.Tensor) else None,
    )
    return replace(prepared, dataset=truncated_dataset)


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
    total_parameters: int,
    runtime_sec: float,
    checkpoint_size_bytes: int | None,
    split_plan: SplitPlan,
) -> dict[str, Any]:
    return {
        "Dataset": dataset_name,
        "Strategy": MODEL_DISPLAY_NAMES[model_name],
        "Trainable parameters": trainable_parameters,
        "Total parameters": total_parameters,
        "Runtime (sec)": runtime_sec,
        "Checkpoint size (bytes)": (
            checkpoint_size_bytes if checkpoint_size_bytes is not None else 0
        ),
        "Macro F1": _scalar_value(metrics, "macro_f1"),
        "Weighted F1": _scalar_value(metrics, "weighted_f1"),
        "Accuracy": _scalar_value(metrics, "accuracy"),
        "Balanced accuracy": _scalar_value(metrics, "balanced_accuracy"),
        "Macro precision": _scalar_value(metrics, "macro_precision"),
        "Macro recall": _scalar_value(metrics, "macro_recall"),
        "Cohen kappa": _scalar_value(metrics, "cohen_kappa"),
        "MCC": _scalar_value(metrics, "mcc"),
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
    fold: int,
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
                "total_parameters": 0,
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
            total_parameters=0,
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
        "fold": fold,
        "label_fraction": split_plan.label_fraction,
        "cross_study_fold": split_plan.cross_study_fold,
        "runtime_sec": runtime_sec,
        "peak_memory_mb": _process_peak_memory_mb(),
        "trainable_parameters": 0,
        "total_parameters": 0,
        "checkpoint_size_bytes": 0,
        "artifact_dir": str(output_dir),
        "best_model_artifact": None,
        "batch_metrics_artifact": str(output_dir / "batch_metrics.csv"),
        "confusion_matrix_artifact": str(output_dir / "confusion_matrix.png"),
        "latent_umap_artifact": str(output_dir / "latent_umap.png"),
        **{key: value for key, value in metrics.items() if isinstance(value, (int, float, list))},
        **_batch_metric_summary(batch_metrics),
    }


def _run_scgpt_strategy(
    *,
    adata: AnnData,
    prepared: ScGPTPreparedData,
    label_key: str,
    batch_key: str | None,
    split_plan: SplitPlan,
    dataset_name: str,
    model_name: str,
    fold: int,
    seed: int,
    profile_settings: dict[str, int],
    output_dir: Path,
    preloaded_state_dict: dict | None = None,
) -> dict[str, Any]:
    train_dataset = _subset_dataset(prepared.dataset, split_plan.train_indices)
    val_dataset = _subset_dataset(prepared.dataset, split_plan.val_indices)
    test_dataset = _subset_dataset(prepared.dataset, split_plan.test_indices)
    test_adata = adata[np.asarray(split_plan.test_indices, dtype=np.int64)].copy()
    label_categories = list(prepared.label_categories or [])

    started_at = perf_counter()
    if model_name == "scgpt_frozen_probe":
        model = load_scgpt_model(
            "whole-human", device="auto", preloaded_state_dict=preloaded_state_dict
        )
        total_parameters = count_total_parameters(model)
        trainer = Trainer(
            model=model,
            task="representation",
            batch_size=prepared.batch_size,
            device="auto",
            epochs=1,
            mixed_precision=torch.cuda.is_available(),
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
            preloaded_state_dict=preloaded_state_dict,
        )
        total_parameters = count_total_parameters(model)
        trainer = Trainer(
            model=model,
            task="classification",
            batch_size=prepared.batch_size,
            epochs=profile_settings[epoch_key],
            lr=lr,
            device="auto",
            mixed_precision=torch.cuda.is_available(),
            early_stopping_patience=5,
            lr_schedule_gamma=0.9,
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
                "total_parameters": total_parameters,
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
            total_parameters=total_parameters,
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
        "fold": fold,
        "label_fraction": split_plan.label_fraction,
        "cross_study_fold": split_plan.cross_study_fold,
        "runtime_sec": runtime_sec,
        "peak_memory_mb": _process_peak_memory_mb(),
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "artifact_dir": str(output_dir),
        "best_model_artifact": best_model_artifact,
        "batch_metrics_artifact": str(output_dir / "batch_metrics.csv"),
        "confusion_matrix_artifact": str(output_dir / "confusion_matrix.png"),
        "latent_umap_artifact": str(output_dir / "latent_umap.png"),
        **{key: value for key, value in metrics.items() if isinstance(value, (int, float, list))},
        **_batch_metric_summary(batch_metrics),
    }


def _run_single_benchmark(
    *,
    dataset_name: str,
    adata: AnnData,
    prepared: ScGPTPreparedData | None,
    label_key: str,
    batch_key: str | None,
    regime: str,
    split_plan: SplitPlan,
    model_name: str,
    fold: int,
    seed: int,
    profile_settings: dict[str, int],
    output_dir: Path,
    resume: bool,
    preloaded_state_dict: dict | None = None,
) -> dict[str, Any]:
    run_dir = _run_output_dir(
        output_dir,
        regime=regime,
        dataset_name=dataset_name,
        model_name=model_name,
        fold=fold,
        label_fraction=split_plan.label_fraction,
        cross_study_fold=split_plan.cross_study_fold,
    )
    if resume:
        existing_row = _load_run_row(run_dir)
        if existing_row is not None:
            return existing_row
    # Derive a per-fold seed for reproducible model init and training
    fold_seed = seed + fold
    if model_name == "pca_logistic_annotation":
        row = _run_pca_logistic_strategy(
            adata=adata,
            label_key=label_key,
            batch_key=batch_key,
            split_plan=split_plan,
            dataset_name=dataset_name,
            fold=fold,
            seed=fold_seed,
            output_dir=run_dir,
        )
    else:
        if prepared is None:
            prepared = prepare_scgpt_data(
                adata,
                checkpoint="whole-human",
                label_key=label_key,
                batch_size=64,
                use_raw=True,
                min_gene_overlap=profile_settings["min_gene_overlap"],
            )
            prepared = _truncate_prepared_data(
                prepared,
                max_token_length=profile_settings["token_max_length"],
            )
        row = _run_scgpt_strategy(
            adata=adata,
            prepared=prepared,
            label_key=label_key,
            batch_key=batch_key,
            split_plan=split_plan,
            dataset_name=dataset_name,
            model_name=model_name,
            fold=fold,
            seed=fold_seed,
            profile_settings=profile_settings,
            output_dir=run_dir,
            preloaded_state_dict=preloaded_state_dict,
        )
    row["regime"] = regime
    _write_run_row(run_dir, row)
    return row


def _prepare_scgpt_data_for_dataset(
    dataset_name: str,
    benchmark_adata: AnnData,
    *,
    label_key: str,
    profile_settings: dict[str, int],
    pancreas_prepared: ScGPTPreparedData | None,
) -> ScGPTPreparedData | None:
    """Return prepared scGPT data for a dataset, computing it once if needed."""
    if dataset_name == "openproblems_human_pancreas" and pancreas_prepared is not None:
        return pancreas_prepared
    dataset_prepared = prepare_scgpt_data(
        benchmark_adata,
        checkpoint="whole-human",
        label_key=label_key,
        batch_size=64,
        use_raw=True,
        min_gene_overlap=profile_settings["min_gene_overlap"],
    )
    return _truncate_prepared_data(
        dataset_prepared,
        max_token_length=profile_settings["token_max_length"],
    )


def _collect_full_label_rows(
    *,
    dataset_name: str,
    adata: AnnData,
    prepared: ScGPTPreparedData | None,
    label_key: str,
    batch_key: str | None,
    n_folds: int,
    seed: int,
    strategies: tuple[str, ...],
    output_dir: Path,
    profile_settings: dict[str, int],
    profile: str,
    resume: bool,
    preloaded_state_dict: dict | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scgpt_strategies = [s for s in strategies if s != "pca_logistic_annotation"]
    benchmark_adata = _prepare_annotation_benchmark_adata(
        dataset_name,
        adata,
        label_key=label_key,
        seed=seed,
        profile=profile,
    )
    benchmark_prepared = (
        _prepare_scgpt_data_for_dataset(
            dataset_name,
            benchmark_adata,
            label_key=label_key,
            profile_settings=profile_settings,
            pancreas_prepared=prepared,
        )
        if scgpt_strategies
        else None
    )
    fold_splits = build_kfold_splits(
        benchmark_adata,
        label_key=label_key,
        n_folds=n_folds,
        seed=seed,
    )
    for split_plan in fold_splits:
        for model_name in strategies:
            rows.append(
                _run_single_benchmark(
                    dataset_name=dataset_name,
                    adata=benchmark_adata,
                    prepared=benchmark_prepared,
                    label_key=label_key,
                    batch_key=batch_key,
                    regime="full_label",
                    split_plan=split_plan,
                    model_name=model_name,
                    fold=split_plan.fold,
                    seed=seed,
                    profile_settings=profile_settings,
                    output_dir=output_dir,
                    resume=resume,
                    preloaded_state_dict=preloaded_state_dict,
                )
            )
    return rows


def _collect_low_label_rows(
    *,
    dataset_name: str,
    adata: AnnData,
    prepared: ScGPTPreparedData | None,
    label_key: str,
    batch_key: str | None,
    n_folds: int,
    seed: int,
    label_fractions: tuple[float, ...],
    strategies: tuple[str, ...],
    output_dir: Path,
    profile_settings: dict[str, int],
    profile: str,
    resume: bool,
    preloaded_state_dict: dict | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scgpt_strategies = [s for s in strategies if s != "pca_logistic_annotation"]
    benchmark_adata = _prepare_annotation_benchmark_adata(
        dataset_name,
        adata,
        label_key=label_key,
        seed=seed,
        profile=profile,
    )
    benchmark_prepared = (
        _prepare_scgpt_data_for_dataset(
            dataset_name,
            benchmark_adata,
            label_key=label_key,
            profile_settings=profile_settings,
            pancreas_prepared=prepared,
        )
        if scgpt_strategies
        else None
    )
    fold_splits = build_kfold_splits(
        benchmark_adata,
        label_key=label_key,
        n_folds=n_folds,
        seed=seed,
    )
    for full_split in fold_splits:
        for label_fraction in label_fractions:
            split_plan = build_low_label_split(
                full_split,
                benchmark_adata,
                label_key=label_key,
                label_fraction=label_fraction,
                seed=seed + full_split.fold,
            )
            for model_name in strategies:
                rows.append(
                    _run_single_benchmark(
                        dataset_name=dataset_name,
                        adata=benchmark_adata,
                        prepared=benchmark_prepared,
                        label_key=label_key,
                        batch_key=batch_key,
                        regime="low_label",
                        split_plan=split_plan,
                        model_name=model_name,
                        fold=split_plan.fold,
                        seed=seed,
                        profile_settings=profile_settings,
                        output_dir=output_dir,
                        resume=resume,
                        preloaded_state_dict=preloaded_state_dict,
                    )
                )
    return rows


def _collect_cross_study_rows(
    *,
    dataset_name: str,
    adata: AnnData,
    prepared: ScGPTPreparedData | None,
    label_key: str,
    batch_key: str,
    seed: int,
    fold_names: tuple[str, ...],
    strategies: tuple[str, ...],
    output_dir: Path,
    profile_settings: dict[str, int],
    resume: bool,
    preloaded_state_dict: dict | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    available_folds = {
        fold.name: fold for fold in ANNOTATION_REGIME_SPECS["cross_study"].cross_study_folds
    }
    for fold_index, fold_name in enumerate(fold_names):
        fold = available_folds[fold_name]
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
                    prepared=prepared,
                    label_key=label_key,
                    batch_key=batch_key,
                    regime="cross_study",
                    split_plan=split_plan,
                    model_name=model_name,
                    fold=fold_index,
                    seed=seed,
                    profile_settings=profile_settings,
                    output_dir=output_dir,
                    resume=resume,
                    preloaded_state_dict=preloaded_state_dict,
                )
            )
    return rows


def _collect_existing_rows(output_dir: Path) -> list[dict[str, Any]]:
    runs_root = output_dir / "runs"
    if not runs_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for row_path in sorted(runs_root.rglob("row.json")):
        payload = json.loads(row_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _row_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    """Build a stable run identity for merging in-memory and on-disk rows."""
    return (
        row.get("dataset"),
        row.get("regime"),
        row.get("model"),
        row.get("fold"),
        row.get("label_fraction"),
        row.get("cross_study_fold"),
    )


def _merge_rows(
    in_memory_rows: list[dict[str, Any]],
    existing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge rows collected in the current process with saved row.json outputs.

    Saved rows are treated as authoritative when the same run appears in both
    lists, which matches the normal benchmark path where each run persists a
    `row.json` file before aggregation.
    """
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in in_memory_rows:
        merged[_row_identity(row)] = row
    for row in existing_rows:
        merged[_row_identity(row)] = row
    return list(merged.values())


def _ordered_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(rows)
    sort_cols = ["dataset", "regime", "cross_study_fold", "label_fraction", "model"]
    if "fold" in frame.columns:
        sort_cols.append("fold")
    elif "seed" in frame.columns:
        sort_cols.append("seed")
    return frame.sort_values(
        sort_cols,
        key=lambda column: column.map(_strategy_sort_value) if column.name == "model" else column,
        kind="mergesort",
    ).reset_index(drop=True)


def _publication_style() -> None:
    """Apply Nature-style matplotlib defaults for publication figures."""
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )


STRATEGY_PALETTE = {
    "PCA + logistic regression": "#999999",
    "Frozen scGPT probe": "#4E79A7",
    "scGPT head-only tuning": "#F28E2B",
    "scGPT full fine-tuning": "#E15759",
    "scGPT LoRA": "#76B7B2",
    "scGPT adapters": "#59A14F",
    "scGPT prefix tuning": "#EDC948",
    "scGPT IA3": "#B07AA1",
}


def _save_performance_figure(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    from matplotlib import pyplot as plt

    _publication_style()
    metric_cols = ["macro_f1", "weighted_f1", "balanced_accuracy"]
    metric_labels = ["Macro F1", "Weighted F1", "Balanced Acc."]

    if frame.empty:
        summary = pd.DataFrame(
            columns=["dataset", "model", "strategy"] + metric_cols
        )
    else:
        agg_metrics = [c for c in metric_cols if c in frame.columns] + [
            c for c in ["accuracy", "cohen_kappa", "mcc", "auroc_ovr"] if c in frame.columns
        ]
        group_mean = (
            frame.groupby(["dataset", "model", "strategy"], as_index=False)[agg_metrics]
            .mean(numeric_only=True)
        )
        group_std = (
            frame.groupby(["dataset", "model", "strategy"], as_index=False)[agg_metrics]
            .std(numeric_only=True)
            .rename(columns={c: f"{c}_std" for c in agg_metrics})
        )
        summary = group_mean.merge(
            group_std, on=["dataset", "model", "strategy"], how="left"
        ).fillna(0).sort_values(
            ["dataset", "model"],
            key=lambda column: (
                column.map(_strategy_sort_value) if column.name == "model" else column
            ),
        )
    summary.to_csv(output_dir / "annotation_performance.csv", index=False)

    datasets = list(summary["dataset"].drop_duplicates()) if not summary.empty else []
    n_datasets = max(1, len(datasets))
    figure, axes = plt.subplots(
        nrows=n_datasets,
        ncols=1,
        figsize=(10, 3.5 * n_datasets),
        squeeze=False,
    )
    metric_colors = ["#4E79A7", "#59A14F", "#F28E2B"]
    if not datasets:
        axes[0, 0].text(0.5, 0.5, "No full-label rows available", ha="center", va="center")
        axes[0, 0].axis("off")
    else:
        available_metrics = [c for c in metric_cols if c in summary.columns]
        available_labels = [metric_labels[metric_cols.index(c)] for c in available_metrics]
        n_metrics = len(available_metrics)
        bar_width = 0.8 / max(n_metrics, 1)
        for axis, dataset_name in zip(axes[:, 0], datasets, strict=False):
            df = summary[summary["dataset"] == dataset_name].copy()
            df = df.sort_values("model", key=lambda c: c.map(_strategy_sort_value))
            x = np.arange(len(df))
            for i, (col, label) in enumerate(zip(available_metrics, available_labels)):
                offset = (i - (n_metrics - 1) / 2) * bar_width
                std_col = f"{col}_std"
                yerr = df[std_col].values if std_col in df.columns else None
                axis.bar(
                    x + offset,
                    df[col],
                    width=bar_width * 0.9,
                    label=label,
                    yerr=yerr,
                    capsize=2,
                    error_kw={"linewidth": 0.8},
                    color=metric_colors[i % len(metric_colors)],
                    edgecolor="white",
                    linewidth=0.5,
                )
            dataset_label = dataset_name.replace("_", " ").title()
            axis.set_title(dataset_label, fontweight="bold")
            axis.set_ylabel("Score")
            axis.set_ylim(0.0, 1.05)
            axis.set_xticks(x)
            axis.set_xticklabels(
                [s.replace("scGPT ", "") for s in df["strategy"]],
                rotation=30,
                ha="right",
            )
            axis.legend(frameon=False, ncol=n_metrics, loc="upper right")
            axis.axhline(y=1.0, color="#cccccc", linewidth=0.5, zorder=0)
    figure.tight_layout()
    figure.savefig(output_dir / "annotation_performance.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / "annotation_performance.svg", bbox_inches="tight")
    plt.close(figure)
    return summary


def _save_low_label_figure(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    from matplotlib import pyplot as plt

    _publication_style()

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
        group_mean = (
            frame.groupby(
                ["dataset", "label_fraction", "model", "strategy"], as_index=False
            )[["macro_f1", "balanced_accuracy"]]
            .mean(numeric_only=True)
        )
        group_std = (
            frame.groupby(
                ["dataset", "label_fraction", "model", "strategy"], as_index=False
            )[["macro_f1", "balanced_accuracy"]]
            .std(numeric_only=True)
            .rename(columns={"macro_f1": "macro_f1_std", "balanced_accuracy": "balanced_accuracy_std"})
        )
        summary = group_mean.merge(
            group_std, on=["dataset", "label_fraction", "model", "strategy"], how="left"
        ).fillna(0).sort_values(
            ["dataset", "label_fraction", "model"],
            key=lambda column: (
                column.map(_strategy_sort_value) if column.name == "model" else column
            ),
        )
    summary.to_csv(output_dir / "annotation_low_label_curves.csv", index=False)

    datasets = list(summary["dataset"].drop_duplicates()) if not summary.empty else []
    n_datasets = max(1, len(datasets))
    figure, axes = plt.subplots(
        nrows=n_datasets,
        ncols=1,
        figsize=(10, 3.5 * n_datasets),
        squeeze=False,
    )
    if not datasets:
        axes[0, 0].text(0.5, 0.5, "No low-label rows available", ha="center", va="center")
        axes[0, 0].axis("off")
    else:
        for axis, dataset_name in zip(axes[:, 0], datasets, strict=False):
            dataset_frame = summary[summary["dataset"] == dataset_name]
            for model_name in dataset_frame["model"].drop_duplicates():
                mf = dataset_frame[dataset_frame["model"] == model_name].sort_values(
                    "label_fraction"
                )
                display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
                color = STRATEGY_PALETTE.get(display, None)
                fracs = mf["label_fraction"].values
                means = mf["macro_f1"].values
                stds = mf["macro_f1_std"].values if "macro_f1_std" in mf.columns else np.zeros_like(means)
                short = display.replace("scGPT ", "").replace("PCA + logistic regression", "PCA+LR")
                axis.plot(fracs, means, marker="o", markersize=5, linewidth=1.5, label=short, color=color)
                axis.fill_between(fracs, means - stds, means + stds, alpha=0.15, color=color)
            dataset_label = dataset_name.replace("_", " ").title()
            axis.set_title(f"Label Efficiency: {dataset_label}", fontweight="bold")
            axis.set_xlabel("Labeled fraction of training data")
            axis.set_ylabel("Macro F1")
            axis.set_ylim(0.0, 1.05)
            axis.set_xlim(-0.005, max(0.12, float(dataset_frame["label_fraction"].max()) + 0.01))
            axis.legend(
                frameon=False, fontsize=6.5, loc="upper left",
                bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
            )
    figure.tight_layout()
    figure.savefig(output_dir / "annotation_low_label_curves.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / "annotation_low_label_curves.svg", bbox_inches="tight")
    plt.close(figure)
    return summary


def _save_cross_study_figure(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    import seaborn as sns
    from matplotlib import pyplot as plt

    _publication_style()

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

    if summary.empty:
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.text(0.5, 0.5, "No cross-study rows available", ha="center", va="center")
        axis.axis("off")
    else:
        pivot = summary.pivot_table(
            values="macro_f1", index="strategy", columns="cross_study_fold", aggfunc="mean"
        )
        order = [
            MODEL_DISPLAY_NAMES[m]
            for m in BENCHMARK_MODELS
            if MODEL_DISPLAY_NAMES[m] in pivot.index
        ]
        pivot = pivot.reindex(order)
        fold_labels = [f.replace("_", " ").title() for f in pivot.columns]
        strategy_labels = [s.replace("scGPT ", "") for s in pivot.index]

        figure, axis = plt.subplots(figsize=(6, max(3, 0.45 * len(pivot))))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Macro F1", "shrink": 0.8},
            ax=axis,
            xticklabels=fold_labels,
            yticklabels=strategy_labels,
        )
        axis.set_title("Cross-Study Generalization (Macro F1)", fontweight="bold")
        axis.set_xlabel("Held-out technology fold")
        axis.set_ylabel("")
        axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    figure.savefig(output_dir / "annotation_cross_study.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / "annotation_cross_study.svg", bbox_inches="tight")
    plt.close(figure)
    return summary


def _save_pareto_figure(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    _publication_style()

    if frame.empty:
        summary = pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "strategy",
                "macro_f1",
                "trainable_parameters",
                "total_parameters",
                "runtime_sec",
            ]
        )
    else:
        agg_cols = ["macro_f1", "trainable_parameters", "runtime_sec"]
        if "total_parameters" in frame.columns:
            agg_cols.append("total_parameters")
        summary = (
            frame.groupby(["dataset", "model", "strategy"], as_index=False)[agg_cols]
            .mean(numeric_only=True)
            .sort_values(
                ["dataset", "model"],
                key=lambda column: (
                    column.map(_strategy_sort_value) if column.name == "model" else column
                ),
            )
        )
        if "total_parameters" in summary.columns:
            total = summary["total_parameters"].replace(0, np.nan)
            summary["trainable_fraction"] = (
                summary["trainable_parameters"] / total
            ).fillna(0)
        else:
            summary["trainable_fraction"] = 0.0
    summary.to_csv(output_dir / "annotation_pareto.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if summary.empty:
        for axis in axes:
            axis.text(0.5, 0.5, "No full-label rows available", ha="center", va="center")
            axis.axis("off")
    else:
        datasets = list(summary["dataset"].drop_duplicates())
        markers = ["o", "s", "D", "^", "v"]
        for di, dataset_name in enumerate(datasets):
            df = summary[summary["dataset"] == dataset_name]
            marker = markers[di % len(markers)]
            for _, row in df.iterrows():
                color = STRATEGY_PALETTE.get(row["strategy"], "#888888")
                short = row["strategy"].replace("scGPT ", "").replace("PCA + logistic regression", "PCA+LR")
                for ax_i, x_val, x_label in [
                    (0, row["trainable_parameters"], "Trainable parameters"),
                    (1, row["runtime_sec"], "Runtime (sec)"),
                ]:
                    size = max(30, min(300, row["runtime_sec"] * 3)) if ax_i == 0 else 50
                    axes[ax_i].scatter(
                        x_val, row["macro_f1"],
                        s=size, c=color, marker=marker,
                        edgecolors="black", linewidths=0.5, zorder=3,
                    )
                    axes[ax_i].annotate(
                        short, (x_val, row["macro_f1"]),
                        fontsize=6, ha="left", va="bottom",
                        xytext=(4, 3), textcoords="offset points",
                    )
        axes[0].set_xscale("symlog", linthresh=1)
        axes[0].set_xlabel("Trainable parameters")
        axes[0].set_ylabel("Macro F1")
        axes[0].set_title("Efficiency-Performance Trade-off", fontweight="bold")
        axes[1].set_xlabel("Wall-clock runtime (sec)")
        axes[1].set_ylabel("Macro F1")
        axes[1].set_title("Runtime-Performance Trade-off", fontweight="bold")
        if len(datasets) > 1:
            legend_elements = [
                Line2D([0], [0], marker=markers[i % len(markers)], color="w",
                       markerfacecolor="#666666", markersize=7, label=d.replace("_", " "))
                for i, d in enumerate(datasets)
            ]
            axes[0].legend(handles=legend_elements, frameon=False, fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "annotation_pareto.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / "annotation_pareto.svg", bbox_inches="tight")
    plt.close(figure)
    return summary


def _save_radar_figure(frame: pd.DataFrame, output_dir: Path) -> None:
    """Radar / spider chart comparing strategies across multiple metrics."""
    from matplotlib import pyplot as plt

    _publication_style()
    radar_metrics = ["macro_f1", "weighted_f1", "balanced_accuracy", "macro_precision", "macro_recall"]
    radar_labels = ["Macro F1", "Weighted F1", "Balanced Acc.", "Precision", "Recall"]
    available = [m for m in radar_metrics if m in frame.columns]
    if frame.empty or len(available) < 3:
        return
    labels_used = [radar_labels[radar_metrics.index(m)] for m in available]

    summary = (
        frame.groupby(["model", "strategy"], as_index=False)[available]
        .mean(numeric_only=True)
        .sort_values("model", key=lambda c: c.map(_strategy_sort_value))
    )
    n = len(available)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    figure, axis = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    axis.set_theta_offset(np.pi / 2)
    axis.set_theta_direction(-1)
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(labels_used, fontsize=8)
    axis.set_ylim(0, 1.0)
    axis.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=6, color="#666666")
    axis.spines["polar"].set_visible(False)

    for _, row in summary.iterrows():
        values = [float(row[m]) for m in available]
        values += values[:1]
        color = STRATEGY_PALETTE.get(row["strategy"], None)
        short = str(row["strategy"]).replace("scGPT ", "")
        axis.plot(angles, values, linewidth=1.3, label=short, color=color)
        axis.fill(angles, values, alpha=0.08, color=color)
    axis.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), frameon=False, fontsize=7)
    axis.set_title("Multi-Metric Strategy Comparison", fontweight="bold", pad=20)
    figure.tight_layout()
    figure.savefig(output_dir / "annotation_radar.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / "annotation_radar.svg", bbox_inches="tight")
    summary.to_csv(output_dir / "annotation_radar.csv", index=False)
    plt.close(figure)


def _save_per_class_heatmap(frame: pd.DataFrame, output_dir: Path) -> None:
    """Per-class F1 heatmap for the best seed of each strategy (full-label, per dataset)."""
    import seaborn as sns
    from matplotlib import pyplot as plt

    _publication_style()
    if frame.empty or "per_class_f1" not in frame.columns:
        return
    rows_with_pcf1 = frame.dropna(subset=["per_class_f1"])
    if rows_with_pcf1.empty:
        return

    dataset_names = list(rows_with_pcf1["dataset"].unique())
    for dataset_name in dataset_names:
        df = rows_with_pcf1[rows_with_pcf1["dataset"] == dataset_name]
        best_rows = df.sort_values("macro_f1", ascending=False).drop_duplicates(subset=["model"])
        best_rows = best_rows.sort_values("model", key=lambda c: c.map(_strategy_sort_value))

        heatmap_data: dict[str, list[float]] = {}
        max_classes = 0
        for _, row in best_rows.iterrows():
            pcf1 = row["per_class_f1"]
            if isinstance(pcf1, str):
                pcf1 = json.loads(pcf1)
            if isinstance(pcf1, list):
                short = str(row["strategy"]).replace("scGPT ", "")
                heatmap_data[short] = [float(v) for v in pcf1]
                max_classes = max(max_classes, len(pcf1))
        if not heatmap_data or max_classes == 0:
            continue
        for key in heatmap_data:
            while len(heatmap_data[key]) < max_classes:
                heatmap_data[key].append(float("nan"))

        matrix = pd.DataFrame(heatmap_data).T
        matrix.columns = [f"Class {i}" for i in range(max_classes)]

        figure, axis = plt.subplots(figsize=(max(6, 0.5 * max_classes), max(3, 0.4 * len(matrix))))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"label": "F1 Score", "shrink": 0.8},
            ax=axis,
        )
        dataset_label = dataset_name.replace("_", " ").title()
        axis.set_title(f"Per-Class F1 Scores: {dataset_label}", fontweight="bold")
        axis.set_ylabel("")
        axis.tick_params(axis="y", rotation=0)
        figure.tight_layout()
        suffix = f"_{dataset_name}" if len(dataset_names) > 1 else ""
        figure.savefig(
            output_dir / f"annotation_per_class_f1{suffix}.png", dpi=300, bbox_inches="tight"
        )
        figure.savefig(output_dir / f"annotation_per_class_f1{suffix}.svg", bbox_inches="tight")
        matrix.to_csv(output_dir / f"annotation_per_class_f1{suffix}.csv")
        plt.close(figure)


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
    completed_rows: int,
    expected_rows: int,
    strategies: tuple[str, ...] = BENCHMARK_MODELS,
) -> dict[str, Any]:
    full_bests: dict[str, dict[str, Any]] = {}
    if not full_label_frame.empty:
        for dataset_name in full_label_frame["dataset"].drop_duplicates():
            dataset_frame = full_label_frame[full_label_frame["dataset"] == dataset_name]
            if dataset_frame.empty:
                continue
            # Aggregate across folds to get per-strategy mean (not single-fold max)
            agg_cols = [
                c for c in ["macro_f1", "balanced_accuracy", "accuracy",
                            "trainable_parameters", "runtime_sec"]
                if c in dataset_frame.columns
            ]
            strategy_means = (
                dataset_frame.groupby(["model", "strategy"], as_index=False)[agg_cols]
                .mean(numeric_only=True)
            )
            ranked = _ranking_columns(strategy_means)
            if ranked.empty:
                continue
            best = ranked.iloc[0]
            full_bests[dataset_name] = {
                "model": str(best["model"]),
                "strategy": str(best["strategy"]),
                "macro_f1": round(float(best["macro_f1"]), 4),
                "balanced_accuracy": round(float(best["balanced_accuracy"]), 4),
            }

    datasets = sorted(
        set(full_label_frame.get("dataset", pd.Series(dtype=object)))
        .union(low_label_frame.get("dataset", pd.Series(dtype=object)))
        .union(cross_study_frame.get("dataset", pd.Series(dtype=object)))
    )
    summary = {
        "task": ANNOTATION_TASK_SPEC.task_name,
        "datasets": datasets,
        "strategies": list(strategies),
        "full_label_rows": int(len(full_label_frame)),
        "low_label_rows": int(len(low_label_frame)),
        "cross_study_rows": int(len(cross_study_frame)),
        "completed_rows": int(completed_rows),
        "expected_rows": int(expected_rows),
        "complete": bool(completed_rows >= expected_rows),
        "best_full_label_by_dataset": full_bests,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Annotation pillar benchmark summary",
        "",
        f"- Full-label rows: `{summary['full_label_rows']}`",
        f"- Low-label rows: `{summary['low_label_rows']}`",
        f"- Cross-study rows: `{summary['cross_study_rows']}`",
        f"- Completed rows: `{summary['completed_rows']}` / `{summary['expected_rows']}`",
        f"- Complete bundle: `{summary['complete']}`",
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
    n_folds: int,
    seed: int,
    label_fractions: tuple[float, ...],
    cross_study_folds: tuple[str, ...],
    resume: bool = False,
    aggregate_only: bool = False,
) -> dict[str, Path]:
    dataset_profile, profile_settings = _benchmark_profile(profile)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    # Pre-load the scGPT checkpoint weights once for the entire benchmark run.
    # All strategy model loads reuse this in-memory dict instead of re-reading
    # the checkpoint file from disk for every run.
    scgpt_strategies = [s for s in strategies if s != "pca_logistic_annotation"]
    preloaded_state_dict: dict | None = None
    if scgpt_strategies and not aggregate_only:
        preloaded_state_dict = load_scgpt_checkpoint_state_dict("whole-human")

    if not aggregate_only:
        for dataset_name in datasets:
            dataset_spec = ANNOTATION_DATASET_SPECS[dataset_name]
            adata, label_key, batch_key = _load_annotation_dataset(
                dataset_name,
                profile=dataset_profile,
            )
            prepared: ScGPTPreparedData | None = None
            if dataset_name == "openproblems_human_pancreas" and scgpt_strategies:
                prepared = _prepare_scgpt_data_for_dataset(
                    dataset_name,
                    adata,
                    label_key=label_key,
                    profile_settings=profile_settings,
                    pancreas_prepared=None,
                )
            if "full_label" in regimes and "full_label" in dataset_spec.regimes:
                rows.extend(
                    _collect_full_label_rows(
                        dataset_name=dataset_name,
                        adata=adata,
                        prepared=prepared,
                        label_key=label_key,
                        batch_key=batch_key,
                        n_folds=n_folds,
                        seed=seed,
                        strategies=strategies,
                        output_dir=output_dir,
                        profile_settings=profile_settings,
                        profile=dataset_profile,
                        resume=resume,
                        preloaded_state_dict=preloaded_state_dict,
                    )
                )
            if "low_label" in regimes and "low_label" in dataset_spec.regimes:
                rows.extend(
                    _collect_low_label_rows(
                        dataset_name=dataset_name,
                        adata=adata,
                        prepared=prepared,
                        label_key=label_key,
                        batch_key=batch_key,
                        n_folds=n_folds,
                        seed=seed,
                        label_fractions=label_fractions,
                        strategies=strategies,
                        output_dir=output_dir,
                        profile_settings=profile_settings,
                        profile=dataset_profile,
                        resume=resume,
                        preloaded_state_dict=preloaded_state_dict,
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
                        prepared=prepared,
                        label_key=label_key,
                        batch_key=batch_key,
                        seed=seed,
                        fold_names=cross_study_folds,
                        strategies=strategies,
                        output_dir=output_dir,
                        profile_settings=profile_settings,
                        resume=resume,
                        preloaded_state_dict=preloaded_state_dict,
                    )
                )

    existing_rows = _collect_existing_rows(output_dir)
    merged_rows = _merge_rows(rows, existing_rows)
    frame = _ordered_frame(merged_rows)
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

    # Save consolidated results file (single source of truth for all figures)
    frame.to_csv(output_dir / "all_results.csv", index=False)

    full_label_frame.to_csv(full_dir / "strategy_metrics.csv", index=False)
    low_label_frame.to_csv(low_dir / "strategy_metrics.csv", index=False)
    cross_study_frame.to_csv(cross_dir / "strategy_metrics.csv", index=False)

    _save_performance_figure(full_label_frame, figures_dir)
    _save_low_label_figure(low_label_frame, figures_dir)
    _save_cross_study_figure(cross_study_frame, figures_dir)
    _save_pareto_figure(full_label_frame, figures_dir)
    _save_radar_figure(full_label_frame, figures_dir)
    _save_per_class_heatmap(full_label_frame, figures_dir)
    _write_summary(
        full_label_frame=full_label_frame,
        low_label_frame=low_label_frame,
        cross_study_frame=cross_study_frame,
        output_dir=output_dir,
        completed_rows=len(merged_rows),
        expected_rows=_expected_row_count(
            datasets=datasets,
            regimes=regimes,
            strategies=strategies,
            n_folds=n_folds,
            label_fractions=label_fractions,
            cross_study_folds=cross_study_folds,
        ),
        strategies=strategies,
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


def _regenerate_figures_from_csv(output_dir: Path) -> None:
    """Regenerate all figures from the saved all_results.csv."""
    csv_path = output_dir / "all_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No all_results.csv found at {csv_path}. "
            "Run the benchmark or --aggregate-only first."
        )
    frame = pd.read_csv(csv_path)
    full_label_frame = frame[frame["regime"] == "full_label"].copy() if not frame.empty else pd.DataFrame()
    low_label_frame = frame[frame["regime"] == "low_label"].copy() if not frame.empty else pd.DataFrame()
    cross_study_frame = frame[frame["regime"] == "cross_study"].copy() if not frame.empty else pd.DataFrame()

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    _save_performance_figure(full_label_frame, figures_dir)
    _save_low_label_figure(low_label_frame, figures_dir)
    _save_cross_study_figure(cross_study_frame, figures_dir)
    _save_pareto_figure(full_label_frame, figures_dir)
    _save_radar_figure(full_label_frame, figures_dir)
    _save_per_class_heatmap(full_label_frame, figures_dir)
    print(f"Figures regenerated in {figures_dir}")


def main() -> None:
    args = parse_args()
    if args.figures_only:
        _regenerate_figures_from_csv(args.output_dir)
        return
    outputs = run_annotation_benchmark(
        datasets=tuple(args.dataset),
        regimes=tuple(args.regime),
        profile=str(args.profile),
        strategies=tuple(args.strategies),
        output_dir=args.output_dir,
        n_folds=int(args.n_folds),
        seed=int(args.seed),
        label_fractions=tuple(float(value) for value in args.label_fractions),
        cross_study_folds=tuple(str(value) for value in args.cross_study_folds),
        resume=bool(args.resume),
        aggregate_only=bool(args.aggregate_only),
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
