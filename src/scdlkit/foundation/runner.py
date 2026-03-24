"""High-level experimental wrapper for scGPT annotation adaptation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sized
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from sklearn.linear_model import LogisticRegression

from scdlkit.evaluation import evaluate_predictions, save_markdown_report, save_metrics_table
from scdlkit.foundation.annotation import ScGPTAnnotationModel, load_scgpt_annotation_model
from scdlkit.foundation.cache import DEFAULT_SCGPT_CHECKPOINT
from scdlkit.foundation.data import (
    ScGPTAnnotationDataReport,
    ScGPTPreparedData,
    ScGPTSplitData,
    inspect_scgpt_annotation_data,
    prepare_scgpt_data,
    split_scgpt_data,
)
from scdlkit.foundation.lora import ScGPTLoRAConfig
from scdlkit.foundation.peft import (
    AnnotationStrategy,
    LoRAConfig,
    PEFTConfig,
    count_trainable_parameters,
    deserialize_strategy_configs,
    resolve_strategy_configs,
    serialize_strategy_configs,
)
from scdlkit.foundation.scgpt import ScGPTEmbeddingModel, load_scgpt_model
from scdlkit.training import Trainer
from scdlkit.visualization.classification import plot_confusion_matrix

_ALLOWED_STRATEGIES = (
    "frozen_probe",
    "head",
    "full_finetune",
    "lora",
    "adapter",
    "prefix_tuning",
    "ia3",
)
_ALLOWED_STRATEGY_SET = {str(strategy) for strategy in _ALLOWED_STRATEGIES}
_DEFAULT_STRATEGIES: tuple[AnnotationStrategy, ...] = ("frozen_probe", "head")
_STRATEGY_EPOCHS: dict[str, int] = {
    "head": 3,
    "full_finetune": 2,
    "lora": 2,
    "adapter": 3,
    "prefix_tuning": 3,
    "ia3": 3,
}
_STRATEGY_LR: dict[str, float] = {
    "head": 1e-3,
    "full_finetune": 5e-4,
    "lora": 1e-3,
    "adapter": 1e-3,
    "prefix_tuning": 1e-3,
    "ia3": 1e-3,
}


@dataclass(frozen=True, slots=True)
class ScGPTAnnotationRunSummary:
    """Summary of one experimental scGPT annotation wrapper run."""

    strategy_metrics: pd.DataFrame
    best_strategy: str
    output_dir: Path | None
    checkpoint_path: Path | None
    label_categories: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _FrozenProbeState:
    coef: np.ndarray
    intercept: np.ndarray
    classes: np.ndarray


@dataclass(frozen=True, slots=True)
class _StrategyResult:
    strategy: AnnotationStrategy
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    runtime_sec: float
    trainable_parameters: int
    predictions_test: dict[str, np.ndarray]
    model: ScGPTEmbeddingModel | ScGPTAnnotationModel | None
    probe_state: _FrozenProbeState | None


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _expand_probabilities(
    probabilities: np.ndarray,
    classes: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    if probabilities.shape[1] == num_classes:
        return probabilities.astype(np.float32, copy=False)
    expanded = np.zeros((probabilities.shape[0], num_classes), dtype=np.float32)
    class_indices = np.asarray(classes, dtype=int)
    expanded[:, class_indices] = np.asarray(probabilities, dtype=np.float32)
    return expanded


def _frozen_probe_probabilities(latent: np.ndarray, state: _FrozenProbeState) -> np.ndarray:
    if state.classes.size == 2 and state.coef.shape[0] == 1:
        linear = latent @ state.coef[0].T + state.intercept[0]
        positive = 1.0 / (1.0 + np.exp(-linear))
        negative = 1.0 - positive
        return np.column_stack([negative, positive]).astype(np.float32)
    scores = latent @ state.coef.T + state.intercept
    return _softmax(scores.astype(np.float32))


def _prediction_payload(
    *,
    probabilities: np.ndarray,
    latent: np.ndarray,
    label_categories: tuple[str, ...],
) -> dict[str, np.ndarray]:
    label_codes = probabilities.argmax(axis=1).astype(np.int64)
    labels = np.asarray([label_categories[index] for index in label_codes], dtype=object)
    return {
        "label_codes": label_codes,
        "labels": labels,
        "probabilities": probabilities.astype(np.float32, copy=False),
        "latent": latent.astype(np.float32, copy=False),
    }


def _markdown_cell(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isnan(numeric):
            return ""
        return f"{numeric:.4f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        f"| {' | '.join(headers)} |",
        f"| {' | '.join('---' for _ in headers)} |",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = [_markdown_cell(value) for value in row]
        lines.append(f"| {' | '.join(cells)} |")
    return "\n".join(lines)


def _default_min_gene_overlap(adata: AnnData) -> int:
    source = adata.raw.to_adata() if adata.raw is not None else adata
    return max(1, min(500, int(np.ceil(source.n_vars * 0.8))))


def _classification_summary(metrics: Mapping[str, Any]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in ("accuracy", "macro_f1", "balanced_accuracy", "auroc_ovr"):
        value = metrics.get(key)
        if isinstance(value, (float, int, np.floating, np.integer)):
            summary[key] = float(value)
    return summary


def _evaluation_dataset(split: ScGPTSplitData) -> Any:
    if split.val is not None and len(cast(Sized, split.val)) > 0:
        return split.val
    if split.test is not None and len(cast(Sized, split.test)) > 0:
        return split.test
    return split.train


def _test_dataset(split: ScGPTSplitData) -> Any:
    if split.test is not None and len(cast(Sized, split.test)) > 0:
        return split.test
    return _evaluation_dataset(split)


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
    plot_adata.obsm["X_scgpt_wrapper"] = latent
    if label_key not in plot_adata.obs:
        plot_adata.obs[label_key] = np.asarray(["cells"] * plot_adata.n_obs, dtype=object)

    n_obs = int(plot_adata.n_obs)
    if n_obs < 4:
        figure, axis = plt.subplots(figsize=(5, 4))
        axis.scatter(latent[:, 0], latent[:, 1] if latent.shape[1] > 1 else np.zeros(n_obs), s=32)
        axis.set_title("Latent embedding")
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        return

    sc.pp.neighbors(
        plot_adata,
        use_rep="X_scgpt_wrapper",
        n_neighbors=max(2, min(10, n_obs - 1)),
    )
    sc.tl.umap(plot_adata, random_state=seed, init_pos="random")
    figure = sc.pl.umap(plot_adata, color=label_key, return_fig=True, frameon=False)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


class ScGPTAnnotationRunner:
    """High-level experimental wrapper for scGPT annotation adaptation.

    Parameters
    ----------
    label_key
        Required target label column in ``adata.obs``.
    checkpoint
        Experimental checkpoint identifier. The public path currently supports
        only scGPT ``whole-human``.
    strategies
        Strategy ladder to compare. The default keeps the beginner path on
        ``("frozen_probe", "head")``.
    batch_size
        Wrapper batch size for preparation, inference, and training.
    val_size
        Validation split fraction.
    test_size
        Test split fraction.
    random_state
        Random seed used for splitting and trainable strategies.
    device
        ``"auto"``, ``"cpu"``, or ``"cuda"``.
    classifier_dropout
        Dropout applied in the annotation classifier head.
    strategy_configs
        Optional per-strategy PEFT configuration mapping for ``lora``,
        ``adapter``, ``prefix_tuning``, and ``ia3``.
    lora_config
        Backward-compatible LoRA configuration alias. Use
        ``strategy_configs={"lora": LoRAConfig(...)}`` for new code.
    output_dir
        Optional artifact directory for reports, plots, and saved state.

    Notes
    -----
    The default strategy ladder is ``("frozen_probe", "head")`` so the
    beginner path stays CPU-friendly. LoRA remains available through an
    explicit ``strategies=(...)`` override.
    """

    def __init__(
        self,
        *,
        label_key: str,
        checkpoint: str = DEFAULT_SCGPT_CHECKPOINT,
        strategies: tuple[AnnotationStrategy, ...] = _DEFAULT_STRATEGIES,
        batch_size: int = 64,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
        device: str = "auto",
        classifier_dropout: float = 0.1,
        strategy_configs: Mapping[str, PEFTConfig] | None = None,
        lora_config: ScGPTLoRAConfig | LoRAConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        invalid = sorted({str(value) for value in strategies} - _ALLOWED_STRATEGY_SET)
        if invalid:
            msg = (
                "Unsupported scGPT annotation strategies: "
                f"{', '.join(invalid)}. Supported values are {', '.join(_ALLOWED_STRATEGIES)}."
            )
            raise ValueError(msg)
        self.label_key = label_key
        self.checkpoint = checkpoint
        self.strategies: tuple[AnnotationStrategy, ...] = tuple(strategies)
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.device = device
        self.classifier_dropout = classifier_dropout
        self.strategy_configs = resolve_strategy_configs(
            strategies=tuple(str(value) for value in self.strategies),
            strategy_configs=strategy_configs,
            lora_config=LoRAConfig(
                rank=lora_config.rank,
                alpha=lora_config.alpha,
                dropout=lora_config.dropout,
                target_modules=tuple(lora_config.target_modules),
            )
            if lora_config is not None
            else None,
        )
        resolved_lora = self.strategy_configs.get("lora")
        self.lora_config = resolved_lora if isinstance(resolved_lora, LoRAConfig) else None
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.data_report_: ScGPTAnnotationDataReport | None = None
        self.summary_: ScGPTAnnotationRunSummary | None = None
        self.best_strategy_: AnnotationStrategy | None = None
        self.label_categories_: tuple[str, ...] | None = None
        self._best_model: ScGPTEmbeddingModel | ScGPTAnnotationModel | None = None
        self._best_probe_state: _FrozenProbeState | None = None
        self._checkpoint_path: Path | None = None
        self._cache_dir: Path | None = None

    def inspect(self, adata: AnnData) -> ScGPTAnnotationDataReport:
        """Inspect dataset compatibility before fitting.

        Parameters
        ----------
        adata
            Labeled human single-cell ``AnnData`` to inspect.

        Returns
        -------
        ScGPTAnnotationDataReport
            Compatibility report with overlap, class-balance, and split
            warnings.
        """

        report = inspect_scgpt_annotation_data(
            adata,
            label_key=self.label_key,
            checkpoint=self.checkpoint,
        )
        self.data_report_ = report
        self.label_categories_ = report.label_categories
        return report

    def _prepare_data(self, adata: AnnData, *, label_key: str | None) -> ScGPTPreparedData:
        return prepare_scgpt_data(
            adata,
            checkpoint=self.checkpoint,
            label_key=label_key,
            batch_size=self.batch_size,
            min_gene_overlap=_default_min_gene_overlap(adata),
        )

    def _run_frozen_probe(self, *, split: ScGPTSplitData) -> _StrategyResult:
        started_at = perf_counter()
        model = load_scgpt_model(
            self.checkpoint,
            device=self.device,
            cache_dir=self._cache_dir,
        )
        trainer = Trainer(
            model=model,
            task="representation",
            batch_size=split.batch_size,
            device=self.device,
        )
        train_predictions = trainer.predict_dataset(split.train)
        validation_predictions = trainer.predict_dataset(_evaluation_dataset(split))
        test_predictions = trainer.predict_dataset(_test_dataset(split))

        classifier = LogisticRegression(max_iter=1000, random_state=self.random_state)
        classifier.fit(train_predictions["latent"], train_predictions["y"])
        num_classes = len(self.label_categories_ or ())
        validation_logits = _expand_probabilities(
            classifier.predict_proba(validation_predictions["latent"]),
            np.asarray(classifier.classes_, dtype=np.int64),
            num_classes=num_classes,
        )
        test_logits = _expand_probabilities(
            classifier.predict_proba(test_predictions["latent"]),
            np.asarray(classifier.classes_, dtype=np.int64),
            num_classes=num_classes,
        )
        validation_metrics = evaluate_predictions(
            "classification",
            {"y": validation_predictions["y"], "logits": validation_logits},
        )
        test_metrics = evaluate_predictions(
            "classification",
            {"y": test_predictions["y"], "logits": test_logits},
        )
        runtime_sec = perf_counter() - started_at
        return _StrategyResult(
            strategy="frozen_probe",
            validation_metrics=_classification_summary(validation_metrics),
            test_metrics=_classification_summary(test_metrics),
            runtime_sec=runtime_sec,
            trainable_parameters=0,
            predictions_test={
                "y": test_predictions["y"],
                "logits": test_logits,
                "latent": test_predictions["latent"],
            },
            model=model,
            probe_state=_FrozenProbeState(
                coef=np.asarray(classifier.coef_, dtype=np.float32),
                intercept=np.asarray(classifier.intercept_, dtype=np.float32),
                classes=np.asarray(classifier.classes_, dtype=np.int64),
            ),
        )

    def _run_trainable_strategy(
        self,
        *,
        split: ScGPTSplitData,
        strategy: AnnotationStrategy,
    ) -> _StrategyResult:
        started_at = perf_counter()
        model = load_scgpt_annotation_model(
            num_classes=len(self.label_categories_ or ()),
            checkpoint=self.checkpoint,
            tuning_strategy=strategy,
            label_categories=self.label_categories_,
            strategy_config=self.strategy_configs.get(strategy),
            classifier_dropout=self.classifier_dropout,
            device=self.device,
            cache_dir=self._cache_dir,
        )
        trainer = Trainer(
            model=model,
            task="classification",
            batch_size=split.batch_size,
            epochs=_STRATEGY_EPOCHS[strategy],
            lr=_STRATEGY_LR[strategy],
            device=self.device,
            early_stopping_patience=2,
            seed=self.random_state,
        )
        trainer.fit(split.train, split.val)
        validation_predictions = trainer.predict_dataset(_evaluation_dataset(split))
        test_predictions = trainer.predict_dataset(_test_dataset(split))
        validation_metrics = evaluate_predictions("classification", validation_predictions)
        test_metrics = evaluate_predictions("classification", test_predictions)
        runtime_sec = perf_counter() - started_at
        return _StrategyResult(
            strategy=strategy,
            validation_metrics=_classification_summary(validation_metrics),
            test_metrics=_classification_summary(test_metrics),
            runtime_sec=runtime_sec,
            trainable_parameters=count_trainable_parameters(trainer.model),
            predictions_test=test_predictions,
            model=cast(ScGPTAnnotationModel, trainer.model),
            probe_state=None,
        )

    def _strategy_frame(self, results: list[_StrategyResult]) -> pd.DataFrame:
        records = [
            {
                "strategy": result.strategy,
                "validation_accuracy": result.validation_metrics["accuracy"],
                "validation_macro_f1": result.validation_metrics["macro_f1"],
                "validation_balanced_accuracy": result.validation_metrics.get(
                    "balanced_accuracy",
                    float("nan"),
                ),
                "validation_auroc_ovr": result.validation_metrics.get("auroc_ovr", float("nan")),
                "test_accuracy": result.test_metrics["accuracy"],
                "test_macro_f1": result.test_metrics["macro_f1"],
                "test_balanced_accuracy": result.test_metrics.get(
                    "balanced_accuracy",
                    float("nan"),
                ),
                "test_auroc_ovr": result.test_metrics.get("auroc_ovr", float("nan")),
                "runtime_sec": result.runtime_sec,
                "trainable_parameters": result.trainable_parameters,
            }
            for result in results
        ]
        frame = pd.DataFrame.from_records(records)
        frame["strategy_rank"] = frame["strategy"].map(
            {name: index for index, name in enumerate(self.strategies)}
        )
        ordered = frame.sort_values(
            [
                "validation_macro_f1",
                "validation_balanced_accuracy",
                "validation_accuracy",
                "trainable_parameters",
                "runtime_sec",
                "strategy_rank",
            ],
            ascending=[False, False, False, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        return ordered.drop(columns=["strategy_rank"])

    def _write_artifacts(
        self,
        *,
        adata: AnnData,
        strategy_metrics: pd.DataFrame,
        results: dict[str, _StrategyResult],
    ) -> None:
        if self.output_dir is None or self.summary_ is None or self.best_strategy_ is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        strategy_metrics.to_csv(self.output_dir / "strategy_metrics.csv", index=False)

        best_row = strategy_metrics.loc[strategy_metrics["strategy"] == self.best_strategy_].iloc[0]
        report_metrics = {
            "best_strategy": self.best_strategy_,
            "validation_accuracy": float(best_row["validation_accuracy"]),
            "validation_macro_f1": float(best_row["validation_macro_f1"]),
            "validation_balanced_accuracy": float(best_row["validation_balanced_accuracy"]),
            "test_accuracy": float(best_row["test_accuracy"]),
            "test_macro_f1": float(best_row["test_macro_f1"]),
            "test_balanced_accuracy": float(best_row["test_balanced_accuracy"]),
            "num_cells": float(self.data_report_.num_cells if self.data_report_ else adata.n_obs),
            "num_genes_matched": float(
                self.data_report_.num_genes_matched if self.data_report_ else 0
            ),
        }
        if not pd.isna(best_row["validation_auroc_ovr"]):
            report_metrics["validation_auroc_ovr"] = float(best_row["validation_auroc_ovr"])
        if not pd.isna(best_row["test_auroc_ovr"]):
            report_metrics["test_auroc_ovr"] = float(best_row["test_auroc_ovr"])
        extra_sections: list[str] = []
        if self.data_report_ is not None:
            if self.data_report_.warnings:
                extra_sections.extend(
                    [
                        "## Inspection warnings",
                        "",
                        *[f"- {warning}" for warning in self.data_report_.warnings],
                        "",
                    ]
                )
            extra_sections.extend(
                [
                    "## Dataset inspection",
                    "",
                    f"- Label key: `{self.data_report_.label_key}`",
                    (
                        f"- Matched genes: `{self.data_report_.num_genes_matched}` / "
                        f"`{self.data_report_.num_input_genes}`"
                    ),
                    f"- Smallest class size: `{self.data_report_.min_class_count}`",
                    "",
                ]
            )
        extra_sections.extend(
            [
                "## Strategy comparison",
                "",
                _markdown_table(strategy_metrics),
            ]
        )
        save_markdown_report(
            report_metrics,
            path=self.output_dir / "report.md",
            title="Experimental scGPT dataset-specific annotation report",
            extra_sections=extra_sections,
        )
        save_metrics_table(report_metrics, self.output_dir / "report.csv")

        best_metrics = evaluate_predictions(
            "classification",
            results[self.best_strategy_].predictions_test,
        )
        confusion_figure, _ = plot_confusion_matrix(
            best_metrics["confusion_matrix"],
            class_names=list(self.label_categories_ or ()),
        )
        confusion_figure.savefig(
            self.output_dir / "best_strategy_confusion_matrix.png",
            dpi=150,
            bbox_inches="tight",
        )
        confusion_figure.clf()

        frozen_result = results.get("frozen_probe")
        if (
            frozen_result is not None
            and frozen_result.model is not None
            and frozen_result.probe_state is not None
        ):
            frozen_predictions = self._predict_with_frozen_probe(
                adata,
                model=cast(ScGPTEmbeddingModel, frozen_result.model),
                probe_state=frozen_result.probe_state,
            )
            _save_scanpy_umap(
                adata,
                frozen_predictions["latent"],
                self.label_key,
                self.output_dir / "frozen_embedding_umap.png",
                seed=self.random_state,
            )
        best_predictions = self.predict(adata)
        _save_scanpy_umap(
            adata,
            best_predictions["latent"],
            self.label_key,
            self.output_dir / "best_strategy_embedding_umap.png",
            seed=self.random_state,
        )

    def _predict_with_frozen_probe(
        self,
        adata: AnnData,
        *,
        model: ScGPTEmbeddingModel,
        probe_state: _FrozenProbeState,
    ) -> dict[str, np.ndarray]:
        prepared = self._prepare_data(adata, label_key=None)
        trainer = Trainer(
            model=model,
            task="representation",
            batch_size=prepared.batch_size,
            device=self.device,
        )
        predictions = trainer.predict_dataset(prepared.dataset)
        probabilities = _expand_probabilities(
            _frozen_probe_probabilities(predictions["latent"], probe_state),
            probe_state.classes,
            num_classes=len(self.label_categories_ or ()),
        )
        return _prediction_payload(
            probabilities=probabilities,
            latent=predictions["latent"],
            label_categories=self.label_categories_ or (),
        )

    def fit_compare(self, adata: AnnData) -> ScGPTAnnotationRunSummary:
        """Inspect, compare strategies, and keep the best fitted strategy.

        Parameters
        ----------
        adata
            Labeled human single-cell ``AnnData`` to adapt on.

        Returns
        -------
        ScGPTAnnotationRunSummary
            Summary with the strategy metrics table, best strategy, output
            directory, and checkpoint path when saved later.

        Raises
        ------
        ValueError
            If the dataset does not expose at least two label categories.
        """

        report = self.inspect(adata)
        if len(report.label_categories) < 2:
            msg = "Experimental scGPT annotation requires at least two label categories."
            raise ValueError(msg)
        prepared = self._prepare_data(adata, label_key=self.label_key)
        split = split_scgpt_data(
            prepared,
            val_size=self.val_size,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=True,
        )

        results: list[_StrategyResult] = []
        for strategy in self.strategies:
            if strategy == "frozen_probe":
                results.append(self._run_frozen_probe(split=split))
            else:
                results.append(self._run_trainable_strategy(split=split, strategy=strategy))

        strategy_frame = self._strategy_frame(results)
        best_strategy = cast(AnnotationStrategy, str(strategy_frame.iloc[0]["strategy"]))
        result_by_strategy: dict[str, _StrategyResult] = {
            str(result.strategy): result for result in results
        }
        best_result = result_by_strategy[best_strategy]
        self.best_strategy_ = best_strategy
        self.label_categories_ = report.label_categories
        self._best_model = best_result.model
        self._best_probe_state = best_result.probe_state
        self.summary_ = ScGPTAnnotationRunSummary(
            strategy_metrics=strategy_frame.copy(),
            best_strategy=best_strategy,
            output_dir=self.output_dir,
            checkpoint_path=None,
            label_categories=report.label_categories,
        )
        self._write_artifacts(
            adata=adata,
            strategy_metrics=strategy_frame,
            results=result_by_strategy,
        )
        return self.summary_

    def predict(self, adata: AnnData) -> dict[str, np.ndarray]:
        """Predict labels, probabilities, and embeddings with the best strategy.

        Parameters
        ----------
        adata
            ``AnnData`` to annotate with the fitted or loaded runner.

        Returns
        -------
        dict[str, numpy.ndarray]
            Prediction payload containing ``label_codes``, ``labels``,
            ``probabilities``, and ``latent``.

        Raises
        ------
        RuntimeError
            If the runner has not been fitted or loaded yet.
        """

        if (
            self.best_strategy_ is None
            or self._best_model is None
            or self.label_categories_ is None
        ):
            msg = "No fitted scGPT annotation strategy is available. Call fit_compare() or load()."
            raise RuntimeError(msg)

        prepared = self._prepare_data(adata, label_key=None)
        if self.best_strategy_ == "frozen_probe":
            if self._best_probe_state is None:
                msg = "Frozen probe state is unavailable. Fit or load the runner again."
                raise RuntimeError(msg)
            trainer = Trainer(
                model=self._best_model,
                task="representation",
                batch_size=prepared.batch_size,
                device=self.device,
            )
            predictions = trainer.predict_dataset(prepared.dataset)
            probabilities = _expand_probabilities(
                _frozen_probe_probabilities(predictions["latent"], self._best_probe_state),
                self._best_probe_state.classes,
                num_classes=len(self.label_categories_),
            )
            return _prediction_payload(
                probabilities=probabilities,
                latent=predictions["latent"],
                label_categories=self.label_categories_,
            )

        trainer = Trainer(
            model=self._best_model,
            task="classification",
            batch_size=prepared.batch_size,
            device=self.device,
        )
        predictions = trainer.predict_dataset(prepared.dataset)
        probabilities = _softmax(np.asarray(predictions["logits"], dtype=np.float32))
        return _prediction_payload(
            probabilities=probabilities,
            latent=np.asarray(predictions["latent"], dtype=np.float32),
            label_categories=self.label_categories_,
        )

    def annotate_adata(
        self,
        adata: AnnData,
        *,
        obs_key: str = "scgpt_label",
        embedding_key: str = "X_scgpt_best",
        inplace: bool = True,
    ) -> AnnData | None:
        """Write predicted labels and latent embeddings back into ``AnnData``.

        Parameters
        ----------
        adata
            Target ``AnnData`` to annotate.
        obs_key
            Base observation key for predicted labels.
        embedding_key
            ``adata.obsm`` key for the best-strategy latent embedding.
        inplace
            When ``True``, write directly into ``adata`` and return ``None``.

        Returns
        -------
        AnnData | None
            ``None`` for in-place writes, otherwise a copied annotated
            ``AnnData``.
        """

        target = adata if inplace else adata.copy()
        predictions = self.predict(target)
        target.obs[obs_key] = pd.Categorical(
            predictions["labels"],
            categories=list(self.label_categories_ or ()),
        )
        target.obs[f"{obs_key}_code"] = predictions["label_codes"]
        target.obs[f"{obs_key}_confidence"] = predictions["probabilities"].max(axis=1)
        target.obsm[embedding_key] = predictions["latent"]
        if inplace:
            return None
        return target

    def save(self, path: str | Path) -> Path:
        """Persist the best fitted wrapper state.

        Parameters
        ----------
        path
            Output directory for ``manifest.json`` and ``model_state.pt``.

        Returns
        -------
        pathlib.Path
            Directory containing the saved runner state.

        Raises
        ------
        RuntimeError
            If no fitted or loaded best strategy is available.
        """

        if (
            self.best_strategy_ is None
            or self._best_model is None
            or self.label_categories_ is None
        ):
            msg = "No fitted scGPT annotation strategy is available to save."
            raise RuntimeError(msg)

        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        strategy_metrics = []
        if self.summary_ is not None:
            strategy_metrics = self.summary_.strategy_metrics.to_dict(orient="records")
        best_metrics: dict[str, Any] = next(
            (row for row in strategy_metrics if str(row.get("strategy")) == self.best_strategy_),
            {},
        )
        manifest = {
            "checkpoint_id": self.checkpoint,
            "label_key": self.label_key,
            "label_categories": list(self.label_categories_),
            "best_strategy": self.best_strategy_,
            "strategies": list(self.strategies),
            "batch_size": self.batch_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "classifier_dropout": self.classifier_dropout,
            "strategy_configs": serialize_strategy_configs(self.strategy_configs),
            "lora_config": (
                {
                    "config_type": "lora",
                    "rank": self.lora_config.rank,
                    "alpha": self.lora_config.alpha,
                    "dropout": self.lora_config.dropout,
                    "target_modules": list(self.lora_config.target_modules),
                }
                if self.lora_config is not None
                else None
            ),
            "metrics": best_metrics,
            "strategy_metrics": strategy_metrics,
            "output_dir": str(self.output_dir) if self.output_dir is not None else None,
        }
        (output_path / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        if self.best_strategy_ == "frozen_probe":
            if self._best_probe_state is None:
                msg = "Frozen probe state is unavailable."
                raise RuntimeError(msg)
            payload: dict[str, Any] = {
                "kind": "frozen_probe",
                "coef": self._best_probe_state.coef,
                "intercept": self._best_probe_state.intercept,
                "classes": self._best_probe_state.classes,
            }
        else:
            payload = {
                "kind": "torch_state_dict",
                "state_dict": self._best_model.state_dict(),
            }
        torch.save(payload, output_path / "model_state.pt")
        self._checkpoint_path = output_path
        if self.summary_ is not None:
            self.summary_ = ScGPTAnnotationRunSummary(
                strategy_metrics=self.summary_.strategy_metrics.copy(),
                best_strategy=self.summary_.best_strategy,
                output_dir=self.summary_.output_dir,
                checkpoint_path=output_path,
                label_categories=self.summary_.label_categories,
            )
        return output_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "auto",
        cache_dir: str | Path | None = None,
    ) -> ScGPTAnnotationRunner:
        """Load a saved experimental scGPT annotation runner.

        Parameters
        ----------
        path
            Directory containing ``manifest.json`` and ``model_state.pt``.
        device
            ``"auto"``, ``"cpu"``, or ``"cuda"`` for the reloaded model.
        cache_dir
            Optional checkpoint cache root for the base scGPT checkpoint.

        Returns
        -------
        ScGPTAnnotationRunner
            Reloaded runner with the saved best strategy restored.

        Raises
        ------
        ValueError
            If the saved runner payload is incomplete or malformed.
        """

        path_obj = Path(path)
        manifest_path = path_obj / "manifest.json"
        model_state_path = path_obj / "model_state.pt"
        if not manifest_path.exists() or not model_state_path.exists():
            msg = (
                "Saved scGPT annotation runner is incomplete. Expected both "
                "`manifest.json` and `model_state.pt`."
            )
            raise ValueError(msg)

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "best_strategy" not in manifest or "label_key" not in manifest:
            msg = "Saved scGPT annotation manifest is missing required fields."
            raise ValueError(msg)

        strategy_configs_payload = manifest.get("strategy_configs")
        lora_config = manifest.get("lora_config")
        deserialized_configs = deserialize_strategy_configs(strategy_configs_payload)
        compatibility_lora: LoRAConfig | None = None
        if not deserialized_configs and lora_config is not None:
            maybe_lora = deserialize_strategy_configs({"lora": lora_config}).get("lora")
            if isinstance(maybe_lora, LoRAConfig):
                compatibility_lora = maybe_lora
        runner = cls(
            label_key=str(manifest["label_key"]),
            checkpoint=str(manifest.get("checkpoint_id", DEFAULT_SCGPT_CHECKPOINT)),
            strategies=tuple(
                cast(AnnotationStrategy, str(value))
                for value in manifest.get("strategies", _DEFAULT_STRATEGIES)
            ),
            batch_size=int(manifest.get("batch_size", 64)),
            val_size=float(manifest.get("val_size", 0.15)),
            test_size=float(manifest.get("test_size", 0.15)),
            random_state=int(manifest.get("random_state", 42)),
            device=device,
            classifier_dropout=float(manifest.get("classifier_dropout", 0.1)),
            strategy_configs=deserialized_configs or None,
            lora_config=compatibility_lora,
            output_dir=manifest.get("output_dir"),
        )
        runner._cache_dir = Path(cache_dir) if cache_dir is not None else None
        runner.best_strategy_ = cast(AnnotationStrategy, str(manifest["best_strategy"]))
        runner.label_categories_ = tuple(
            str(value) for value in manifest.get("label_categories", ())
        )
        payload = torch.load(model_state_path, map_location="cpu", weights_only=False)
        if runner.best_strategy_ == "frozen_probe":
            if payload.get("kind") != "frozen_probe":
                msg = "Saved frozen probe payload is malformed."
                raise ValueError(msg)
            runner._best_model = load_scgpt_model(
                runner.checkpoint,
                device=device,
                cache_dir=runner._cache_dir,
            )
            runner._best_probe_state = _FrozenProbeState(
                coef=np.asarray(payload["coef"], dtype=np.float32),
                intercept=np.asarray(payload["intercept"], dtype=np.float32),
                classes=np.asarray(payload["classes"], dtype=np.int64),
            )
        else:
            if payload.get("kind") != "torch_state_dict":
                msg = "Saved scGPT annotation payload is malformed."
                raise ValueError(msg)
            model = load_scgpt_annotation_model(
                num_classes=len(runner.label_categories_),
                checkpoint=runner.checkpoint,
                tuning_strategy=cast(AnnotationStrategy, runner.best_strategy_),
                label_categories=runner.label_categories_,
                strategy_config=runner.strategy_configs.get(runner.best_strategy_),
                classifier_dropout=runner.classifier_dropout,
                device=device,
                cache_dir=runner._cache_dir,
            )
            model.load_state_dict(payload["state_dict"])
            runner._best_model = model
        strategy_metrics = pd.DataFrame.from_records(manifest.get("strategy_metrics", []))
        runner.summary_ = ScGPTAnnotationRunSummary(
            strategy_metrics=strategy_metrics,
            best_strategy=runner.best_strategy_,
            output_dir=runner.output_dir,
            checkpoint_path=path_obj,
            label_categories=runner.label_categories_,
        )
        runner._checkpoint_path = path_obj
        return runner


def adapt_scgpt_annotation(
    adata: AnnData,
    *,
    label_key: str,
    checkpoint: str = DEFAULT_SCGPT_CHECKPOINT,
    strategies: tuple[AnnotationStrategy, ...] = _DEFAULT_STRATEGIES,
    batch_size: int = 64,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    device: str = "auto",
    strategy_configs: Mapping[str, PEFTConfig] | None = None,
    lora_config: ScGPTLoRAConfig | LoRAConfig | None = None,
    output_dir: str | Path | None = None,
) -> ScGPTAnnotationRunner:
    """Run the experimental wrapper-first scGPT annotation workflow in one call.

    Parameters
    ----------
    adata
        Labeled human single-cell ``AnnData`` to adapt on.
    label_key
        Required label column in ``adata.obs``.
    checkpoint
        Experimental checkpoint identifier. The public route currently supports
        only scGPT ``whole-human``.
    strategies
        Strategy ladder to compare. Defaults to ``("frozen_probe", "head")``.
    batch_size
        Wrapper batch size for preparation, inference, and training.
    val_size
        Validation split fraction.
    test_size
        Test split fraction.
    random_state
        Random seed used for splitting and trainable strategies.
    device
        ``"auto"``, ``"cpu"``, or ``"cuda"``.
    strategy_configs
        Optional per-strategy PEFT configuration mapping.
    lora_config
        Backward-compatible LoRA config alias for ``strategy_configs={"lora": ...}``.
    output_dir
        Optional artifact directory for reports, plots, and saved state.

    Returns
    -------
    ScGPTAnnotationRunner
        Fitted experimental runner holding the best strategy.

    Notes
    -----
    The default quickstart compares ``frozen_probe`` and ``head`` only. LoRA
    remains available by passing it explicitly in ``strategies``.
    """

    runner = ScGPTAnnotationRunner(
        label_key=label_key,
        checkpoint=checkpoint,
        strategies=strategies,
        batch_size=batch_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        device=device,
        strategy_configs=strategy_configs,
        lora_config=lora_config,
        output_dir=output_dir,
    )
    runner.fit_compare(adata)
    return runner
