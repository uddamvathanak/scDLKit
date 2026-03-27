"""Locked task and regime specification for the annotation paper pillar."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CrossStudyFoldSpec:
    """Definition of a held-out cross-study evaluation fold."""

    name: str
    held_out_batches: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnnotationDatasetSpec:
    """Dataset contract for the annotation benchmark."""

    name: str
    label_key: str
    batch_key: str | None
    regimes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnnotationRegimeSpec:
    """Execution contract for one annotation benchmark regime."""

    name: str
    n_folds: int
    seed: int
    val_size: float
    test_size: float | None = None
    label_fractions: tuple[float, ...] = ()
    cross_study_folds: tuple[CrossStudyFoldSpec, ...] = ()


@dataclass(frozen=True, slots=True)
class AnnotationTaskSpec:
    """Top-level task contract for the annotation paper pillar."""

    task_name: str
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...]
    efficiency_metrics: tuple[str, ...]
    datasets: tuple[AnnotationDatasetSpec, ...]
    regimes: tuple[AnnotationRegimeSpec, ...]


PBMC_ANNOTATION_DATASET = AnnotationDatasetSpec(
    name="pbmc68k_reduced",
    label_key="bulk_labels",
    batch_key=None,
    regimes=("full_label", "low_label"),
)

PANCREAS_ANNOTATION_DATASET = AnnotationDatasetSpec(
    name="openproblems_human_pancreas",
    label_key="cell_type",
    batch_key="batch",
    regimes=("full_label", "low_label", "cross_study"),
)

FULL_LABEL_REGIME = AnnotationRegimeSpec(
    name="full_label",
    n_folds=5,
    seed=42,
    val_size=0.15,
    test_size=0.15,
)

LOW_LABEL_REGIME = AnnotationRegimeSpec(
    name="low_label",
    n_folds=5,
    seed=42,
    val_size=0.15,
    test_size=0.15,
    label_fractions=(0.01, 0.05, 0.10),
)

CROSS_STUDY_REGIME = AnnotationRegimeSpec(
    name="cross_study",
    n_folds=1,
    seed=42,
    val_size=0.15,
    cross_study_folds=(
        CrossStudyFoldSpec(name="plate_like", held_out_batches=("smartseq2", "smarter")),
        CrossStudyFoldSpec(name="celseq_family", held_out_batches=("celseq", "celseq2")),
        CrossStudyFoldSpec(
            name="droplet_family",
            held_out_batches=("inDrop1", "inDrop2", "inDrop3", "inDrop4", "fluidigmc1"),
        ),
    ),
)

ANNOTATION_TASK_SPEC = AnnotationTaskSpec(
    task_name="annotation",
    primary_metrics=("macro_f1", "accuracy", "balanced_accuracy"),
    secondary_metrics=("auroc_ovr", "confusion_matrix", "batch_metrics"),
    efficiency_metrics=(
        "runtime_sec",
        "trainable_parameters",
        "total_parameters",
        "peak_memory_mb",
        "checkpoint_size_bytes",
    ),
    datasets=(PBMC_ANNOTATION_DATASET, PANCREAS_ANNOTATION_DATASET),
    regimes=(FULL_LABEL_REGIME, LOW_LABEL_REGIME, CROSS_STUDY_REGIME),
)

ANNOTATION_DATASET_SPECS = {
    PBMC_ANNOTATION_DATASET.name: PBMC_ANNOTATION_DATASET,
    PANCREAS_ANNOTATION_DATASET.name: PANCREAS_ANNOTATION_DATASET,
}

ANNOTATION_REGIME_SPECS = {
    FULL_LABEL_REGIME.name: FULL_LABEL_REGIME,
    LOW_LABEL_REGIME.name: LOW_LABEL_REGIME,
    CROSS_STUDY_REGIME.name: CROSS_STUDY_REGIME,
}


__all__ = [
    "ANNOTATION_DATASET_SPECS",
    "ANNOTATION_REGIME_SPECS",
    "ANNOTATION_TASK_SPEC",
    "AnnotationDatasetSpec",
    "AnnotationRegimeSpec",
    "AnnotationTaskSpec",
    "CrossStudyFoldSpec",
]
