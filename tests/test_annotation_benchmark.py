from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

annotation_benchmark = importlib.import_module("run_annotation_benchmark")


def _benchmark_adata() -> AnnData:
    rng = np.random.default_rng(11)
    genes = [f"gene_{index}" for index in range(12)]
    counts = rng.poisson(lam=3.0, size=(72, len(genes))).astype("float32")
    labels = np.array(
        ["alpha"] * 24 + ["beta"] * 24 + ["delta"] * 24,
        dtype=object,
    )
    batches = np.array(
        ["smartseq2"] * 8
        + ["smarter"] * 8
        + ["celseq"] * 8
        + ["celseq2"] * 8
        + ["inDrop1"] * 8
        + ["inDrop2"] * 8
        + ["fluidigmc1"] * 8
        + ["inDrop3"] * 8
        + ["inDrop4"] * 8,
        dtype=object,
    )
    obs = pd.DataFrame(
        {
            "cell_type": labels,
            "batch": batches,
        },
        index=[f"cell_{index}" for index in range(72)],
    )
    adata = AnnData(X=counts, obs=obs)
    adata.var_names = genes
    adata.raw = adata.copy()
    return adata


def test_build_full_label_split_is_deterministic() -> None:
    adata = _benchmark_adata()
    split_a = annotation_benchmark.build_full_label_split(
        adata,
        label_key="cell_type",
        seed=42,
    )
    split_b = annotation_benchmark.build_full_label_split(
        adata,
        label_key="cell_type",
        seed=42,
    )
    np.testing.assert_array_equal(split_a.train_indices, split_b.train_indices)
    np.testing.assert_array_equal(split_a.val_indices, split_b.val_indices)
    np.testing.assert_array_equal(split_a.test_indices, split_b.test_indices)


def test_build_low_label_split_is_deterministic_and_retains_all_classes() -> None:
    adata = _benchmark_adata()
    full_split = annotation_benchmark.build_full_label_split(
        adata,
        label_key="cell_type",
        seed=42,
    )
    low_a = annotation_benchmark.build_low_label_split(
        full_split,
        adata,
        label_key="cell_type",
        label_fraction=0.1,
        seed=42,
    )
    low_b = annotation_benchmark.build_low_label_split(
        full_split,
        adata,
        label_key="cell_type",
        label_fraction=0.1,
        seed=42,
    )
    np.testing.assert_array_equal(low_a.train_indices, low_b.train_indices)
    retained_labels = set(adata.obs.iloc[low_a.train_indices]["cell_type"])
    assert retained_labels == {"alpha", "beta", "delta"}


def test_build_cross_study_split_is_leakage_free() -> None:
    adata = _benchmark_adata()
    fold = annotation_benchmark.ANNOTATION_REGIME_SPECS["cross_study"].cross_study_folds[0]
    split = annotation_benchmark.build_cross_study_split(
        adata,
        label_key="cell_type",
        batch_key="batch",
        fold=fold,
        seed=42,
    )
    test_batches = set(adata.obs.iloc[split.test_indices]["batch"])
    train_batches = set(adata.obs.iloc[split.train_indices]["batch"])
    val_batches = set(adata.obs.iloc[split.val_indices]["batch"])
    assert test_batches == set(fold.held_out_batches)
    assert train_batches.isdisjoint(test_batches)
    assert val_batches.isdisjoint(test_batches)


def test_run_annotation_benchmark_writes_required_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_load_annotation_dataset(
        dataset_name: str, *, profile: str
    ) -> tuple[AnnData, str, str | None]:
        return (
            _benchmark_adata(),
            "cell_type",
            "batch" if dataset_name == "openproblems_human_pancreas" else None,
        )

    def fake_run_single_benchmark(
        *,
        dataset_name: str,
        adata: AnnData,
        label_key: str,
        batch_key: str | None,
        regime: str,
        split_plan,
        model_name: str,
        seed: int,
        profile_settings: dict[str, int],
        output_dir: Path,
    ) -> dict[str, object]:
        return {
            "dataset": dataset_name,
            "regime": regime,
            "model": model_name,
            "strategy": annotation_benchmark.MODEL_DISPLAY_NAMES[model_name],
            "seed": seed,
            "label_fraction": split_plan.label_fraction,
            "cross_study_fold": split_plan.cross_study_fold,
            "accuracy": 0.8,
            "macro_f1": 0.75,
            "balanced_accuracy": 0.74,
            "runtime_sec": 1.5,
            "trainable_parameters": 0 if model_name == "pca_logistic_annotation" else 10,
            "checkpoint_size_bytes": 0,
        }

    monkeypatch.setattr(
        annotation_benchmark, "_load_annotation_dataset", fake_load_annotation_dataset
    )
    monkeypatch.setattr(
        annotation_benchmark,
        "_run_single_benchmark",
        fake_run_single_benchmark,
    )

    output_dir = tmp_path / "annotation_pillar"
    annotation_benchmark.run_annotation_benchmark(
        datasets=("pbmc68k_reduced", "openproblems_human_pancreas"),
        regimes=("full_label", "low_label", "cross_study"),
        profile="quickstart",
        strategies=("pca_logistic_annotation", "scgpt_head"),
        output_dir=output_dir,
        seeds=(42,),
        label_fractions=(0.1,),
        cross_study_folds=("plate_like",),
    )

    assert (output_dir / "full_label" / "strategy_metrics.csv").exists()
    assert (output_dir / "low_label" / "strategy_metrics.csv").exists()
    assert (output_dir / "cross_study" / "strategy_metrics.csv").exists()
    assert (output_dir / "figures" / "annotation_performance.csv").exists()
    assert (output_dir / "figures" / "annotation_performance.png").exists()
    assert (output_dir / "figures" / "annotation_low_label_curves.csv").exists()
    assert (output_dir / "figures" / "annotation_low_label_curves.png").exists()
    assert (output_dir / "figures" / "annotation_cross_study.csv").exists()
    assert (output_dir / "figures" / "annotation_cross_study.png").exists()
    assert (output_dir / "figures" / "annotation_pareto.csv").exists()
    assert (output_dir / "figures" / "annotation_pareto.png").exists()
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "tutorial" / "manifest.json").exists()
