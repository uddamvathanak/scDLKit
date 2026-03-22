from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

quality_suite = importlib.import_module("run_quality_suite")


def _make_artifact_dir(tmp_path: Path, filenames: tuple[str, ...]) -> Path:
    artifact_dir = tmp_path / f"artifacts_{len(list(tmp_path.iterdir()))}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        (artifact_dir / filename).write_text("ok", encoding="utf-8")
    return artifact_dir


def _tutorial_summary(*, passed: bool = True, runtime_passed: bool = True) -> dict[str, object]:
    missing_files: list[str] = [] if passed else ["artifacts/pbmc_compare/benchmark_metrics.csv"]
    return {
        "validated": True,
        "passed": passed,
        "notebooks": [
            {"name": "scanpy_pbmc_quickstart"},
            {"name": "downstream_scanpy_after_scdlkit"},
            {"name": "pbmc_model_comparison"},
            {"name": "reconstruction_sanity_pbmc"},
            {"name": "pbmc_classification"},
            {"name": "custom_model_extension"},
            {"name": "scgpt_pbmc_embeddings"},
            {"name": "scgpt_cell_type_annotation"},
            {"name": "scgpt_dataset_specific_annotation"},
            {"name": "synthetic_smoke"},
        ],
        "runtime": {
            "total_sec": 120.0,
            "budget_sec": 480.0,
            "passed": runtime_passed,
            "notebook_count": 10,
        },
        "artifact_checks": {
            "passed": passed,
            "missing_files": missing_files,
        },
        "issues": [] if passed and runtime_passed else ["tutorial regression"],
    }


def _metrics_frame(
    tmp_path: Path,
    *,
    silhouette: tuple[float, float, float] = (0.20, 0.18, 0.19),
    knn: tuple[float, float, float] = (0.90, 0.88, 0.89),
    pearson: tuple[float, float, float] = (0.22, 0.24, 0.21),
    accuracy: float = 0.91,
    macro_f1: float = 0.86,
    scgpt_pbmc_probe_macro_f1: float = 0.93,
    scgpt_pbmc_probe_accuracy: float = 0.94,
    scgpt_pbmc_silhouette: float = 0.18,
    scgpt_pbmc68k_probe_macro_f1: float = 0.81,
    scgpt_pbmc68k_probe_accuracy: float = 0.82,
    scgpt_pbmc68k_silhouette: float = 0.16,
    scgpt_head_accuracy: float = 0.88,
    scgpt_head_macro_f1: float = 0.84,
    scgpt_lora_accuracy: float = 0.87,
    scgpt_lora_macro_f1: float = 0.83,
    pancreas_head_macro_f1: float = 0.74,
    pancreas_lora_macro_f1: float = 0.76,
    pancreas_best_macro_f1: float = 0.78,
    include_paul15: bool = True,
    include_transformer: bool = True,
    include_scgpt: bool = True,
    include_foundation_annotation: bool = True,
    include_pancreas: bool = False,
) -> pd.DataFrame:
    rows = []
    rows.append(
        {
            "dataset": "pbmc3k_processed",
            "task": "representation",
            "model": "pca",
            "seed": 42,
            "runtime_sec": 0.1,
            "artifact_dir": str(
                _make_artifact_dir(tmp_path, ("report.md", "report.csv", "latent_umap.png"))
            ),
            "silhouette": 0.17,
            "probe_accuracy": 0.92,
            "probe_macro_f1": 0.90,
        }
    )
    rows.append(
        {
            "dataset": "pbmc3k_processed",
            "task": "representation",
            "model": "autoencoder",
            "seed": 42,
            "runtime_sec": 3.0,
            "artifact_dir": str(
                _make_artifact_dir(
                    tmp_path,
                    ("report.md", "report.csv", "loss_curve.png", "latent_umap.png"),
                )
            ),
        }
    )
    if include_transformer:
        rows.append(
            {
                "dataset": "pbmc3k_processed",
                "task": "representation",
                "model": "transformer_ae",
                "seed": 42,
                "runtime_sec": 7.5,
                "artifact_dir": str(
                    _make_artifact_dir(
                        tmp_path,
                        ("report.md", "report.csv", "loss_curve.png", "latent_umap.png"),
                    )
                ),
            }
        )
    for index, seed in enumerate((42, 52, 62)):
        rows.append(
            {
                "dataset": "pbmc3k_processed",
                "task": "representation",
                "model": "vae",
                "seed": seed,
                "runtime_sec": 6.0,
                "artifact_dir": str(
                    _make_artifact_dir(
                        tmp_path,
                        ("report.md", "report.csv", "loss_curve.png", "latent_umap.png"),
                    )
                ),
                "silhouette": silhouette[index],
                "knn_label_consistency": knn[index],
                "pearson": pearson[index],
            }
        )
    if include_scgpt:
        rows.extend(
            [
                {
                    "dataset": "pbmc3k_processed",
                    "task": "foundation",
                    "model": "pca_foundation",
                    "seed": 42,
                    "runtime_sec": 0.3,
                    "artifact_dir": str(
                        _make_artifact_dir(
                            tmp_path,
                            ("report.md", "report.csv", "latent_umap.png"),
                        )
                    ),
                    "silhouette": 0.16,
                    "probe_accuracy": 0.92,
                    "probe_macro_f1": 0.90,
                },
                {
                    "dataset": "pbmc3k_processed",
                    "task": "foundation",
                    "model": "scgpt_whole_human",
                    "seed": 42,
                    "runtime_sec": 9.0,
                    "artifact_dir": str(
                        _make_artifact_dir(
                            tmp_path,
                            ("report.md", "report.csv", "latent_umap.png"),
                        )
                    ),
                    "silhouette": scgpt_pbmc_silhouette,
                    "probe_accuracy": scgpt_pbmc_probe_accuracy,
                    "probe_macro_f1": scgpt_pbmc_probe_macro_f1,
                },
                {
                    "dataset": "pbmc68k_reduced",
                    "task": "foundation",
                    "model": "pca_foundation",
                    "seed": 42,
                    "runtime_sec": 0.3,
                    "artifact_dir": str(
                        _make_artifact_dir(
                            tmp_path,
                            ("report.md", "report.csv", "latent_umap.png"),
                        )
                    ),
                    "silhouette": 0.15,
                    "probe_accuracy": 0.80,
                    "probe_macro_f1": 0.79,
                },
                {
                    "dataset": "pbmc68k_reduced",
                    "task": "foundation",
                    "model": "scgpt_whole_human",
                    "seed": 42,
                    "runtime_sec": 11.0,
                    "artifact_dir": str(
                        _make_artifact_dir(
                            tmp_path,
                            ("report.md", "report.csv", "latent_umap.png"),
                        )
                    ),
                    "silhouette": scgpt_pbmc68k_silhouette,
                    "probe_accuracy": scgpt_pbmc68k_probe_accuracy,
                    "probe_macro_f1": scgpt_pbmc68k_probe_macro_f1,
                },
            ]
        )
    if include_foundation_annotation:
        rows.extend(
            [
                {
                    "dataset": "pbmc3k_processed",
                    "task": "foundation_annotation",
                    "model": "pca_logistic_annotation",
                    "seed": 42,
                    "runtime_sec": 0.6,
                    "artifact_dir": str(
                        _make_artifact_dir(
                            tmp_path,
                            (
                                "report.md",
                                "report.csv",
                                "batch_metrics.csv",
                                "confusion_matrix.png",
                                "latent_umap.png",
                            ),
                        )
                    ),
                    "accuracy": 0.81,
                    "macro_f1": 0.77,
                    "trainable_parameters": 0,
                },
                {
                    "dataset": "pbmc3k_processed",
                    "task": "foundation_annotation",
                    "model": "scgpt_frozen_probe",
                    "seed": 42,
                    "runtime_sec": 9.5,
                    "artifact_dir": str(
                        _make_artifact_dir(
                            tmp_path,
                            (
                                "report.md",
                                "report.csv",
                                "batch_metrics.csv",
                                "confusion_matrix.png",
                                "latent_umap.png",
                            ),
                        )
                    ),
                    "accuracy": 0.74,
                    "macro_f1": 0.69,
                    "trainable_parameters": 0,
                },
                {
                    "dataset": "pbmc3k_processed",
                    "task": "foundation_annotation",
                    "model": "scgpt_head",
                    "seed": 42,
                    "runtime_sec": 12.0,
                    "artifact_dir": str(
                        _make_artifact_dir(
                            tmp_path,
                            (
                                "report.md",
                                "report.csv",
                                "batch_metrics.csv",
                                "confusion_matrix.png",
                                "latent_umap.png",
                            ),
                        )
                    ),
                    "accuracy": scgpt_head_accuracy,
                    "macro_f1": scgpt_head_macro_f1,
                    "trainable_parameters": 195,
                },
                {
                    "dataset": "pbmc3k_processed",
                    "task": "foundation_annotation",
                    "model": "scgpt_lora",
                    "seed": 42,
                    "runtime_sec": 18.0,
                    "artifact_dir": str(
                        _make_artifact_dir(
                            tmp_path,
                            (
                                "report.md",
                                "report.csv",
                                "batch_metrics.csv",
                                "confusion_matrix.png",
                                "latent_umap.png",
                            ),
                        )
                    ),
                    "accuracy": scgpt_lora_accuracy,
                    "macro_f1": scgpt_lora_macro_f1,
                    "trainable_parameters": 731,
                },
            ]
        )
    if include_pancreas:
        rows.extend(
            [
                {
                    "dataset": "openproblems_human_pancreas",
                    "task": "foundation_annotation",
                    "model": "pca_logistic_annotation",
                    "seed": 42,
                    "runtime_sec": 1.2,
                    "artifact_dir": str(
                        _make_artifact_dir(
                            tmp_path,
                            (
                                "report.md",
                                "report.csv",
                                "batch_metrics.csv",
                                "confusion_matrix.png",
                                "latent_umap.png",
                            ),
                        )
                    ),
                    "accuracy": 0.72,
                    "macro_f1": pancreas_best_macro_f1,
                    "trainable_parameters": 0,
                },
                {
                    "dataset": "openproblems_human_pancreas",
                    "task": "foundation_annotation",
                    "model": "scgpt_frozen_probe",
                    "seed": 42,
                    "runtime_sec": 14.0,
                    "artifact_dir": str(
                        _make_artifact_dir(
                            tmp_path,
                            (
                                "report.md",
                                "report.csv",
                                "batch_metrics.csv",
                                "confusion_matrix.png",
                                "latent_umap.png",
                            ),
                        )
                    ),
                    "accuracy": 0.70,
                    "macro_f1": 0.73,
                    "trainable_parameters": 0,
                },
                {
                    "dataset": "openproblems_human_pancreas",
                    "task": "foundation_annotation",
                    "model": "scgpt_head",
                    "seed": 42,
                    "runtime_sec": 18.0,
                    "artifact_dir": str(
                        _make_artifact_dir(
                            tmp_path,
                            (
                                "report.md",
                                "report.csv",
                                "batch_metrics.csv",
                                "confusion_matrix.png",
                                "latent_umap.png",
                            ),
                        )
                    ),
                    "accuracy": 0.73,
                    "macro_f1": pancreas_head_macro_f1,
                    "trainable_parameters": 220,
                },
                {
                    "dataset": "openproblems_human_pancreas",
                    "task": "foundation_annotation",
                    "model": "scgpt_lora",
                    "seed": 42,
                    "runtime_sec": 26.0,
                    "artifact_dir": str(
                        _make_artifact_dir(
                            tmp_path,
                            (
                                "report.md",
                                "report.csv",
                                "batch_metrics.csv",
                                "confusion_matrix.png",
                                "latent_umap.png",
                            ),
                        )
                    ),
                    "accuracy": 0.75,
                    "macro_f1": pancreas_lora_macro_f1,
                    "trainable_parameters": 760,
                },
            ]
        )
    rows.append(
        {
            "dataset": "pbmc3k_processed",
            "task": "classification",
            "model": "mlp_classifier",
            "seed": 42,
            "runtime_sec": 2.0,
            "artifact_dir": str(
                _make_artifact_dir(
                    tmp_path,
                    ("report.md", "report.csv", "loss_curve.png", "confusion_matrix.png"),
                )
            ),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }
    )
    rows.append(
        {
            "dataset": "pbmc3k_processed",
            "task": "classification",
            "model": "logistic_regression_pca",
            "seed": 42,
            "runtime_sec": 0.2,
            "artifact_dir": str(
                _make_artifact_dir(tmp_path, ("report.md", "report.csv", "confusion_matrix.png"))
            ),
        }
    )
    if include_paul15:
        rows.extend(
            [
                {
                    "dataset": "paul15",
                    "task": "representation",
                    "model": "pca",
                    "seed": 42,
                    "runtime_sec": 0.2,
                    "artifact_dir": str(
                        _make_artifact_dir(
                            tmp_path,
                            ("report.md", "report.csv", "latent_umap.png"),
                        )
                    ),
                },
                {
                    "dataset": "paul15",
                    "task": "representation",
                    "model": "vae",
                    "seed": 42,
                    "runtime_sec": 8.0,
                    "artifact_dir": str(
                        _make_artifact_dir(
                            tmp_path,
                            ("report.md", "report.csv", "loss_curve.png", "latent_umap.png"),
                        )
                    ),
                },
            ]
        )
    return pd.DataFrame.from_records(rows)


def test_summary_marks_release_ready_for_healthy_inputs(tmp_path: Path) -> None:
    summary = quality_suite.build_summary(
        _metrics_frame(tmp_path),
        profile="ci",
        tutorial_summary=_tutorial_summary(),
    )
    assert summary["gates"]["passed"] is True
    assert summary["benchmark_ready"] is True
    assert summary["release_rc_ready"] is True
    assert summary["missing_runs"] == []
    assert summary["artifact_checks"]["benchmark"]["passed"] is True
    assert summary["artifact_checks"]["tutorials"]["passed"] is True


def test_missing_required_run_blocks_release_readiness(tmp_path: Path) -> None:
    summary = quality_suite.build_summary(
        _metrics_frame(tmp_path, include_transformer=False),
        profile="ci",
        tutorial_summary=_tutorial_summary(),
    )
    assert summary["benchmark_ready"] is False
    assert any("transformer_ae" in missing for missing in summary["missing_runs"])


def test_runtime_budget_regression_blocks_benchmark_ready(tmp_path: Path) -> None:
    metrics_frame = _metrics_frame(tmp_path)
    summary = quality_suite.build_summary(
        metrics_frame,
        profile="ci",
        suite_runtime_sec=400.0,
        tutorial_summary=_tutorial_summary(),
    )
    assert summary["benchmark_ready"] is False
    assert any("Quality-suite runtime exceeded" in issue for issue in summary["runtime"]["issues"])


def test_tutorial_artifact_failure_blocks_release_ready(tmp_path: Path) -> None:
    summary = quality_suite.build_summary(
        _metrics_frame(tmp_path),
        profile="ci",
        tutorial_summary=_tutorial_summary(passed=False),
    )
    assert summary["benchmark_ready"] is True
    assert summary["release_rc_ready"] is False
    assert summary["artifact_checks"]["tutorials"]["artifact_checks"]["passed"] is False


def test_tutorial_runtime_failure_does_not_relabel_benchmark_as_failed(tmp_path: Path) -> None:
    summary = quality_suite.build_summary(
        _metrics_frame(tmp_path),
        profile="ci",
        tutorial_summary=_tutorial_summary(passed=False, runtime_passed=False),
    )
    assert summary["benchmark_ready"] is True
    assert summary["release_rc_ready"] is False
    assert summary["runtime"]["tutorial_issues"]
    assert summary["runtime"]["benchmark_issues"] == []


def test_quality_gates_fail_for_regression_metrics(tmp_path: Path) -> None:
    summary = quality_suite.build_summary(
        _metrics_frame(
            tmp_path,
            silhouette=(0.04, 0.05, 0.04),
            knn=(0.72, 0.71, 0.70),
            pearson=(0.08, 0.10, 0.09),
            accuracy=0.72,
            macro_f1=0.68,
            scgpt_pbmc_probe_macro_f1=0.84,
            scgpt_pbmc_probe_accuracy=0.86,
            scgpt_pbmc_silhouette=0.10,
            scgpt_pbmc68k_probe_macro_f1=0.75,
            scgpt_pbmc68k_probe_accuracy=0.76,
            scgpt_pbmc68k_silhouette=0.11,
            scgpt_head_accuracy=0.76,
            scgpt_head_macro_f1=0.70,
            scgpt_lora_accuracy=0.72,
            scgpt_lora_macro_f1=0.68,
        ),
        profile="ci",
        tutorial_summary=_tutorial_summary(),
    )
    assert summary["gates"]["passed"] is False
    assert any("silhouette" in issue.lower() for issue in summary["gates"]["issues"])
    assert any("accuracy" in issue.lower() for issue in summary["gates"]["issues"])


def test_markdown_summary_includes_runtime_and_artifact_sections(tmp_path: Path) -> None:
    summary = quality_suite.build_summary(
        _metrics_frame(tmp_path),
        profile="ci",
        tutorial_summary=_tutorial_summary(),
    )
    markdown = quality_suite.render_summary_markdown(summary)
    assert "## Runtime" in markdown
    assert "## Artifact checks" in markdown
    assert "Release RC ready" in markdown


def test_missing_scgpt_run_blocks_release_readiness(tmp_path: Path) -> None:
    summary = quality_suite.build_summary(
        _metrics_frame(tmp_path, include_scgpt=False),
        profile="ci",
        tutorial_summary=_tutorial_summary(),
    )
    assert summary["benchmark_ready"] is False
    assert any("scgpt_whole_human" in missing for missing in summary["missing_runs"])


def test_missing_foundation_annotation_run_blocks_release_readiness(tmp_path: Path) -> None:
    summary = quality_suite.build_summary(
        _metrics_frame(tmp_path, include_foundation_annotation=False),
        profile="ci",
        tutorial_summary=_tutorial_summary(),
    )
    assert summary["benchmark_ready"] is False
    assert any("scgpt_head" in missing for missing in summary["missing_runs"])


def test_pancreas_gate_requires_a_tuned_strategy_close_to_the_best_result(tmp_path: Path) -> None:
    passing_summary = quality_suite.build_summary(
        _metrics_frame(tmp_path, include_pancreas=True),
        profile="full",
        tutorial_summary=_tutorial_summary(),
    )
    assert passing_summary["gates"]["passed"] is True

    failing_summary = quality_suite.build_summary(
        _metrics_frame(
            tmp_path,
            include_pancreas=True,
            pancreas_head_macro_f1=0.70,
            pancreas_lora_macro_f1=0.72,
            pancreas_best_macro_f1=0.78,
        ),
        profile="full",
        tutorial_summary=_tutorial_summary(),
    )
    assert failing_summary["gates"]["passed"] is False
    assert any(
        "OpenProblems human pancreas" in issue
        for issue in failing_summary["gates"]["issues"]
    )
