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
            {"name": "pbmc_model_comparison"},
            {"name": "pbmc_classification"},
            {"name": "custom_model_extension"},
            {"name": "synthetic_smoke"},
        ],
        "runtime": {
            "total_sec": 120.0,
            "budget_sec": 480.0,
            "passed": runtime_passed,
            "notebook_count": 5,
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
    include_paul15: bool = True,
    include_transformer: bool = True,
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
