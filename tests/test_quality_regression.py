from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

quality_suite = importlib.import_module("run_quality_suite")


def _metrics_frame(
    *,
    silhouette: tuple[float, float, float] = (0.20, 0.18, 0.19),
    knn: tuple[float, float, float] = (0.90, 0.88, 0.89),
    pearson: tuple[float, float, float] = (0.22, 0.24, 0.21),
    accuracy: float = 0.91,
    macro_f1: float = 0.86,
) -> pd.DataFrame:
    rows = [
        {
            "dataset": "pbmc3k_processed",
            "task": "representation",
            "model": "vae",
            "seed": seed,
            "silhouette": silhouette[index],
            "knn_label_consistency": knn[index],
            "pearson": pearson[index],
        }
        for index, seed in enumerate((42, 52, 62))
    ]
    rows.append(
        {
            "dataset": "pbmc3k_processed",
            "task": "classification",
            "model": "mlp_classifier",
            "seed": 42,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }
    )
    return pd.DataFrame.from_records(rows)


def test_quality_gates_pass_for_healthy_metrics() -> None:
    summary = quality_suite.build_summary(_metrics_frame(), profile="ci")
    assert summary["gates"]["passed"] is True
    assert summary["gates"]["issues"] == []


def test_quality_gates_fail_for_regression() -> None:
    summary = quality_suite.build_summary(
        _metrics_frame(
            silhouette=(0.04, 0.05, 0.04),
            knn=(0.72, 0.71, 0.70),
            pearson=(0.08, 0.10, 0.09),
            accuracy=0.72,
            macro_f1=0.68,
        ),
        profile="ci",
    )
    assert summary["gates"]["passed"] is False
    assert any("silhouette" in issue.lower() for issue in summary["gates"]["issues"])
    assert any("accuracy" in issue.lower() for issue in summary["gates"]["issues"])
