"""Evaluation utilities."""

from scdlkit.evaluation.compare import BenchmarkResult, compare_models
from scdlkit.evaluation.evaluator import evaluate_predictions
from scdlkit.evaluation.report import save_markdown_report, save_metrics_table

__all__ = [
    "BenchmarkResult",
    "compare_models",
    "evaluate_predictions",
    "save_markdown_report",
    "save_metrics_table",
]
