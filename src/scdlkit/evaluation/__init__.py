"""Public evaluation and report-export helpers.

This namespace collects the stable evaluation surfaces used throughout the
tutorials, benchmark scripts, and lower-level :class:`~scdlkit.training.trainer.Trainer`
workflows.

The most common entrypoints are:

- :func:`evaluate_predictions` for task-aware metric computation
- :func:`save_markdown_report` for human-readable report export
- :func:`save_metrics_table` for CSV-style metric export
- :func:`compare_models` for baseline benchmarking
"""

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
