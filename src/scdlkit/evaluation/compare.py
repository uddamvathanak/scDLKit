"""Compare multiple models on the same AnnData workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import pandas as pd

from scdlkit.evaluation.report import save_markdown_report
from scdlkit.utils import ensure_directory


@dataclass(slots=True)
class BenchmarkResult:
    """Collected results from comparing multiple models.

    Attributes
    ----------
    metrics_frame
        Tabular metrics summary with one row per compared model.
    runners
        Fitted :class:`scdlkit.runner.TaskRunner` objects keyed by model name.
    output_paths
        Optional output artifact paths when ``output_dir`` was provided.
    """

    metrics_frame: pd.DataFrame
    runners: dict[str, Any]
    output_paths: dict[str, str] = field(default_factory=dict)


def compare_models(
    adata: Any,
    *,
    models: list[str],
    task: str,
    shared_kwargs: dict[str, Any] | None = None,
    output_dir: str | None = None,
) -> BenchmarkResult:
    """Train and evaluate several models with shared configuration.

    Parameters
    ----------
    adata
        AnnData-like object passed to every compared run.
    models
        Built-in model names to compare.
    task
        Task name shared across all compared models.
    shared_kwargs
        Keyword arguments forwarded to each :class:`scdlkit.runner.TaskRunner`.
    output_dir
        Optional directory for CSV, Markdown, and comparison-plot artifacts.

    Returns
    -------
    BenchmarkResult
        Metrics table, fitted runners, and optional artifact paths.
    """

    from scdlkit.runner import TaskRunner
    from scdlkit.visualization.compare import plot_model_comparison

    shared = dict(shared_kwargs or {})
    records: list[dict[str, Any]] = []
    runners: dict[str, TaskRunner] = {}
    output_paths: dict[str, str] = {}
    for model_name in models:
        runner = TaskRunner(model=model_name, task=task, **shared)
        started_at = perf_counter()
        runner.fit(adata)
        metrics = runner.evaluate()
        runtime_sec = perf_counter() - started_at
        scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        records.append({"model": model_name, "runtime_sec": runtime_sec, **scalar_metrics})
        runners[model_name] = runner

    metrics_frame = pd.DataFrame.from_records(records).sort_values("model").reset_index(drop=True)
    if output_dir is not None:
        directory = ensure_directory(output_dir)
        csv_path = directory / "benchmark_metrics.csv"
        md_path = directory / "benchmark_report.md"
        png_path = directory / "benchmark_comparison.png"
        metrics_frame.to_csv(csv_path, index=False)
        fig, _ = plot_model_comparison(metrics_frame)
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        report_lines = ["## Compared models", "", *[f"- `{name}`" for name in models]]
        save_markdown_report(
            {"num_models": len(models), "task": task},
            path=md_path,
            title="Benchmark Report",
            extra_sections=report_lines,
        )
        output_paths = {
            "metrics_csv": str(csv_path),
            "report_md": str(md_path),
            "comparison_png": str(png_path),
        }
    return BenchmarkResult(metrics_frame=metrics_frame, runners=runners, output_paths=output_paths)
