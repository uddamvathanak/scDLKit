"""Report export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def save_metrics_table(metrics: dict[str, Any], path: str | Path) -> Path:
    """Write scalar metrics to CSV."""

    output = Path(path)
    frame = pd.DataFrame([_scalar_metrics(metrics)])
    frame.to_csv(output, index=False)
    return output


def save_markdown_report(
    metrics: dict[str, Any],
    *,
    path: str | Path,
    title: str,
    extra_sections: list[str] | None = None,
) -> Path:
    """Write a markdown report with scalar and structured metrics."""

    output = Path(path)
    lines = [f"# {title}", "", "## Metrics", ""]
    for key, value in metrics.items():
        lines.append(f"- **{key}**: {value}")
    if extra_sections:
        lines.extend(["", *extra_sections])
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
