"""Report export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def save_metrics_table(metrics: dict[str, Any], path: str | Path) -> Path:
    """Write scalar metrics to CSV.

    Parameters
    ----------
    metrics
        Metrics dictionary that may contain scalar and structured values.
    path
        CSV output path.

    Returns
    -------
    pathlib.Path
        The written CSV path.

    Notes
    -----
    Only scalar metrics are written to the CSV table. Structured values such as
    confusion matrices remain in the Markdown report or in the original metric
    dictionary.
    """

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
    """Write a Markdown report with scalar and structured metrics.

    Parameters
    ----------
    metrics
        Metrics dictionary to serialize.
    path
        Markdown output path.
    title
        Report title rendered as the first heading.
    extra_sections
        Optional additional Markdown lines appended after the metrics section.

    Returns
    -------
    pathlib.Path
        The written Markdown report path.

    Notes
    -----
    Markdown reports include both scalar and structured values. Extra sections
    are appended verbatim after the main metrics block.
    """

    output = Path(path)
    lines = [f"# {title}", "", "## Metrics", ""]
    for key, value in metrics.items():
        lines.append(f"- **{key}**: {value}")
    if extra_sections:
        lines.extend(["", *extra_sections])
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
