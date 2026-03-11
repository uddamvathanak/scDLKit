"""Model comparison plots."""

from __future__ import annotations

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_model_comparison(
    metrics_frame: pd.DataFrame,
    *,
    metric: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a metric comparison across models."""

    candidate_metric = metric
    if candidate_metric is None:
        for key in ("pearson", "silhouette", "accuracy", "mse"):
            if key in metrics_frame.columns:
                candidate_metric = key
                break
    if candidate_metric is None:
        msg = "No plottable metric found in metrics_frame."
        raise ValueError(msg)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=metrics_frame, x="model", y=candidate_metric, ax=ax)
    ax.set_title(f"Model comparison: {candidate_metric}")
    ax.set_xlabel("Model")
    ax.set_ylabel(candidate_metric)
    ax.tick_params(axis="x", rotation=30)
    return fig, ax
