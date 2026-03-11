"""Reconstruction plots."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt


def plot_reconstruction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    feature_index: int = 0,
    feature_name: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot predicted versus true values for one feature."""

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true[:, feature_index], y_pred[:, feature_index], alpha=0.6, s=18)
    name = feature_name or f"feature_{feature_index}"
    ax.set_xlabel(f"True {name}")
    ax.set_ylabel(f"Reconstructed {name}")
    ax.set_title("Reconstruction scatter")
    return fig, ax
