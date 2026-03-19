"""Classification plots."""

from __future__ import annotations

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_confusion_matrix(
    confusion: list[list[int]] | np.ndarray,
    *,
    class_names: list[str] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a confusion matrix heatmap."""

    matrix = np.asarray(confusion)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    tick_labels: list[str] | None = None
    if class_names is not None and len(class_names) >= matrix.shape[0]:
        tick_labels = [str(name) for name in class_names[: matrix.shape[0]]]
    elif class_names is None:
        tick_labels = None
    else:
        tick_labels = [str(index) for index in range(matrix.shape[0])]
    if tick_labels is not None:
        tick_positions = np.arange(matrix.shape[0]) + 0.5
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticklabels(tick_labels, rotation=0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    return fig, ax
