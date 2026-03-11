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
    if class_names is not None:
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names, rotation=0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    return fig, ax
