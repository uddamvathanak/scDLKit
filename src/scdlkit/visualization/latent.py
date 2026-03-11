"""Latent space visualizations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP


def plot_latent_embedding(
    latent: np.ndarray,
    *,
    color: np.ndarray | None = None,
    method: str = "umap",
) -> tuple[plt.Figure, plt.Axes]:
    """Reduce latent vectors to 2D and plot them."""

    reducer = UMAP(random_state=42) if method == "umap" else PCA(n_components=2, random_state=42)
    reduced = reducer.fit_transform(latent)
    frame = pd.DataFrame({"dim1": reduced[:, 0], "dim2": reduced[:, 1]})
    if color is not None:
        frame["color"] = color.astype(str)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=frame,
        x="dim1",
        y="dim2",
        hue="color" if color is not None else None,
        s=30,
        ax=ax,
    )
    ax.set_title(f"Latent {method.upper()}")
    return fig, ax
