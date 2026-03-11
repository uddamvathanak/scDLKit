"""Training history visualization."""

from __future__ import annotations

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_losses(history: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    """Plot training and validation loss curves."""

    fig, ax = plt.subplots(figsize=(7, 4))
    if "train_loss" in history:
        sns.lineplot(data=history, x="epoch", y="train_loss", ax=ax, label="train")
    if "val_loss" in history:
        sns.lineplot(data=history, x="epoch", y="val_loss", ax=ax, label="val")
    ax.set_title("Training history")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    return fig, ax
