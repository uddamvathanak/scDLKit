"""Task-aware evaluation entrypoints."""

from __future__ import annotations

from typing import Any

import numpy as np

from scdlkit.evaluation.metrics import (
    classification_metrics,
    reconstruction_metrics,
    representation_metrics,
)


def evaluate_predictions(task: str, predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    """Evaluate model predictions for a task."""

    if task == "classification":
        if "y" not in predictions:
            msg = "Classification evaluation requires encoded labels."
            raise ValueError(msg)
        return classification_metrics(predictions["y"], predictions["logits"])

    metrics = reconstruction_metrics(predictions["x"], predictions["reconstruction"])
    if task == "representation":
        metrics.update(
            representation_metrics(
                predictions.get("latent", np.empty((predictions["x"].shape[0], 0))),
                predictions.get("y"),
                predictions.get("batch"),
            )
        )
    return metrics
