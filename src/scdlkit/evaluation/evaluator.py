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
    """Evaluate predictions produced by a scDLKit task.

    Parameters
    ----------
    task:
        Task name. Supported public values are ``"representation"``,
        ``"reconstruction"``, and ``"classification"``.
    predictions:
        Dictionary returned by :meth:`scdlkit.training.trainer.Trainer.predict_dataset`
        or an equivalent workflow. Expected keys depend on the task:

        - classification: ``"logits"`` and encoded labels under ``"y"``
        - reconstruction: ``"x"`` and ``"reconstruction"``
        - representation: ``"latent"`` and, when available, ``"y"`` or ``"batch"``

    Returns
    -------
    dict[str, Any]
        Metric dictionary appropriate for the requested task.

    Raises
    ------
    ValueError
        If the required arrays for the selected task are missing.

    Notes
    -----
    Representation evaluation reuses reconstruction metrics when both the input
    matrix and reconstructed values are available.
    """

    if task == "classification":
        if "y" not in predictions:
            msg = "Classification evaluation requires encoded labels."
            raise ValueError(msg)
        return classification_metrics(predictions["y"], predictions["logits"])

    metrics: dict[str, Any] = {}
    if "x" in predictions and "reconstruction" in predictions:
        metrics.update(reconstruction_metrics(predictions["x"], predictions["reconstruction"]))
    if task == "representation":
        if "latent" not in predictions:
            msg = "Representation evaluation requires latent embeddings."
            raise ValueError(msg)
        metrics.update(
            representation_metrics(
                predictions["latent"],
                predictions.get("y"),
                predictions.get("batch"),
            )
        )
    elif not metrics:
        msg = "Reconstruction evaluation requires input and reconstruction arrays."
        raise ValueError(msg)
    return metrics
