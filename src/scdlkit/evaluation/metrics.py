"""Metric helpers for reconstruction, representation, and classification."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


def _safe_correlation(
    func: Callable[[np.ndarray, np.ndarray], tuple[float, float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    flat_true = np.ravel(y_true)
    flat_pred = np.ravel(y_pred)
    if np.std(flat_true) == 0 or np.std(flat_pred) == 0:
        return 0.0
    corr, _ = func(flat_true, flat_pred)
    return 0.0 if math.isnan(corr) else float(corr)


def reconstruction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_true - y_pred
    return {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "pearson": _safe_correlation(pearsonr, y_true, y_pred),
        "spearman": _safe_correlation(spearmanr, y_true, y_pred),
    }


def knn_label_consistency(latent: np.ndarray, labels: np.ndarray, n_neighbors: int = 10) -> float:
    if len(np.unique(labels)) < 2 or latent.shape[0] <= 1:
        return 0.0
    neighbors = min(n_neighbors + 1, latent.shape[0])
    knn = NearestNeighbors(n_neighbors=neighbors)
    knn.fit(latent)
    indices = knn.kneighbors(latent, return_distance=False)[:, 1:]
    votes = labels[indices]
    majority = np.array([np.bincount(row).argmax() for row in votes])
    return float(np.mean(majority == labels))


def representation_metrics(
    latent: np.ndarray,
    labels: np.ndarray | None,
    batches: np.ndarray | None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    unique_labels = np.unique(labels) if labels is not None else np.array([])
    unique_batches = np.unique(batches) if batches is not None else np.array([])
    if labels is not None and latent.shape[0] > len(unique_labels) and len(unique_labels) > 1:
        metrics["silhouette"] = float(silhouette_score(latent, labels))
        metrics["knn_label_consistency"] = knn_label_consistency(latent, labels)
        kmeans = KMeans(n_clusters=len(unique_labels), random_state=42, n_init="auto")
        clusters = kmeans.fit_predict(latent)
        metrics["ari"] = float(adjusted_rand_score(labels, clusters))
        metrics["nmi"] = float(normalized_mutual_info_score(labels, clusters))
    if batches is not None and latent.shape[0] > len(unique_batches) and len(unique_batches) > 1:
        metrics["batch_silhouette"] = float(silhouette_score(latent, batches))
    return metrics


def classification_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict[str, object]:
    predicted = logits.argmax(axis=1)
    labels = np.arange(logits.shape[1]) if logits.ndim == 2 else None
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=labels).tolist(),
    }
