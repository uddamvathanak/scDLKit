"""AnnData preparation and transformation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from anndata import AnnData
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, StandardScaler

from scdlkit.data.schemas import PreparedData, SplitData
from scdlkit.data.splits import build_splits


def _require_scanpy() -> Any:
    try:
        import scanpy as sc
    except ImportError as exc:
        msg = "scanpy-backed preprocessing requires `pip install scdlkit[scanpy]`."
        raise ImportError(msg) from exc
    return sc


def _encode_obs(values: np.ndarray | None) -> tuple[np.ndarray | None, dict[str, int] | None]:
    if values is None:
        return None, None
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(values.astype(str))
    mapping = {label: int(index) for index, label in enumerate(encoder.classes_)}
    return encoded, mapping


def _transform_obs(values: np.ndarray | None, mapping: dict[str, int] | None) -> np.ndarray | None:
    if values is None or mapping is None:
        return None
    encoded = np.empty(values.shape[0], dtype=int)
    for index, value in enumerate(values.astype(str)):
        if value not in mapping:
            msg = f"Encountered unseen label '{value}' during transform."
            raise ValueError(msg)
        encoded[index] = mapping[value]
    return encoded


def _extract_matrix(adata: AnnData, layer: str) -> Any:
    if layer == "X":
        return adata.X
    if layer not in adata.layers:
        msg = f"Layer '{layer}' not found in AnnData.layers."
        raise ValueError(msg)
    return adata.layers[layer]


def _to_split_data(
    x_matrix: Any,
    indices: np.ndarray,
    *,
    labels: np.ndarray | None,
    batches: np.ndarray | None,
    obs_names: list[str],
) -> SplitData:
    return SplitData(
        X=x_matrix[indices],
        labels=labels[indices] if labels is not None else None,
        batches=batches[indices] if batches is not None else None,
        obs_names=[obs_names[index] for index in indices],
    )


def _prepare_matrix(
    adata: AnnData,
    *,
    layer: str,
    use_hvg: bool,
    n_top_genes: int,
    normalize: bool,
    log1p: bool,
    scale: bool,
) -> tuple[AnnData, Any, list[str], StandardScaler | None]:
    working = adata
    if normalize or log1p or use_hvg:
        sc = _require_scanpy()
        if normalize:
            sc.pp.normalize_total(working)
        if log1p:
            sc.pp.log1p(working)
        if use_hvg:
            sc.pp.highly_variable_genes(working, n_top_genes=n_top_genes, subset=True)
    x_matrix = _extract_matrix(working, layer)
    scaler: StandardScaler | None = None
    if scale:
        scaler = StandardScaler(with_mean=not sparse.issparse(x_matrix))
        x_matrix = scaler.fit_transform(x_matrix)
        if sparse.issparse(x_matrix):
            x_matrix = x_matrix.tocsr()
    feature_names = working.var_names.astype(str).tolist()
    return working, x_matrix, feature_names, scaler


def prepare_data(
    adata: AnnData,
    *,
    layer: str = "X",
    use_hvg: bool = False,
    n_top_genes: int = 2000,
    normalize: bool = False,
    log1p: bool = False,
    scale: bool = False,
    label_key: str | None = None,
    batch_key: str | None = None,
    val_size: float = 0.15,
    test_size: float = 0.15,
    batch_aware_split: bool = False,
    random_state: int = 42,
    copy: bool = True,
) -> PreparedData:
    """Prepare AnnData splits and preprocessing metadata.

    Parameters
    ----------
    adata
        Input AnnData object.
    layer
        Matrix layer to read. ``"X"`` uses ``adata.X``.
    use_hvg
        Whether to run Scanpy highly-variable-gene selection.
    n_top_genes
        Number of genes retained when ``use_hvg=True``.
    normalize
        Whether to run Scanpy total-count normalization.
    log1p
        Whether to run Scanpy ``log1p`` transformation.
    scale
        Whether to standardize features.
    label_key
        Observation column used for labels and supervised metrics.
    batch_key
        Observation column used for batch-aware splitting and optional metrics.
    val_size
        Validation split fraction.
    test_size
        Test split fraction.
    batch_aware_split
        Whether to keep batch groups together when splitting.
    random_state
        Random seed used for deterministic splits.
    copy
        Whether to copy the input AnnData before preprocessing.

    Returns
    -------
    PreparedData
        Prepared train/validation/test splits plus preprocessing metadata.

    Raises
    ------
    ValueError
        If requested label or batch columns are missing.
    ImportError
        If Scanpy-backed preprocessing is requested without the scanpy extra.
    """

    working = adata.copy() if copy else adata
    working, x_matrix, feature_names, scaler = _prepare_matrix(
        working,
        layer=layer,
        use_hvg=use_hvg,
        n_top_genes=n_top_genes,
        normalize=normalize,
        log1p=log1p,
        scale=scale,
    )
    labels_raw = (
        working.obs[label_key].astype(str).to_numpy()
        if label_key is not None and label_key in working.obs
        else None
    )
    if label_key is not None and labels_raw is None:
        msg = f"label_key '{label_key}' not found in adata.obs."
        raise ValueError(msg)
    batches_raw = (
        working.obs[batch_key].astype(str).to_numpy()
        if batch_key is not None and batch_key in working.obs
        else None
    )
    if batch_key is not None and batches_raw is None:
        msg = f"batch_key '{batch_key}' not found in adata.obs."
        raise ValueError(msg)
    labels, label_encoder = _encode_obs(labels_raw)
    batches, batch_encoder = _encode_obs(batches_raw)

    split_indices = build_splits(
        working.n_obs,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if label_key is not None else None,
        groups=batches if batch_aware_split and batch_key is not None else None,
    )
    obs_names = working.obs_names.astype(str).tolist()
    train = _to_split_data(
        x_matrix,
        split_indices.train,
        labels=labels,
        batches=batches,
        obs_names=obs_names,
    )
    val = (
        _to_split_data(
            x_matrix,
            split_indices.val,
            labels=labels,
            batches=batches,
            obs_names=obs_names,
        )
        if split_indices.val.size
        else None
    )
    test = (
        _to_split_data(
            x_matrix,
            split_indices.test,
            labels=labels,
            batches=batches,
            obs_names=obs_names,
        )
        if split_indices.test.size
        else None
    )
    preprocessing = {
        "layer": layer,
        "use_hvg": use_hvg,
        "n_top_genes": n_top_genes,
        "normalize": normalize,
        "log1p": log1p,
        "scale": scale,
        "scaler": scaler,
        "feature_names": feature_names,
        "label_key": label_key,
        "batch_key": batch_key,
        "batch_aware_split": batch_aware_split,
    }
    return PreparedData(
        train=train,
        val=val,
        test=test,
        input_dim=int(x_matrix.shape[1]),
        feature_names=feature_names,
        label_encoder=label_encoder,
        batch_encoder=batch_encoder,
        preprocessing=preprocessing,
    )


def transform_adata(
    adata: AnnData,
    preprocessing: dict[str, Any],
    *,
    label_encoder: dict[str, int] | None = None,
    batch_encoder: dict[str, int] | None = None,
    copy: bool = True,
) -> SplitData:
    """Transform new AnnData with previously stored preprocessing metadata.

    Parameters
    ----------
    adata
        New AnnData object to transform.
    preprocessing
        Metadata emitted by :func:`prepare_data`.
    label_encoder
        Optional fitted label mapping from :func:`prepare_data`.
    batch_encoder
        Optional fitted batch mapping from :func:`prepare_data`.
    copy
        Whether to copy the AnnData before transformation.

    Returns
    -------
    SplitData
        A transformed dataset that can be passed to scDLKit inference utilities.

    Raises
    ------
    ValueError
        If required features are missing or unseen labels are encountered.
    """

    working = adata.copy() if copy else adata
    working, x_matrix, _, _ = _prepare_matrix(
        working,
        layer=preprocessing["layer"],
        use_hvg=preprocessing["use_hvg"],
        n_top_genes=preprocessing["n_top_genes"],
        normalize=preprocessing["normalize"],
        log1p=preprocessing["log1p"],
        scale=False,
    )
    feature_names = preprocessing["feature_names"]
    if list(working.var_names.astype(str)) != feature_names:
        missing = sorted(set(feature_names) - set(working.var_names.astype(str)))
        if missing:
            msg = f"AnnData is missing required features: {missing[:5]}"
            raise ValueError(msg)
        working = working[:, feature_names].copy()
        x_matrix = _extract_matrix(working, preprocessing["layer"])
    scaler = preprocessing.get("scaler")
    if scaler is not None:
        x_matrix = scaler.transform(x_matrix)
        if sparse.issparse(x_matrix):
            x_matrix = x_matrix.tocsr()
    labels = None
    if preprocessing["label_key"] is not None and preprocessing["label_key"] in working.obs:
        labels = _transform_obs(
            working.obs[preprocessing["label_key"]].astype(str).to_numpy(),
            label_encoder,
        )
    batches = None
    if preprocessing["batch_key"] is not None and preprocessing["batch_key"] in working.obs:
        batches = _transform_obs(
            working.obs[preprocessing["batch_key"]].astype(str).to_numpy(),
            batch_encoder,
        )
    return SplitData(
        X=x_matrix,
        labels=labels,
        batches=batches,
        obs_names=working.obs_names.astype(str).tolist(),
    )
