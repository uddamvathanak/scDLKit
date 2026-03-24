"""Internal cached dataset helpers for OpenProblems benchmarks."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
import anndata as ad
from anndata import AnnData, read_h5ad
from scipy import sparse
from sklearn.model_selection import train_test_split

from scdlkit.foundation.cache import get_dataset_cache_root

_OPENPROBLEMS_PANCREAS = "openproblems_v1/pancreas"
_OPENPROBLEMS_PANCREAS_URL = "https://ndownloader.figshare.com/files/36086813"
_OPENPROBLEMS_PANCREAS_SOURCE_URL = (
    "https://openproblems.bio/datasets/openproblems_v1/pancreas"
)
_OPENPROBLEMS_PANCREAS_BENCHMARK_URL = (
    "https://theislab.github.io/scib-reproducibility/dataset_pancreas.html"
)
_PROFILE_CONFIG: dict[str, dict[str, int]] = {
    "quickstart": {"max_cells": 512, "max_genes": 1024, "seed": 42},
    "full": {"max_cells": 2048, "max_genes": 2048, "seed": 42},
}
_TOP_CELL_TYPES = 8
_MIN_GENE_OVERLAP = 500
_LABEL_KEY_ALIASES = ("cell_type", "celltype")
_BATCH_KEY_ALIASES = ("batch", "tech")
_FEATURE_NAME_ALIASES = ("feature_name", "gene_symbol", "symbol")


@dataclass(frozen=True, slots=True)
class OpenProblemsDatasetSpec:
    dataset_id: str
    download_url: str
    source_url: str
    label_key: str
    batch_key: str | None
    organism: str


_DATASET_SPECS = {
    _OPENPROBLEMS_PANCREAS: OpenProblemsDatasetSpec(
        dataset_id=_OPENPROBLEMS_PANCREAS,
        download_url=_OPENPROBLEMS_PANCREAS_URL,
        source_url=_OPENPROBLEMS_PANCREAS_SOURCE_URL,
        label_key="cell_type",
        batch_key="batch",
        organism="homo_sapiens",
    ),
    "openproblems_human_pancreas": OpenProblemsDatasetSpec(
        dataset_id=_OPENPROBLEMS_PANCREAS,
        download_url=_OPENPROBLEMS_PANCREAS_URL,
        source_url=_OPENPROBLEMS_PANCREAS_SOURCE_URL,
        label_key="cell_type",
        batch_key="batch",
        organism="homo_sapiens",
    ),
}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_spec(dataset_id: str) -> OpenProblemsDatasetSpec:
    try:
        return _DATASET_SPECS[dataset_id]
    except KeyError as exc:
        msg = f"Unsupported OpenProblems dataset '{dataset_id}'."
        raise ValueError(msg) from exc


def _dataset_root(spec: OpenProblemsDatasetSpec, cache_dir: str | Path | None) -> Path:
    dataset_root = get_dataset_cache_root(cache_dir)
    dataset_path = dataset_root / "openproblems_v1" / "pancreas"
    dataset_path.mkdir(parents=True, exist_ok=True)
    return dataset_path


def _raw_dataset_path(spec: OpenProblemsDatasetSpec, cache_dir: str | Path | None) -> Path:
    return _dataset_root(spec, cache_dir) / "raw" / "dataset.h5ad"


def _raw_metadata_path(spec: OpenProblemsDatasetSpec, cache_dir: str | Path | None) -> Path:
    return _dataset_root(spec, cache_dir) / "raw" / "metadata.json"


def _processed_dataset_path(
    spec: OpenProblemsDatasetSpec,
    profile: Literal["quickstart", "full"],
    cache_dir: str | Path | None,
) -> Path:
    return (
        _dataset_root(spec, cache_dir)
        / "processed"
        / f"pancreas_annotation_{profile}.h5ad"
    )


def _processed_metadata_path(
    spec: OpenProblemsDatasetSpec,
    profile: Literal["quickstart", "full"],
    cache_dir: str | Path | None,
) -> Path:
    return (
        _dataset_root(spec, cache_dir)
        / "processed"
        / f"pancreas_annotation_{profile}.json"
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_h5ad_with_nullable_strings(adata: AnnData, path: Path) -> None:
    previous = ad.settings.allow_write_nullable_strings
    ad.settings.allow_write_nullable_strings = True
    try:
        adata.write_h5ad(path)
    finally:
        ad.settings.allow_write_nullable_strings = previous


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, destination.open("wb") as handle:  # noqa: S310
        shutil.copyfileobj(response, handle)


def _normalize_observed_organism(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized in {"human", "homo sapiens"}:
        return "homo_sapiens"
    return normalized.replace(" ", "_")


def _observed_organism(adata: AnnData) -> str | None:
    for key in ("organism", "dataset_organism", "species"):
        if key in adata.uns:
            observed = _normalize_observed_organism(adata.uns[key])
            if observed is not None:
                return observed
    for key in ("organism", "species"):
        if key in adata.obs:
            values = pd.Series(adata.obs[key]).dropna().astype(str).str.strip().str.lower().unique()
            if len(values) == 1:
                return _normalize_observed_organism(values[0])
    return None


def _validate_required_fields(adata: AnnData, spec: OpenProblemsDatasetSpec) -> None:
    missing: list[str] = []
    if spec.label_key not in adata.obs:
        missing.append(f"obs['{spec.label_key}']")
    if spec.batch_key is not None and spec.batch_key not in adata.obs:
        missing.append(f"obs['{spec.batch_key}']")
    if "counts" not in adata.layers:
        missing.append("layers['counts']")
    if "feature_name" not in adata.var.columns:
        missing.append("var['feature_name']")
    if missing:
        msg = (
            "OpenProblems pancreas dataset is missing required fields: "
            + ", ".join(missing)
            + "."
        )
        raise ValueError(msg)
    observed = _observed_organism(adata)
    if observed is not None and observed != spec.organism:
        msg = (
            f"OpenProblems dataset organism '{observed}' does not match the expected "
            f"'{spec.organism}'."
        )
        raise ValueError(msg)


def _resolve_obs_key(
    obs: pd.DataFrame,
    *,
    preferred: str,
    aliases: tuple[str, ...],
) -> str | None:
    for candidate in (preferred, *aliases):
        if candidate in obs.columns:
            return candidate
    return None


def _resolve_feature_names(adata: AnnData) -> pd.Index | None:
    for candidate in _FEATURE_NAME_ALIASES:
        if candidate in adata.var.columns:
            values = adata.var[candidate].astype(str)
            if not values.empty:
                return pd.Index(values, dtype="object")
    if adata.var_names.size:
        return pd.Index(adata.var_names.astype(str), dtype="object")
    return None


def _validate_download_schema(adata: AnnData, spec: OpenProblemsDatasetSpec) -> None:
    missing: list[str] = []
    label_source = _resolve_obs_key(
        adata.obs,
        preferred=spec.label_key,
        aliases=_LABEL_KEY_ALIASES,
    )
    if label_source is None:
        missing.append("obs['cell_type'] or obs['celltype']")
    if spec.batch_key is not None:
        batch_source = _resolve_obs_key(
            adata.obs,
            preferred=spec.batch_key,
            aliases=_BATCH_KEY_ALIASES,
        )
        if batch_source is None:
            missing.append("obs['batch'] or obs['tech']")
    if "counts" not in adata.layers:
        missing.append("layers['counts']")
    if _resolve_feature_names(adata) is None:
        missing.append("var['feature_name'] or var_names")
    if missing:
        msg = (
            "OpenProblems pancreas dataset is missing required fields after schema "
            "normalization: "
            + ", ".join(missing)
            + "."
        )
        raise ValueError(msg)
    observed = _observed_organism(adata)
    if observed is not None and observed != spec.organism:
        msg = (
            f"OpenProblems dataset organism '{observed}' does not match the expected "
            f"'{spec.organism}'."
        )
        raise ValueError(msg)


def _normalize_downloaded_adata(adata: AnnData, spec: OpenProblemsDatasetSpec) -> AnnData:
    _validate_download_schema(adata, spec)
    label_source = _resolve_obs_key(
        adata.obs,
        preferred=spec.label_key,
        aliases=_LABEL_KEY_ALIASES,
    )
    batch_source = (
        _resolve_obs_key(
            adata.obs,
            preferred=spec.batch_key,
            aliases=_BATCH_KEY_ALIASES,
        )
        if spec.batch_key is not None
        else None
    )
    feature_names = _resolve_feature_names(adata)
    if label_source is None or feature_names is None:
        msg = "OpenProblems pancreas dataset schema normalization did not resolve required fields."
        raise ValueError(msg)

    if label_source != spec.label_key:
        adata.obs[spec.label_key] = adata.obs[label_source].astype(str)
    if spec.batch_key is not None and batch_source is not None and batch_source != spec.batch_key:
        adata.obs[spec.batch_key] = adata.obs[batch_source].astype(str)
    if "feature_name" not in adata.var.columns:
        adata.var["feature_name"] = feature_names.to_numpy(dtype=object)
    if "feature_id" not in adata.var.columns:
        adata.var["feature_id"] = pd.Index(adata.var_names.astype(str), dtype="object").to_numpy(
            dtype=object
        )
    return adata


def _ensure_nonnegative_counts(counts: sparse.spmatrix | np.ndarray) -> None:
    if sparse.issparse(counts):
        sparse_counts = cast(sparse.spmatrix, counts)
        minimum = float(sparse_counts.min()) if sparse_counts.nnz else 0.0
    else:
        minimum = float(np.min(np.asarray(counts)))
    if minimum < 0:
        msg = "OpenProblems pancreas counts layer contains negative values."
        raise ValueError(msg)


def _column_variances(matrix: sparse.spmatrix | np.ndarray) -> np.ndarray:
    if sparse.issparse(matrix):
        sparse_matrix = cast(sparse.spmatrix, matrix).astype(np.float32, copy=False)
        mean = np.asarray(sparse_matrix.mean(axis=0)).ravel()
        mean_sq = np.asarray(sparse_matrix.power(2).mean(axis=0)).ravel()
        return mean_sq - np.square(mean)
    return np.var(np.asarray(matrix, dtype=np.float32), axis=0)


def _coerce_nullable_strings_frame(frame: pd.DataFrame) -> pd.DataFrame:
    coerced = frame.copy()
    if isinstance(coerced.index.array, pd.arrays.StringArray):
        coerced.index = pd.Index(
            coerced.index.astype(str).to_numpy(dtype=object),
            name=coerced.index.name,
            dtype=object,
        )
    for column in coerced.columns:
        series = coerced[column]
        if isinstance(series.array, pd.arrays.StringArray):
            coerced[column] = pd.Series(
                series.astype(str).to_numpy(dtype=object),
                index=coerced.index,
                dtype=object,
            )
    return coerced


def _top_cell_types(obs: pd.DataFrame, label_key: str, *, top_k: int) -> list[str]:
    counts = obs[label_key].astype(str).value_counts(sort=True)
    ordered = (
        counts.rename_axis("label")
        .reset_index(name="count")
        .sort_values(["count", "label"], ascending=[False, True], kind="mergesort")
    )
    return ordered.head(top_k)["label"].tolist()


def _sampling_labels(
    obs: pd.DataFrame,
    *,
    label_key: str,
    batch_key: str | None,
    max_cells: int,
) -> np.ndarray | None:
    labels = obs[label_key].astype(str)
    if batch_key is not None and batch_key in obs:
        combined = labels + "::" + obs[batch_key].astype(str)
        combined_counts = combined.value_counts()
        if combined_counts.min() >= 2 and combined_counts.index.size < max_cells:
            return combined.to_numpy(dtype=object)
    label_counts = labels.value_counts()
    if label_counts.min() >= 2 and label_counts.index.size < max_cells:
        return labels.to_numpy(dtype=object)
    return None


def _select_obs_indices(
    obs: pd.DataFrame,
    *,
    label_key: str,
    batch_key: str | None,
    max_cells: int,
    seed: int,
) -> np.ndarray:
    indices = np.arange(obs.shape[0], dtype=int)
    if obs.shape[0] <= max_cells:
        return indices
    stratify = _sampling_labels(obs, label_key=label_key, batch_key=batch_key, max_cells=max_cells)
    selected, _ = train_test_split(
        indices,
        train_size=max_cells,
        random_state=seed,
        stratify=stratify,
    )
    return np.sort(np.asarray(selected, dtype=int))


def _canonicalize_subset(subset: AnnData) -> AnnData:
    counts = subset.layers["counts"]
    _ensure_nonnegative_counts(counts)
    if sparse.issparse(counts):
        x_matrix = counts.copy()
    else:
        x_matrix = np.asarray(counts, dtype=np.float32).copy()
    var = subset.var.copy()
    feature_names = var["feature_name"].astype(str)
    var["feature_name"] = feature_names.to_numpy(dtype=object)
    if "feature_id" not in var.columns:
        var["feature_id"] = subset.var_names.astype(str).to_numpy(dtype=object)
    else:
        var["feature_id"] = var["feature_id"].astype(str).to_numpy(dtype=object)
    obs = subset.obs.copy()
    obs["cell_type"] = obs["cell_type"].astype(str).to_numpy(dtype=object)
    if "batch" in obs.columns:
        obs["batch"] = obs["batch"].astype(str).to_numpy(dtype=object)
    canonical = AnnData(
        X=x_matrix,
        obs=_coerce_nullable_strings_frame(obs),
        var=_coerce_nullable_strings_frame(var),
    )
    if sparse.issparse(x_matrix):
        canonical.layers["counts"] = x_matrix.copy()
    else:
        canonical.layers["counts"] = np.asarray(x_matrix).copy()
    canonical.var_names = pd.Index(feature_names.to_numpy(dtype=object), dtype=object)
    canonical.var_names_make_unique()
    canonical.var = _coerce_nullable_strings_frame(canonical.var)
    canonical.raw = canonical.copy()
    return canonical


def _validate_subset_gene_overlap(
    adata: AnnData,
    *,
    min_gene_overlap: int,
) -> None:
    from scdlkit.foundation import prepare_scgpt_data

    prepare_scgpt_data(
        adata,
        checkpoint="whole-human",
        label_key="cell_type",
        batch_size=32,
        min_gene_overlap=min_gene_overlap,
    )


def _build_processed_subset(
    raw_path: Path,
    *,
    spec: OpenProblemsDatasetSpec,
    profile: Literal["quickstart", "full"],
    cache_dir: str | Path | None,
) -> tuple[AnnData, dict[str, object]]:
    profile_config = _PROFILE_CONFIG[profile]
    raw_adata = _normalize_downloaded_adata(read_h5ad(raw_path), spec)
    _validate_required_fields(raw_adata, spec)

    top_cell_types = _top_cell_types(raw_adata.obs, spec.label_key, top_k=_TOP_CELL_TYPES)
    filtered = raw_adata[raw_adata.obs[spec.label_key].astype(str).isin(top_cell_types)].copy()
    selected_obs = _select_obs_indices(
        filtered.obs,
        label_key=spec.label_key,
        batch_key=spec.batch_key,
        max_cells=profile_config["max_cells"],
        seed=profile_config["seed"],
    )
    filtered = filtered[selected_obs].copy()

    counts = filtered.layers["counts"]
    gene_variances = _column_variances(counts)
    max_genes = min(profile_config["max_genes"], filtered.n_vars)
    keep_indices = np.argsort(gene_variances)[-max_genes:]
    filtered = filtered[:, np.sort(keep_indices)].copy()

    canonical = _canonicalize_subset(filtered)
    _validate_subset_gene_overlap(
        canonical,
        min_gene_overlap=min(_MIN_GENE_OVERLAP, canonical.n_vars),
    )

    label_counts = canonical.obs[spec.label_key].astype(str).value_counts()
    batch_count = (
        int(canonical.obs[spec.batch_key].astype(str).nunique())
        if spec.batch_key is not None and spec.batch_key in canonical.obs
        else 0
    )
    metadata = {
        "dataset_id": spec.dataset_id,
        "profile": profile,
        "source_url": spec.source_url,
        "benchmark_source_url": _OPENPROBLEMS_PANCREAS_BENCHMARK_URL,
        "selected_cell_types": sorted(label_counts.index.tolist()),
        "num_cells": int(canonical.n_obs),
        "num_genes": int(canonical.n_vars),
        "min_class_count": int(label_counts.min()),
        "num_batches": batch_count,
        "seed": int(profile_config["seed"]),
        "label_key": spec.label_key,
        "batch_key": spec.batch_key,
    }
    return canonical, metadata


def ensure_openproblems_dataset(
    dataset_id: str,
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> Path:
    """Download a supported OpenProblems dataset into the shared cache if needed."""

    spec = _resolve_spec(dataset_id)
    raw_path = _raw_dataset_path(spec, cache_dir)
    metadata_path = _raw_metadata_path(spec, cache_dir)
    if raw_path.exists() and not force_download:
        return raw_path

    temp_path = raw_path.parent / ".dataset.download"
    if temp_path.exists():
        temp_path.unlink()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _download_file(spec.download_url, temp_path)
    except URLError as exc:
        msg = f"Failed to download OpenProblems dataset '{spec.dataset_id}'."
        raise RuntimeError(msg) from exc
    if raw_path.exists():
        raw_path.unlink()
    temp_path.replace(raw_path)

    try:
        downloaded = read_h5ad(raw_path)
        _validate_download_schema(downloaded, spec)
        _ensure_nonnegative_counts(downloaded.layers["counts"])
    except Exception:
        raw_path.unlink(missing_ok=True)
        raise

    _write_json(
        metadata_path,
        {
            "dataset_id": spec.dataset_id,
            "download_url": spec.download_url,
            "source_url": spec.source_url,
            "benchmark_source_url": _OPENPROBLEMS_PANCREAS_BENCHMARK_URL,
            "downloaded_at": _utc_timestamp(),
            "file_size": raw_path.stat().st_size,
        },
    )
    return raw_path


def load_openproblems_pancreas_annotation_dataset(
    *,
    profile: Literal["quickstart", "full"],
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    force_rebuild: bool = False,
) -> AnnData:
    """Load a deterministic cached OpenProblems human-pancreas annotation subset."""

    if profile not in _PROFILE_CONFIG:
        msg = f"Unsupported pancreas annotation profile '{profile}'."
        raise ValueError(msg)
    spec = _resolve_spec(_OPENPROBLEMS_PANCREAS)
    raw_path = ensure_openproblems_dataset(
        spec.dataset_id,
        cache_dir=cache_dir,
        force_download=force_download,
    )
    processed_path = _processed_dataset_path(spec, profile, cache_dir)
    metadata_path = _processed_metadata_path(spec, profile, cache_dir)
    if processed_path.exists() and metadata_path.exists() and not force_rebuild:
        return read_h5ad(processed_path)

    processed, metadata = _build_processed_subset(
        raw_path,
        spec=spec,
        profile=profile,
        cache_dir=cache_dir,
    )
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    _write_h5ad_with_nullable_strings(processed, processed_path)
    _write_json(metadata_path, metadata)
    return processed
