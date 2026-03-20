"""Tokenized data preparation for scGPT embedding workflows."""

from __future__ import annotations

from collections.abc import Sized
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy import sparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from scdlkit.foundation.cache import DEFAULT_SCGPT_CHECKPOINT
from scdlkit.foundation.scgpt import GeneVocab, _load_scgpt_assets


def _digitize(x: np.ndarray, bins: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    left_digits = np.digitize(x, bins)
    right_digits = np.digitize(x, bins, right=True)
    random_offsets = rng.random(len(x))
    digits = np.ceil(random_offsets * (right_digits - left_digits) + left_digits).astype(np.int64)
    return digits


def _bin_row(
    values: torch.Tensor,
    n_bins: int,
    *,
    rng: np.random.Generator,
) -> torch.Tensor:
    row = values.detach().cpu().numpy()
    if row.size == 0:
        return values
    if float(row.max()) == 0.0:
        return torch.zeros_like(values)
    if float(row.min()) <= 0.0:
        non_zero_ids = np.nonzero(row)[0]
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        binned = np.zeros_like(row, dtype=np.int64)
        binned[non_zero_ids] = _digitize(non_zero_row, bins, rng=rng)
    else:
        bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
        binned = _digitize(row, bins, rng=rng)
    return torch.as_tensor(binned, dtype=values.dtype)


@dataclass(frozen=True, slots=True)
class ScGPTPreparedData:
    """Tokenized dataset and metadata for scGPT workflows."""

    dataset: Dataset[dict[str, torch.Tensor]]
    gene_names: tuple[str, ...]
    checkpoint_id: str
    label_key: str | None
    label_categories: tuple[str, ...] | None
    batch_size: int
    num_cells: int
    num_genes_matched: int


@dataclass(frozen=True, slots=True)
class ScGPTSplitData:
    """Train, validation, and test datasets for scGPT annotation workflows."""

    train: Dataset[dict[str, torch.Tensor]]
    val: Dataset[dict[str, torch.Tensor]] | None
    test: Dataset[dict[str, torch.Tensor]] | None
    checkpoint_id: str
    label_key: str | None
    label_categories: tuple[str, ...] | None
    gene_names: tuple[str, ...]
    batch_size: int
    num_cells: int
    num_genes_matched: int


@dataclass(frozen=True, slots=True)
class ScGPTAnnotationDataReport:
    """Compatibility summary for experimental scGPT annotation workflows."""

    checkpoint_id: str
    label_key: str
    num_cells: int
    num_input_genes: int
    num_genes_matched: int
    gene_overlap_ratio: float
    label_categories: tuple[str, ...]
    class_counts: dict[str, int]
    min_class_count: int
    stratify_possible: bool
    warnings: tuple[str, ...]


class _ScGPTTokenSourceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        matrix: Any,
        gene_ids: np.ndarray,
        labels: np.ndarray | None,
        cls_token_id: int,
        pad_value: int,
    ) -> None:
        self.matrix = matrix
        self.gene_ids = gene_ids
        self.labels = labels
        self.cls_token_id = cls_token_id
        self.pad_value = pad_value

    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.matrix[index]
        dense_row = row.toarray().ravel() if sparse.issparse(row) else np.asarray(row).ravel()
        non_zero_idx = np.nonzero(dense_row)[0]
        values = dense_row[non_zero_idx]
        genes = self.gene_ids[non_zero_idx]
        genes = np.insert(genes, 0, self.cls_token_id)
        values = np.insert(values, 0, self.pad_value)
        sample: dict[str, torch.Tensor] = {
            "genes": torch.as_tensor(genes, dtype=torch.long),
            "expressions": torch.as_tensor(values, dtype=torch.float32),
        }
        if self.labels is not None:
            sample["y"] = torch.as_tensor(int(self.labels[index]), dtype=torch.long)
        return sample


class _FixedTensorDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        padding_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> None:
        self.gene_ids = gene_ids
        self.values = values
        self.padding_mask = padding_mask
        self.labels = labels

    def __len__(self) -> int:
        return int(self.gene_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample: dict[str, torch.Tensor] = {
            "gene_ids": self.gene_ids[index],
            "values": self.values[index],
            "padding_mask": self.padding_mask[index],
        }
        if self.labels is not None:
            sample["y"] = self.labels[index]
        return sample


class _ScGPTDataCollator:
    def __init__(
        self,
        *,
        pad_token_id: int,
        pad_value: int,
        max_length: int,
        n_bins: int,
        seed: int = 0,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.pad_value = pad_value
        self.max_length = max_length
        self.n_bins = n_bins
        self.keep_first_n_tokens = 1
        self._generator = torch.Generator().manual_seed(seed)
        self._rng = np.random.default_rng(seed)

    def _sample_or_pad(
        self,
        genes: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(genes) > self.max_length:
            indices = torch.randperm(
                len(genes) - self.keep_first_n_tokens,
                generator=self._generator,
            )[: self.max_length - self.keep_first_n_tokens]
            indices = torch.cat(
                [
                    torch.arange(self.keep_first_n_tokens, dtype=torch.long),
                    indices + self.keep_first_n_tokens,
                ]
            )
            genes = genes[indices]
            values = values[indices]
        elif len(genes) < self.max_length:
            genes = torch.cat(
                [
                    genes,
                    torch.full(
                        (self.max_length - len(genes),),
                        self.pad_token_id,
                        dtype=genes.dtype,
                    ),
                ]
            )
            values = torch.cat(
                [
                    values,
                    torch.full(
                        (self.max_length - len(values),),
                        self.pad_value,
                        dtype=values.dtype,
                    ),
                ]
            )
        return genes, values

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        batched_gene_ids: list[torch.Tensor] = []
        batched_values: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for example in examples:
            genes = example["genes"]
            values = example["expressions"].clone()
            if len(values) > self.keep_first_n_tokens:
                values[self.keep_first_n_tokens :] = _bin_row(
                    values[self.keep_first_n_tokens :],
                    n_bins=self.n_bins,
                    rng=self._rng,
                )
            padded_genes, padded_values = self._sample_or_pad(genes, values)
            batched_gene_ids.append(padded_genes)
            batched_values.append(padded_values)
            if "y" in example:
                labels.append(example["y"])
        gene_ids = torch.stack(batched_gene_ids, dim=0)
        values = torch.stack(batched_values, dim=0)
        batch: dict[str, torch.Tensor] = {
            "gene_ids": gene_ids,
            "values": values,
            "padding_mask": gene_ids.eq(self.pad_token_id),
        }
        if labels:
            batch["y"] = torch.stack(labels, dim=0)
        return batch


def _validate_expression_values(matrix: Any) -> None:
    min_value = float(matrix.min()) if sparse.issparse(matrix) else float(np.min(matrix))
    if min_value < 0:
        msg = (
            "scGPT preparation requires non-negative expression values. "
            f"Observed minimum value {min_value:.4f}."
        )
        raise ValueError(msg)


def _encode_labels(
    adata: AnnData,
    label_key: str | None,
) -> tuple[np.ndarray | None, tuple[str, ...] | None]:
    if label_key is None:
        return None, None
    if label_key not in adata.obs:
        msg = f"Label key '{label_key}' is not present in adata.obs."
        raise ValueError(msg)
    categories = pd.Categorical(adata.obs[label_key].astype(str))
    return categories.codes.astype(np.int64), tuple(str(value) for value in categories.categories)


def _select_expression_adata(adata: AnnData, *, use_raw: bool) -> AnnData:
    if use_raw and adata.raw is not None:
        return adata.raw.to_adata()
    return adata.copy()


def _match_genes(
    adata: AnnData,
    vocab: GeneVocab,
    *,
    min_gene_overlap: int,
) -> tuple[Any, tuple[str, ...], np.ndarray]:
    gene_names = [str(name) for name in adata.var_names]
    vocab_ids = np.array(
        [vocab.get(name, -1) for name in gene_names],
        dtype=np.int64,
    )
    matched_mask = vocab_ids >= 0
    num_matched = int(matched_mask.sum())
    if num_matched < min_gene_overlap:
        msg = (
            "Insufficient gene overlap with the scGPT checkpoint vocabulary: "
            f"matched {num_matched} genes, required at least {min_gene_overlap}."
        )
        raise ValueError(msg)
    filtered_matrix = adata[:, matched_mask].X
    matched_gene_names = tuple(np.asarray(gene_names, dtype=object)[matched_mask].tolist())
    matched_vocab_ids = vocab_ids[matched_mask]
    return filtered_matrix, matched_gene_names, matched_vocab_ids


def _count_class_labels(
    adata: AnnData,
    label_key: str,
) -> tuple[tuple[str, ...], dict[str, int]]:
    if label_key not in adata.obs:
        msg = f"Label key '{label_key}' is not present in adata.obs."
        raise ValueError(msg)
    categories = pd.Categorical(adata.obs[label_key].astype(str))
    label_categories = tuple(str(value) for value in categories.categories)
    counts = pd.Series(categories).value_counts(sort=False)
    class_counts = {
        category: int(counts.get(category, 0))
        for category in label_categories
    }
    return label_categories, class_counts


def inspect_scgpt_annotation_data(
    adata: AnnData,
    *,
    label_key: str,
    checkpoint: str = DEFAULT_SCGPT_CHECKPOINT,
    use_raw: bool = True,
    min_gene_overlap: int = 500,
    min_cells_per_class: int = 10,
) -> ScGPTAnnotationDataReport:
    """Inspect whether a dataset is a reasonable candidate for scGPT annotation.

    Parameters
    ----------
    adata
        Input single-cell dataset.
    label_key
        Categorical label column in ``adata.obs``.
    checkpoint
        scGPT checkpoint identifier. Only ``"whole-human"`` is supported
        publicly at the moment.
    use_raw
        Use ``adata.raw`` when available.
    min_gene_overlap
        Warning threshold for matched genes against the scGPT vocabulary.
    min_cells_per_class
        Warning threshold for the smallest class size.

    Returns
    -------
    ScGPTAnnotationDataReport
        Lightweight compatibility report with overlap and class-balance checks.

    Raises
    ------
    ValueError
        If labels are missing or the expression matrix contains negative values.
    """

    _, _, vocab = _load_scgpt_assets(checkpoint)
    expression_adata = _select_expression_adata(adata, use_raw=use_raw)
    _validate_expression_values(expression_adata.X)
    gene_names = [str(name) for name in expression_adata.var_names]
    matched_mask = np.array(
        [
            (vocab_index is not None) and vocab_index >= 0
            for vocab_index in (vocab.get(name) for name in gene_names)
        ],
        dtype=bool,
    )
    num_input_genes = int(len(gene_names))
    num_genes_matched = int(matched_mask.sum())
    overlap_ratio = (
        float(num_genes_matched / num_input_genes)
        if num_input_genes > 0
        else 0.0
    )
    label_categories, class_counts = _count_class_labels(adata, label_key)
    min_class_count = min(class_counts.values()) if class_counts else 0
    stratify_possible = (
        len(label_categories) > 1 and min_class_count >= max(2, min_cells_per_class)
    )
    warnings: list[str] = []
    if num_genes_matched < min_gene_overlap:
        warnings.append(
            "Gene overlap with the scGPT checkpoint vocabulary is below the "
            f"recommended threshold ({num_genes_matched} matched, recommended at least "
            f"{min_gene_overlap})."
        )
    if len(label_categories) < 2:
        warnings.append(
            "Annotation adaptation expects at least two label categories. "
            f"Observed {len(label_categories)}."
        )
    if min_class_count < min_cells_per_class:
        warnings.append(
            "The smallest label class is below the recommended size for stable tuning "
            f"({min_class_count} cells, recommended at least {min_cells_per_class})."
        )
    if not stratify_possible:
        warnings.append(
            "Class counts are likely too sparse for strict stratified train/validation/test "
            "splits with the default wrapper settings."
        )
    return ScGPTAnnotationDataReport(
        checkpoint_id=checkpoint,
        label_key=label_key,
        num_cells=int(expression_adata.n_obs),
        num_input_genes=num_input_genes,
        num_genes_matched=num_genes_matched,
        gene_overlap_ratio=overlap_ratio,
        label_categories=label_categories,
        class_counts=class_counts,
        min_class_count=min_class_count,
        stratify_possible=stratify_possible,
        warnings=tuple(warnings),
    )


def prepare_scgpt_data(
    adata: AnnData,
    *,
    checkpoint: str = DEFAULT_SCGPT_CHECKPOINT,
    label_key: str | None = None,
    batch_size: int = 64,
    use_raw: bool = True,
    min_gene_overlap: int = 500,
) -> ScGPTPreparedData:
    """Prepare tokenized data for frozen scGPT embedding inference."""

    _, config, vocab = _load_scgpt_assets(checkpoint)
    expression_adata = _select_expression_adata(adata, use_raw=use_raw)
    _validate_expression_values(expression_adata.X)
    filtered_matrix, gene_names, matched_vocab_ids = _match_genes(
        expression_adata,
        vocab,
        min_gene_overlap=min_gene_overlap,
    )
    labels, label_categories = _encode_labels(adata, label_key)
    source_dataset = _ScGPTTokenSourceDataset(
        filtered_matrix,
        matched_vocab_ids,
        labels,
        cls_token_id=vocab["<cls>"],
        pad_value=int(config["pad_value"]),
    )
    collator = _ScGPTDataCollator(
        pad_token_id=vocab[str(config["pad_token"])],
        pad_value=int(config["pad_value"]),
        max_length=min(int(config.get("max_seq_len", 1200)), len(gene_names) + 1),
        n_bins=int(config.get("n_bins", 51)),
    )
    loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    gene_id_batches: list[torch.Tensor] = []
    value_batches: list[torch.Tensor] = []
    padding_mask_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    for batch in loader:
        gene_id_batches.append(batch["gene_ids"].cpu())
        value_batches.append(batch["values"].cpu())
        padding_mask_batches.append(batch["padding_mask"].cpu())
        if "y" in batch:
            label_batches.append(batch["y"].cpu())

    prepared_dataset = _FixedTensorDataset(
        gene_ids=torch.cat(gene_id_batches, dim=0),
        values=torch.cat(value_batches, dim=0),
        padding_mask=torch.cat(padding_mask_batches, dim=0),
        labels=torch.cat(label_batches, dim=0) if label_batches else None,
    )
    return ScGPTPreparedData(
        dataset=prepared_dataset,
        gene_names=gene_names,
        checkpoint_id=checkpoint,
        label_key=label_key,
        label_categories=label_categories,
        batch_size=batch_size,
        num_cells=int(expression_adata.n_obs),
        num_genes_matched=len(gene_names),
    )


def _labels_for_dataset(dataset: Dataset[dict[str, torch.Tensor]]) -> np.ndarray | None:
    labels = getattr(dataset, "labels", None)
    if isinstance(labels, torch.Tensor):
        return labels.detach().cpu().numpy()
    return None


def _split_with_optional_stratification(
    indices: np.ndarray,
    *,
    labels: np.ndarray | None,
    split_size: float,
    random_state: int,
    stratify: bool,
) -> tuple[np.ndarray, np.ndarray]:
    stratify_labels = None
    if stratify and labels is not None and len(np.unique(labels)) > 1:
        stratify_labels = labels
    try:
        train_idx, holdout_idx = train_test_split(
            indices,
            test_size=split_size,
            random_state=random_state,
            stratify=stratify_labels,
        )
    except ValueError:
        train_idx, holdout_idx = train_test_split(
            indices,
            test_size=split_size,
            random_state=random_state,
            stratify=None,
        )
    return np.sort(train_idx), np.sort(holdout_idx)


def split_scgpt_data(
    prepared: ScGPTPreparedData,
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
) -> ScGPTSplitData:
    """Split a prepared scGPT dataset into train, validation, and test subsets.

    Parameters
    ----------
    prepared
        Tokenized scGPT dataset returned by :func:`prepare_scgpt_data`.
    val_size
        Fraction of cells reserved for validation.
    test_size
        Fraction of cells reserved for test evaluation.
    random_state
        Random seed used for deterministic splitting.
    stratify
        Attempt stratified splitting when labels are available.

    Returns
    -------
    ScGPTSplitData
        Split-aware datasets plus the metadata needed for reporting.

    Raises
    ------
    ValueError
        If the split fractions are invalid.
    """

    if val_size < 0 or test_size < 0:
        msg = "val_size and test_size must be non-negative."
        raise ValueError(msg)
    if val_size + test_size >= 1.0:
        msg = "val_size and test_size must sum to less than 1.0."
        raise ValueError(msg)

    total = len(cast(Sized, prepared.dataset))
    indices = np.arange(total, dtype=np.int64)
    labels = _labels_for_dataset(prepared.dataset)

    train_indices = indices
    val_indices: np.ndarray | None = None
    test_indices: np.ndarray | None = None

    if test_size > 0.0:
        remaining, test_indices = _split_with_optional_stratification(
            train_indices,
            labels=labels,
            split_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        train_indices = remaining

    if val_size > 0.0:
        adjusted_val_size = val_size / (1.0 - test_size)
        train_labels = labels[train_indices] if labels is not None else None
        remaining, val_indices = _split_with_optional_stratification(
            train_indices,
            labels=train_labels,
            split_size=adjusted_val_size,
            random_state=random_state,
            stratify=stratify,
        )
        train_indices = remaining

    return ScGPTSplitData(
        train=Subset(prepared.dataset, train_indices.tolist()),
        val=Subset(prepared.dataset, val_indices.tolist()) if val_indices is not None else None,
        test=Subset(prepared.dataset, test_indices.tolist()) if test_indices is not None else None,
        checkpoint_id=prepared.checkpoint_id,
        label_key=prepared.label_key,
        label_categories=prepared.label_categories,
        gene_names=prepared.gene_names,
        batch_size=prepared.batch_size,
        num_cells=prepared.num_cells,
        num_genes_matched=prepared.num_genes_matched,
    )
