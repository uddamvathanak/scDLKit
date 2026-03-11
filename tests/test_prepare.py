from __future__ import annotations

import pytest

from scdlkit.data.prepare import prepare_data


def test_prepare_data_supports_dense_and_sparse(dense_adata, sparse_adata) -> None:
    dense = prepare_data(dense_adata, label_key="cell_type", batch_key="batch")
    sparse = prepare_data(sparse_adata, label_key="cell_type", batch_key="batch")

    assert dense.input_dim == 20
    assert sparse.input_dim == 20
    assert len(dense.train) > 0
    assert len(sparse.train) > 0
    assert dense.label_encoder is not None
    assert sparse.batch_encoder is not None


def test_prepare_data_batch_aware_split_reduces_batch_leakage(dense_adata) -> None:
    prepared = prepare_data(
        dense_adata,
        label_key="cell_type",
        batch_key="batch",
        batch_aware_split=True,
        test_size=0.2,
        val_size=0.2,
    )

    train_batches = set(prepared.train.batches.tolist())
    test_batches = set(prepared.test.batches.tolist()) if prepared.test is not None else set()
    assert train_batches.isdisjoint(test_batches)


def test_prepare_data_scanpy_hint(monkeypatch, dense_adata) -> None:
    def _boom() -> None:
        raise ImportError("scanpy-backed preprocessing requires `pip install scdlkit[scanpy]`.")

    monkeypatch.setattr("scdlkit.data.prepare._require_scanpy", _boom)
    with pytest.raises(ImportError, match="scdlkit\\[scanpy\\]"):
        prepare_data(dense_adata, use_hvg=True)
