from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

from scdlkit import AnnotationRunner, adapt_annotation, inspect_annotation_data
from scdlkit._datasets import openproblems
from scdlkit.foundation.scgpt import ScGPTBackbone


def _write_checkpoint_dir(path: Path, genes: list[str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    vocab = {"<pad>": 0, "<cls>": 1, "<eoc>": 2}
    vocab.update({gene: index + 3 for index, gene in enumerate(genes)})
    config = {
        "embsize": 16,
        "nheads": 4,
        "d_hid": 32,
        "nlayers": 2,
        "dropout": 0.1,
        "pad_token": "<pad>",
        "pad_value": -2,
        "max_seq_len": 16,
        "n_bins": 8,
    }
    (path / "args.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    backbone = ScGPTBackbone(
        ntoken=len(vocab),
        d_model=config["embsize"],
        nhead=config["nheads"],
        d_hid=config["d_hid"],
        nlayers=config["nlayers"],
        dropout=config["dropout"],
        pad_token_id=vocab["<pad>"],
    )
    torch.save(backbone.state_dict(), path / "best_model.pt")


def _make_raw_pancreas_adata() -> AnnData:
    genes = [
        "MS4A1",
        "CD3D",
        "NKG7",
        "LYZ",
        "PPBP",
        "FCGR3A",
        "IL7R",
        "MALAT1",
        "GNLY",
        "HLA-DRA",
        "S100A8",
        "CTSS",
    ]
    label_counts = {
        "alpha": 18,
        "beta": 17,
        "gamma": 16,
        "delta": 15,
        "ductal": 14,
        "acinar": 13,
        "stellate": 12,
        "endothelial": 11,
        "mast": 10,
        "cycling": 9,
    }
    labels: list[str] = []
    batches: list[str] = []
    for label_index, (label, count) in enumerate(label_counts.items()):
        for item_index in range(count):
            labels.append(label)
            batches.append(f"batch_{(label_index + item_index) % 3}")
    rng = np.random.default_rng(42)
    counts = rng.poisson(lam=3.0, size=(len(labels), len(genes))).astype("float32")
    obs = pd.DataFrame(
        {
            "cell_type": labels,
            "batch": batches,
        },
        index=[f"cell_{index}" for index in range(len(labels))],
    )
    var = pd.DataFrame(
        {
            "feature_name": genes,
            "feature_id": [f"ENSG{index:06d}" for index in range(len(genes))],
        },
        index=[f"id_{index}" for index in range(len(genes))],
    )
    adata = AnnData(X=counts.copy(), obs=obs, var=var)
    adata.layers["counts"] = counts.copy()
    adata.uns["organism"] = "homo_sapiens"
    return adata


@pytest.fixture
def raw_pancreas_path(tmp_path: Path) -> Path:
    path = tmp_path / "openproblems_pancreas_fixture.h5ad"
    _make_raw_pancreas_adata().write_h5ad(path)
    return path


@pytest.fixture
def openproblems_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache_root = tmp_path / "cache"
    checkpoint_dir = cache_root / "foundation" / "scgpt" / "whole-human"
    genes = list(_make_raw_pancreas_adata().var["feature_name"].astype(str))
    _write_checkpoint_dir(checkpoint_dir, genes)
    monkeypatch.setenv("SCDLKIT_CACHE_DIR", str(cache_root))
    return cache_root


def test_ensure_openproblems_dataset_downloads_into_shared_cache(
    raw_pancreas_path: Path,
    openproblems_cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        openproblems,
        "_download_file",
        lambda url, destination: shutil.copyfile(raw_pancreas_path, destination),
    )

    raw_path = openproblems.ensure_openproblems_dataset(
        "openproblems_v1/pancreas",
        cache_dir=openproblems_cache_dir,
    )

    assert raw_path == (
        openproblems_cache_dir
        / "datasets"
        / "openproblems_v1"
        / "pancreas"
        / "raw"
        / "dataset.h5ad"
    )
    assert raw_path.exists()
    metadata_path = raw_path.parent / "metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["dataset_id"] == "openproblems_v1/pancreas"


def test_ensure_openproblems_dataset_raises_for_missing_required_fields(
    tmp_path: Path,
    openproblems_cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken = _make_raw_pancreas_adata()
    del broken.layers["counts"]
    broken_path = tmp_path / "broken_pancreas.h5ad"
    broken.write_h5ad(broken_path)
    monkeypatch.setattr(
        openproblems,
        "_download_file",
        lambda url, destination: shutil.copyfile(broken_path, destination),
    )

    with pytest.raises(ValueError, match="missing required fields"):
        openproblems.ensure_openproblems_dataset(
            "openproblems_v1/pancreas",
            cache_dir=openproblems_cache_dir,
            force_download=True,
        )


def test_processed_pancreas_subset_uses_counts_and_unique_feature_names(
    raw_pancreas_path: Path,
    openproblems_cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        openproblems,
        "_download_file",
        lambda url, destination: shutil.copyfile(raw_pancreas_path, destination),
    )

    adata = openproblems.load_openproblems_pancreas_annotation_dataset(
        profile="quickstart",
        cache_dir=openproblems_cache_dir,
        force_download=True,
        force_rebuild=True,
    )

    np.testing.assert_allclose(np.asarray(adata.X), np.asarray(adata.layers["counts"]))
    np.testing.assert_allclose(np.asarray(adata.raw.X), np.asarray(adata.layers["counts"]))
    assert np.min(np.asarray(adata.X)) >= 0
    assert adata.var_names.is_unique
    assert "feature_id" in adata.var


def test_processed_pancreas_subset_builds_deterministically(
    raw_pancreas_path: Path,
    openproblems_cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        openproblems,
        "_download_file",
        lambda url, destination: shutil.copyfile(raw_pancreas_path, destination),
    )

    quickstart_a = openproblems.load_openproblems_pancreas_annotation_dataset(
        profile="quickstart",
        cache_dir=openproblems_cache_dir,
        force_download=True,
        force_rebuild=True,
    )
    quickstart_b = openproblems.load_openproblems_pancreas_annotation_dataset(
        profile="quickstart",
        cache_dir=openproblems_cache_dir,
        force_rebuild=True,
    )
    full_a = openproblems.load_openproblems_pancreas_annotation_dataset(
        profile="full",
        cache_dir=openproblems_cache_dir,
        force_rebuild=True,
    )
    full_b = openproblems.load_openproblems_pancreas_annotation_dataset(
        profile="full",
        cache_dir=openproblems_cache_dir,
        force_rebuild=True,
    )

    np.testing.assert_allclose(np.asarray(quickstart_a.X), np.asarray(quickstart_b.X))
    np.testing.assert_allclose(np.asarray(full_a.X), np.asarray(full_b.X))
    assert list(quickstart_a.obs_names) == list(quickstart_b.obs_names)
    assert list(full_a.var_names) == list(full_b.var_names)


def test_processed_pancreas_subset_keeps_expected_top_cell_types(
    raw_pancreas_path: Path,
    openproblems_cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        openproblems,
        "_download_file",
        lambda url, destination: shutil.copyfile(raw_pancreas_path, destination),
    )

    adata = openproblems.load_openproblems_pancreas_annotation_dataset(
        profile="quickstart",
        cache_dir=openproblems_cache_dir,
        force_download=True,
        force_rebuild=True,
    )

    expected = {
        "alpha",
        "beta",
        "gamma",
        "delta",
        "ductal",
        "acinar",
        "stellate",
        "endothelial",
    }
    assert set(adata.obs["cell_type"].astype(str).unique()) == expected


def test_sampling_labels_falls_back_from_combined_batch_to_label_only() -> None:
    obs = pd.DataFrame(
        {
            "cell_type": ["a", "a", "b", "b", "c", "c"],
            "batch": ["x1", "x2", "y1", "y2", "z1", "z2"],
        }
    )
    sampling = openproblems._sampling_labels(
        obs,
        label_key="cell_type",
        batch_key="batch",
        max_cells=4,
    )
    assert sampling is not None
    assert sorted(set(sampling.tolist())) == ["a", "b", "c"]


def test_processed_pancreas_subset_is_wrapper_compatible(
    raw_pancreas_path: Path,
    openproblems_cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        openproblems,
        "_download_file",
        lambda url, destination: shutil.copyfile(raw_pancreas_path, destination),
    )
    adata = openproblems.load_openproblems_pancreas_annotation_dataset(
        profile="quickstart",
        cache_dir=openproblems_cache_dir,
        force_download=True,
        force_rebuild=True,
    )
    report = inspect_annotation_data(
        adata,
        label_key="cell_type",
        checkpoint="whole-human",
    )
    assert report.num_genes_matched >= 8

    runner = adapt_annotation(
        adata,
        label_key="cell_type",
        batch_size=16,
        device="cpu",
        output_dir=tmp_path / "pancreas_wrapper",
    )
    save_dir = runner.save(tmp_path / "pancreas_best_model")
    loaded = AnnotationRunner.load(save_dir, device="cpu", cache_dir=openproblems_cache_dir)

    before = runner.predict(adata)
    after = loaded.predict(adata)
    np.testing.assert_array_equal(before["label_codes"], after["label_codes"])
    np.testing.assert_allclose(before["probabilities"], after["probabilities"], atol=1e-6)
