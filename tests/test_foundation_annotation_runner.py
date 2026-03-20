from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

from scdlkit.foundation import (
    ScGPTAnnotationRunner,
    adapt_scgpt_annotation,
    inspect_scgpt_annotation_data,
)
from scdlkit.foundation.scgpt import ScGPTBackbone


def _write_checkpoint_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    vocab = {
        "<pad>": 0,
        "<cls>": 1,
        "<eoc>": 2,
        "MS4A1": 3,
        "CD3D": 4,
        "NKG7": 5,
        "LYZ": 6,
        "PPBP": 7,
        "FCGR3A": 8,
        "IL7R": 9,
    }
    config = {
        "embsize": 16,
        "nheads": 4,
        "d_hid": 32,
        "nlayers": 2,
        "dropout": 0.1,
        "pad_token": "<pad>",
        "pad_value": -2,
        "max_seq_len": 8,
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


@pytest.fixture
def scgpt_runner_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache_root = tmp_path / "cache"
    checkpoint_dir = cache_root / "foundation" / "scgpt" / "whole-human"
    _write_checkpoint_dir(checkpoint_dir)
    monkeypatch.setenv("SCDLKIT_CACHE_DIR", str(cache_root))
    return cache_root


@pytest.fixture
def scgpt_runner_adata() -> AnnData:
    rng = np.random.default_rng(23)
    genes = ["MS4A1", "CD3D", "NKG7", "LYZ", "PPBP", "FCGR3A", "IL7R"]
    counts = rng.poisson(lam=3.0, size=(36, len(genes))).astype("float32")
    labels = np.array(["B_cell"] * 12 + ["T_cell"] * 12 + ["Monocyte"] * 12, dtype=object)
    obs = pd.DataFrame({"louvain": labels}, index=[f"cell_{index}" for index in range(36)])
    adata = AnnData(X=counts, obs=obs)
    adata.var_names = genes
    adata.raw = adata.copy()
    return adata


def test_inspect_scgpt_annotation_data_reports_counts_and_categories(
    scgpt_runner_cache_dir: Path,
    scgpt_runner_adata: AnnData,
) -> None:
    report = inspect_scgpt_annotation_data(
        scgpt_runner_adata,
        label_key="louvain",
        checkpoint="whole-human",
        min_gene_overlap=4,
    )
    assert report.label_categories == ("B_cell", "Monocyte", "T_cell")
    assert report.num_genes_matched >= 4
    assert report.class_counts["B_cell"] == 12
    assert report.stratify_possible is True


def test_inspect_scgpt_annotation_data_warns_for_sparse_classes(
    scgpt_runner_cache_dir: Path,
    scgpt_runner_adata: AnnData,
) -> None:
    sparse = scgpt_runner_adata[:6].copy()
    sparse.obs["louvain"] = ["a", "a", "b", "c", "d", "e"]
    sparse.raw = sparse.copy()
    report = inspect_scgpt_annotation_data(
        sparse,
        label_key="louvain",
        checkpoint="whole-human",
        min_gene_overlap=4,
        min_cells_per_class=3,
    )
    assert report.stratify_possible is False
    assert report.warnings


def test_inspect_scgpt_annotation_data_warns_for_low_gene_overlap(
    scgpt_runner_cache_dir: Path,
    scgpt_runner_adata: AnnData,
) -> None:
    low_overlap = scgpt_runner_adata.copy()
    low_overlap.var_names = [f"gene_{index}" for index in range(low_overlap.n_vars)]
    low_overlap.raw = low_overlap.copy()
    report = inspect_scgpt_annotation_data(
        low_overlap,
        label_key="louvain",
        checkpoint="whole-human",
        min_gene_overlap=4,
    )
    assert report.num_genes_matched == 0
    assert any("Gene overlap" in warning for warning in report.warnings)


def test_runner_fit_compare_populates_summary_and_predicts(
    scgpt_runner_cache_dir: Path,
    scgpt_runner_adata: AnnData,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "wrapper_run"
    runner = ScGPTAnnotationRunner(
        label_key="louvain",
        strategies=("frozen_probe", "head"),
        batch_size=8,
        device="cpu",
        output_dir=output_dir,
    )
    summary = runner.fit_compare(scgpt_runner_adata)
    assert runner.summary_ is not None
    assert runner.best_strategy_ in {"frozen_probe", "head"}
    assert summary.best_strategy == runner.best_strategy_
    assert set(summary.strategy_metrics["strategy"]) == {"frozen_probe", "head"}

    predictions = runner.predict(scgpt_runner_adata)
    assert {"label_codes", "labels", "probabilities", "latent"} <= set(predictions)
    assert predictions["probabilities"].shape[0] == scgpt_runner_adata.n_obs
    assert predictions["latent"].shape[0] == scgpt_runner_adata.n_obs

    annotated = runner.annotate_adata(scgpt_runner_adata, inplace=False)
    assert annotated is not None
    assert "scgpt_label" in annotated.obs
    assert "scgpt_label_code" in annotated.obs
    assert "scgpt_label_confidence" in annotated.obs
    assert "X_scgpt_best" in annotated.obsm

    assert (output_dir / "report.md").exists()
    assert (output_dir / "report.csv").exists()
    assert (output_dir / "strategy_metrics.csv").exists()
    assert (output_dir / "best_strategy_confusion_matrix.png").exists()
    assert (output_dir / "frozen_embedding_umap.png").exists()
    assert (output_dir / "best_strategy_embedding_umap.png").exists()


def test_adapt_scgpt_annotation_returns_fitted_runner(
    scgpt_runner_cache_dir: Path,
    scgpt_runner_adata: AnnData,
    tmp_path: Path,
) -> None:
    runner = adapt_scgpt_annotation(
        scgpt_runner_adata,
        label_key="louvain",
        strategies=("frozen_probe", "head"),
        batch_size=8,
        device="cpu",
        output_dir=tmp_path / "one_shot",
    )
    assert isinstance(runner, ScGPTAnnotationRunner)
    assert runner.summary_ is not None
    assert runner.best_strategy_ in {"frozen_probe", "head"}
