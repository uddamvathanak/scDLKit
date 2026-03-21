from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

from scdlkit import (
    AnnotationRunner,
    adapt_annotation,
    inspect_annotation_data,
)
from scdlkit.foundation import (
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
def alias_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache_root = tmp_path / "cache"
    checkpoint_dir = cache_root / "foundation" / "scgpt" / "whole-human"
    _write_checkpoint_dir(checkpoint_dir)
    monkeypatch.setenv("SCDLKIT_CACHE_DIR", str(cache_root))
    return cache_root


@pytest.fixture
def alias_adata() -> AnnData:
    rng = np.random.default_rng(17)
    genes = ["MS4A1", "CD3D", "NKG7", "LYZ", "PPBP", "FCGR3A", "IL7R"]
    counts = rng.poisson(lam=3.0, size=(24, len(genes))).astype("float32")
    labels = np.array(["B_cell"] * 8 + ["T_cell"] * 8 + ["Monocyte"] * 8, dtype=object)
    obs = pd.DataFrame({"louvain": labels}, index=[f"cell_{index}" for index in range(24)])
    adata = AnnData(X=counts, obs=obs)
    adata.var_names = genes
    adata.raw = adata.copy()
    return adata


def test_top_level_aliases_are_importable() -> None:
    assert AnnotationRunner is not None
    assert adapt_annotation is not None
    assert inspect_annotation_data is not None


def test_inspect_annotation_data_matches_foundation_helper(
    alias_cache_dir: Path,
    alias_adata: AnnData,
) -> None:
    top_level_report = inspect_annotation_data(
        alias_adata,
        label_key="louvain",
        checkpoint="whole-human",
        min_gene_overlap=4,
    )
    foundation_report = inspect_scgpt_annotation_data(
        alias_adata,
        label_key="louvain",
        checkpoint="whole-human",
        min_gene_overlap=4,
    )
    assert top_level_report == foundation_report


def test_adapt_annotation_matches_foundation_for_frozen_probe(
    alias_cache_dir: Path,
    alias_adata: AnnData,
    tmp_path: Path,
) -> None:
    top_level_runner = adapt_annotation(
        alias_adata,
        label_key="louvain",
        strategies=("frozen_probe",),
        batch_size=8,
        device="cpu",
        output_dir=tmp_path / "top_level",
    )
    foundation_runner = adapt_scgpt_annotation(
        alias_adata,
        label_key="louvain",
        strategies=("frozen_probe",),
        batch_size=8,
        device="cpu",
        output_dir=tmp_path / "foundation",
    )

    assert isinstance(top_level_runner, AnnotationRunner)
    assert top_level_runner.best_strategy_ == foundation_runner.best_strategy_
    assert list(top_level_runner.summary_.strategy_metrics["strategy"]) == list(
        foundation_runner.summary_.strategy_metrics["strategy"]
    )

    np.testing.assert_array_equal(
        top_level_runner.predict(alias_adata)["label_codes"],
        foundation_runner.predict(alias_adata)["label_codes"],
    )


def test_annotation_runner_defaults_and_lora_opt_in() -> None:
    runner = AnnotationRunner(label_key="louvain")
    assert runner.strategies == ("frozen_probe", "head")

    lora_runner = AnnotationRunner(
        label_key="louvain",
        strategies=("frozen_probe", "head", "lora"),
    )
    assert lora_runner.strategies == ("frozen_probe", "head", "lora")


def test_annotation_runner_load_round_trip(
    alias_cache_dir: Path,
    alias_adata: AnnData,
    tmp_path: Path,
) -> None:
    runner = adapt_annotation(
        alias_adata,
        label_key="louvain",
        batch_size=8,
        device="cpu",
        output_dir=tmp_path / "alias_round_trip",
    )
    save_dir = runner.save(tmp_path / "saved_alias_runner")
    loaded = AnnotationRunner.load(
        save_dir,
        device="cpu",
        cache_dir=alias_cache_dir,
    )

    assert isinstance(loaded, AnnotationRunner)
    np.testing.assert_array_equal(
        runner.predict(alias_adata)["label_codes"],
        loaded.predict(alias_adata)["label_codes"],
    )
