from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

from scdlkit.foundation import ScGPTAnnotationRunner
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
def scgpt_persistence_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache_root = tmp_path / "cache"
    checkpoint_dir = cache_root / "foundation" / "scgpt" / "whole-human"
    _write_checkpoint_dir(checkpoint_dir)
    monkeypatch.setenv("SCDLKIT_CACHE_DIR", str(cache_root))
    return cache_root


@pytest.fixture
def scgpt_persistence_adata() -> AnnData:
    rng = np.random.default_rng(41)
    genes = ["MS4A1", "CD3D", "NKG7", "LYZ", "PPBP", "FCGR3A", "IL7R"]
    counts = rng.poisson(lam=3.0, size=(30, len(genes))).astype("float32")
    labels = np.array(["B_cell"] * 10 + ["T_cell"] * 10 + ["Monocyte"] * 10, dtype=object)
    obs = pd.DataFrame({"louvain": labels}, index=[f"cell_{index}" for index in range(30)])
    adata = AnnData(X=counts, obs=obs)
    adata.var_names = genes
    adata.raw = adata.copy()
    return adata


@pytest.mark.parametrize(
    "strategy",
    ["head", "full_finetune", "adapter", "prefix_tuning", "ia3", "lora"],
)
def test_runner_save_and_load_round_trip_for_trainable_strategies(
    scgpt_persistence_cache_dir: Path,
    scgpt_persistence_adata: AnnData,
    tmp_path: Path,
    strategy: str,
) -> None:
    runner = ScGPTAnnotationRunner(
        label_key="louvain",
        strategies=(strategy,),
        batch_size=8,
        device="cpu",
    )
    runner.fit_compare(scgpt_persistence_adata)
    save_dir = runner.save(tmp_path / f"saved_{strategy}_runner")
    assert (save_dir / "manifest.json").exists()
    assert (save_dir / "model_state.pt").exists()

    before = runner.predict(scgpt_persistence_adata)
    loaded = ScGPTAnnotationRunner.load(
        save_dir,
        device="cpu",
        cache_dir=scgpt_persistence_cache_dir,
    )
    after = loaded.predict(scgpt_persistence_adata)

    np.testing.assert_array_equal(before["label_codes"], after["label_codes"])
    np.testing.assert_array_equal(before["labels"], after["labels"])
    np.testing.assert_allclose(before["probabilities"], after["probabilities"], atol=1e-6)
    np.testing.assert_allclose(before["latent"], after["latent"], atol=1e-6)


def test_loaded_runner_preserves_default_strategy_ladder(
    scgpt_persistence_cache_dir: Path,
    scgpt_persistence_adata: AnnData,
    tmp_path: Path,
) -> None:
    runner = ScGPTAnnotationRunner(
        label_key="louvain",
        batch_size=8,
        device="cpu",
    )
    runner.fit_compare(scgpt_persistence_adata)
    save_dir = runner.save(tmp_path / "saved_default_runner")
    loaded = ScGPTAnnotationRunner.load(
        save_dir,
        device="cpu",
        cache_dir=scgpt_persistence_cache_dir,
    )

    assert runner.strategies == ("frozen_probe", "head")
    assert loaded.strategies == ("frozen_probe", "head")


def test_runner_save_and_load_round_trip_for_frozen_probe(
    scgpt_persistence_cache_dir: Path,
    scgpt_persistence_adata: AnnData,
    tmp_path: Path,
) -> None:
    runner = ScGPTAnnotationRunner(
        label_key="louvain",
        strategies=("frozen_probe",),
        batch_size=8,
        device="cpu",
    )
    runner.fit_compare(scgpt_persistence_adata)
    save_dir = runner.save(tmp_path / "saved_probe_runner")
    loaded = ScGPTAnnotationRunner.load(
        save_dir,
        device="cpu",
        cache_dir=scgpt_persistence_cache_dir,
    )

    before = runner.predict(scgpt_persistence_adata)
    after = loaded.predict(scgpt_persistence_adata)

    np.testing.assert_array_equal(before["label_codes"], after["label_codes"])
    np.testing.assert_array_equal(before["labels"], after["labels"])
    np.testing.assert_allclose(before["probabilities"], after["probabilities"], atol=1e-6)
    np.testing.assert_allclose(before["latent"], after["latent"], atol=1e-6)
