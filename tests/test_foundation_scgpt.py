from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

from scdlkit.evaluation import evaluate_predictions
from scdlkit.foundation import (
    ensure_scgpt_checkpoint,
    list_scgpt_checkpoints,
    load_scgpt_model,
    prepare_scgpt_data,
)
from scdlkit.foundation.scgpt import ScGPTBackbone
from scdlkit.training import Trainer


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
def scgpt_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache_root = tmp_path / "cache"
    checkpoint_dir = cache_root / "foundation" / "scgpt" / "whole-human"
    _write_checkpoint_dir(checkpoint_dir)
    monkeypatch.setenv("SCDLKIT_CACHE_DIR", str(cache_root))
    return cache_root


@pytest.fixture
def scgpt_adata() -> AnnData:
    rng = np.random.default_rng(19)
    genes = ["MS4A1", "CD3D", "NKG7", "LYZ", "PPBP", "FCGR3A", "IL7R"]
    counts = rng.poisson(lam=3.0, size=(30, len(genes))).astype("float32")
    obs = pd.DataFrame(
        {
            "louvain": np.array(["0", "1", "2"] * 10),
        },
        index=[f"cell_{index}" for index in range(30)],
    )
    adata = AnnData(X=counts, obs=obs)
    adata.var_names = genes
    adata.raw = adata.copy()
    return adata


def test_list_scgpt_checkpoints_contains_whole_human() -> None:
    checkpoints = list_scgpt_checkpoints()
    assert "whole-human" in checkpoints
    assert "url" in checkpoints["whole-human"]


def test_ensure_scgpt_checkpoint_downloads_into_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "download_source" / "scGPT_human"
    _write_checkpoint_dir(source_root)

    def fake_download_folder(
        *,
        id: str,
        output: str,
        quiet: bool,
        remaining_ok: bool,
        resume: bool,
    ) -> list[str]:
        assert id
        assert quiet is False
        assert remaining_ok is True
        assert resume is True
        destination = Path(output) / "scGPT_human"
        shutil.copytree(source_root, destination, dirs_exist_ok=True)
        return [str(destination / "args.json")]

    monkeypatch.setattr("scdlkit.foundation.cache.gdown.download_folder", fake_download_folder)
    target = ensure_scgpt_checkpoint("whole-human", cache_dir=tmp_path / "cache")
    assert (target / "args.json").exists()
    assert (target / "best_model.pt").exists()


def test_prepare_scgpt_data_builds_tokenized_dataset(
    scgpt_cache_dir: Path,
    scgpt_adata: AnnData,
) -> None:
    assert scgpt_cache_dir.exists()
    prepared = prepare_scgpt_data(
        scgpt_adata,
        checkpoint="whole-human",
        label_key="louvain",
        batch_size=8,
        min_gene_overlap=4,
    )
    assert prepared.checkpoint_id == "whole-human"
    assert prepared.num_cells == scgpt_adata.n_obs
    assert prepared.num_genes_matched >= 4
    sample = prepared.dataset[0]
    assert {"gene_ids", "values", "padding_mask", "y"} <= set(sample)


def test_prepare_scgpt_data_rejects_negative_values(
    scgpt_cache_dir: Path,
    scgpt_adata: AnnData,
) -> None:
    assert scgpt_cache_dir.exists()
    scgpt_adata.X[0, 0] = -1.0
    scgpt_adata.raw = scgpt_adata.copy()
    with pytest.raises(ValueError, match="non-negative"):
        prepare_scgpt_data(
            scgpt_adata,
            checkpoint="whole-human",
            label_key="louvain",
            min_gene_overlap=4,
        )


def test_prepare_scgpt_data_rejects_low_gene_overlap(
    scgpt_cache_dir: Path,
    scgpt_adata: AnnData,
) -> None:
    assert scgpt_cache_dir.exists()
    scgpt_adata.var_names = [f"gene_{index}" for index in range(scgpt_adata.n_vars)]
    scgpt_adata.raw = scgpt_adata.copy()
    with pytest.raises(ValueError, match="Insufficient gene overlap"):
        prepare_scgpt_data(
            scgpt_adata,
            checkpoint="whole-human",
            label_key="louvain",
            min_gene_overlap=4,
        )


def test_load_scgpt_model_and_predict_dataset(scgpt_cache_dir: Path, scgpt_adata: AnnData) -> None:
    model = load_scgpt_model("whole-human", device="cpu", cache_dir=scgpt_cache_dir)
    prepared = prepare_scgpt_data(
        scgpt_adata,
        checkpoint="whole-human",
        label_key="louvain",
        batch_size=8,
        min_gene_overlap=4,
    )
    trainer = Trainer(
        model=model,
        task="representation",
        batch_size=prepared.batch_size,
        device="cpu",
    )
    predictions = trainer.predict_dataset(prepared.dataset)
    assert predictions["latent"].shape[0] == scgpt_adata.n_obs
    metrics = evaluate_predictions("representation", predictions)
    assert "silhouette" in metrics


def test_trainer_fit_rejects_inference_only_scgpt_model(
    scgpt_cache_dir: Path,
    scgpt_adata: AnnData,
) -> None:
    model = load_scgpt_model("whole-human", device="cpu", cache_dir=scgpt_cache_dir)
    prepared = prepare_scgpt_data(
        scgpt_adata,
        checkpoint="whole-human",
        label_key="louvain",
        batch_size=8,
        min_gene_overlap=4,
    )
    trainer = Trainer(
        model=model,
        task="representation",
        batch_size=prepared.batch_size,
        device="cpu",
    )
    with pytest.raises(NotImplementedError, match="inference-only"):
        trainer.fit(prepared.dataset)
