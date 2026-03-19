from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

from scdlkit.evaluation import evaluate_predictions
from scdlkit.foundation import (
    ScGPTLoRAConfig,
    load_scgpt_annotation_model,
    load_scgpt_model,
    prepare_scgpt_data,
    split_scgpt_data,
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
def scgpt_annotation_adata() -> AnnData:
    rng = np.random.default_rng(19)
    genes = ["MS4A1", "CD3D", "NKG7", "LYZ", "PPBP", "FCGR3A", "IL7R"]
    counts = rng.poisson(lam=3.0, size=(36, len(genes))).astype("float32")
    labels = np.array(["B_cell"] * 12 + ["T_cell"] * 12 + ["Monocyte"] * 12, dtype=object)
    obs = pd.DataFrame(
        {
            "louvain": labels,
        },
        index=[f"cell_{index}" for index in range(36)],
    )
    adata = AnnData(X=counts, obs=obs)
    adata.var_names = genes
    adata.raw = adata.copy()
    return adata


def _subset_labels(dataset: torch.utils.data.Dataset[dict[str, torch.Tensor]]) -> list[int]:
    return [int(dataset[index]["y"]) for index in range(len(dataset))]


def test_prepare_scgpt_data_returns_label_categories(
    scgpt_cache_dir: Path,
    scgpt_annotation_adata: AnnData,
) -> None:
    prepared = prepare_scgpt_data(
        scgpt_annotation_adata,
        checkpoint="whole-human",
        label_key="louvain",
        batch_size=8,
        min_gene_overlap=4,
    )
    assert prepared.label_categories == ("B_cell", "Monocyte", "T_cell")


def test_split_scgpt_data_preserves_metadata_and_uses_stratified_split(
    scgpt_cache_dir: Path,
    scgpt_annotation_adata: AnnData,
) -> None:
    prepared = prepare_scgpt_data(
        scgpt_annotation_adata,
        checkpoint="whole-human",
        label_key="louvain",
        batch_size=8,
        min_gene_overlap=4,
    )
    split = split_scgpt_data(prepared, val_size=0.2, test_size=0.2, random_state=7)
    assert split.label_categories == prepared.label_categories
    assert split.checkpoint_id == prepared.checkpoint_id
    assert split.num_genes_matched == prepared.num_genes_matched
    assert len(split.train) + len(split.val or []) + len(split.test or []) == prepared.num_cells
    assert set(_subset_labels(split.train)) == {0, 1, 2}
    assert set(_subset_labels(split.val)) == {0, 1, 2}
    assert set(_subset_labels(split.test)) == {0, 1, 2}


def test_split_scgpt_data_falls_back_when_stratification_is_too_sparse(
    scgpt_cache_dir: Path,
    scgpt_annotation_adata: AnnData,
) -> None:
    sparse = scgpt_annotation_adata[:6].copy()
    sparse.obs["louvain"] = ["a", "a", "b", "c", "d", "e"]
    sparse.raw = sparse.copy()
    prepared = prepare_scgpt_data(
        sparse,
        checkpoint="whole-human",
        label_key="louvain",
        batch_size=4,
        min_gene_overlap=4,
    )
    split = split_scgpt_data(prepared, val_size=0.2, test_size=0.2, random_state=3)
    assert len(split.train) + len(split.val or []) + len(split.test or []) == prepared.num_cells


def test_load_scgpt_annotation_model_head_strategy_builds_trainable_head(
    scgpt_cache_dir: Path,
) -> None:
    model = load_scgpt_annotation_model(
        num_classes=3,
        checkpoint="whole-human",
        tuning_strategy="head",
        label_categories=("B_cell", "Monocyte", "T_cell"),
        device="cpu",
        cache_dir=scgpt_cache_dir,
    )
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert trainable_names
    assert all(name.startswith("classifier_head") for name in trainable_names)


def test_load_scgpt_annotation_model_lora_strategy_injects_lora_modules(
    scgpt_cache_dir: Path,
) -> None:
    model = load_scgpt_annotation_model(
        num_classes=3,
        checkpoint="whole-human",
        tuning_strategy="lora",
        label_categories=("B_cell", "Monocyte", "T_cell"),
        lora_config=ScGPTLoRAConfig(rank=4, alpha=8.0, dropout=0.1),
        device="cpu",
        cache_dir=scgpt_cache_dir,
    )
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert any("lora_a" in name for name in trainable_names)
    assert any("lora_b" in name for name in trainable_names)
    frozen_backbone = [
        name
        for name, parameter in model.backbone.named_parameters()
        if not parameter.requires_grad and "lora_" not in name
    ]
    assert frozen_backbone


def test_load_scgpt_annotation_model_rejects_unsupported_strategy(
    scgpt_cache_dir: Path,
) -> None:
    with pytest.raises(ValueError, match="Unsupported tuning strategy"):
        load_scgpt_annotation_model(
            num_classes=3,
            checkpoint="whole-human",
            tuning_strategy="full",
            device="cpu",
            cache_dir=scgpt_cache_dir,
        )


@pytest.mark.parametrize("tuning_strategy", ["head", "lora"])
def test_trainer_fit_and_predict_work_for_scgpt_annotation_model(
    scgpt_cache_dir: Path,
    scgpt_annotation_adata: AnnData,
    tuning_strategy: str,
) -> None:
    prepared = prepare_scgpt_data(
        scgpt_annotation_adata,
        checkpoint="whole-human",
        label_key="louvain",
        batch_size=8,
        min_gene_overlap=4,
    )
    split = split_scgpt_data(prepared, val_size=0.2, test_size=0.2, random_state=11)
    model = load_scgpt_annotation_model(
        num_classes=len(prepared.label_categories or ()),
        checkpoint="whole-human",
        tuning_strategy=tuning_strategy,
        label_categories=prepared.label_categories,
        device="cpu",
        cache_dir=scgpt_cache_dir,
    )
    trainer = Trainer(
        model=model,
        task="classification",
        batch_size=prepared.batch_size,
        epochs=2,
        lr=1e-2,
        device="cpu",
        early_stopping_patience=2,
        seed=5,
    )
    trainer.fit(split.train, split.val)
    predictions = trainer.predict_dataset(split.test)
    assert {"logits", "latent", "y"} <= set(predictions)
    metrics = evaluate_predictions("classification", predictions)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics


def test_frozen_scgpt_embedding_path_remains_available(
    scgpt_cache_dir: Path,
    scgpt_annotation_adata: AnnData,
) -> None:
    prepared = prepare_scgpt_data(
        scgpt_annotation_adata,
        checkpoint="whole-human",
        label_key="louvain",
        batch_size=8,
        min_gene_overlap=4,
    )
    model = load_scgpt_model("whole-human", device="cpu", cache_dir=scgpt_cache_dir)
    trainer = Trainer(
        model=model,
        task="representation",
        batch_size=prepared.batch_size,
        device="cpu",
    )
    predictions = trainer.predict_dataset(prepared.dataset)
    assert "latent" in predictions
