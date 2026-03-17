"""Official scGPT checkpoint loading for frozen embedding extraction."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from scdlkit.foundation.cache import DEFAULT_SCGPT_CHECKPOINT, ensure_scgpt_checkpoint
from scdlkit.utils import resolve_device


class GeneVocab:
    """Minimal gene vocabulary compatible with official scGPT checkpoint files."""

    def __init__(self, token_to_idx: dict[str, int]) -> None:
        self._token_to_idx = dict(token_to_idx)
        self._idx_to_token: list[str | None] = [None] * len(self._token_to_idx)
        for token, index in self._token_to_idx.items():
            self._idx_to_token[index] = token
        self._default_index: int | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> GeneVocab:
        token_to_idx = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(token_to_idx, dict):
            msg = f"Vocabulary file '{path}' does not contain a token-index mapping."
            raise ValueError(msg)
        return cls({str(token): int(index) for token, index in token_to_idx.items()})

    def __contains__(self, token: object) -> bool:
        return bool(isinstance(token, str) and token in self._token_to_idx)

    def __getitem__(self, token: str) -> int:
        if token in self._token_to_idx:
            return self._token_to_idx[token]
        if self._default_index is None:
            raise KeyError(token)
        return self._default_index

    def __len__(self) -> int:
        return len(self._token_to_idx)

    def __call__(self, tokens: list[str]) -> list[int]:
        return [self[token] for token in tokens]

    def get(self, token: str, default: int | None = None) -> int | None:
        return self._token_to_idx.get(token, default)

    def append_token(self, token: str) -> int:
        if token in self._token_to_idx:
            return self._token_to_idx[token]
        index = len(self._idx_to_token)
        self._token_to_idx[token] = index
        self._idx_to_token.append(token)
        return index

    def set_default_index(self, index: int) -> None:
        self._default_index = index


class GeneEncoder(nn.Module):
    """Gene-token embedding block from the official scGPT architecture."""

    def __init__(self, num_embeddings: int, embedding_dim: int, *, padding_idx: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.enc_norm(self.embedding(x))


class ContinuousValueEncoder(nn.Module):
    """Continuous value encoder from the official scGPT architecture."""

    def __init__(self, d_model: int, *, dropout: float, max_value: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        values = torch.clamp(x.unsqueeze(-1), max=self.max_value)
        values = self.activation(self.linear1(values))
        values = self.linear2(values)
        values = self.norm(values)
        return self.dropout(values)


class ScGPTBackbone(nn.Module):
    """Minimal scGPT inference backbone for frozen cell-embedding extraction."""

    def __init__(
        self,
        *,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float,
        pad_token_id: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_fast_transformer = False
        self.encoder = GeneEncoder(ntoken, d_model, padding_idx=pad_token_id)
        self.value_encoder = ContinuousValueEncoder(d_model, dropout=dropout)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, nlayers)

    def _encode(self, gene_ids: Tensor, values: Tensor, padding_mask: Tensor) -> Tensor:
        token_embeddings = self.encoder(gene_ids)
        value_embeddings = self.value_encoder(values)
        return self.transformer_encoder(
            token_embeddings + value_embeddings,
            src_key_padding_mask=padding_mask,
        )

    def forward(self, gene_ids: Tensor, values: Tensor, padding_mask: Tensor) -> Tensor:
        encoded = self._encode(gene_ids, values, padding_mask)
        return encoded[:, 0, :]


def _load_pretrained_weights(
    model: nn.Module,
    pretrained_params: dict[str, Tensor],
) -> nn.Module:
    remapped = {
        key.replace("Wqkv.", "in_proj_"): value for key, value in pretrained_params.items()
    }
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in remapped.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    model_state.update(compatible)
    model.load_state_dict(model_state)
    return model


def _load_scgpt_assets(
    checkpoint: str = DEFAULT_SCGPT_CHECKPOINT,
    *,
    cache_dir: str | Path | None = None,
) -> tuple[Path, dict[str, Any], GeneVocab]:
    checkpoint_dir = ensure_scgpt_checkpoint(checkpoint, cache_dir=cache_dir)
    config = json.loads((checkpoint_dir / "args.json").read_text(encoding="utf-8"))
    vocab = GeneVocab.from_file(checkpoint_dir / "vocab.json")
    for token in ("<pad>", "<cls>", "<eoc>"):
        vocab.append_token(token)
    vocab.set_default_index(vocab["<pad>"])
    return checkpoint_dir, config, vocab


class ScGPTEmbeddingModel(nn.Module):
    """Frozen scGPT wrapper that exposes batch-aware embedding inference."""

    supported_tasks: tuple[str, ...] = ("representation",)
    supports_training = False

    def __init__(self, *, backbone: ScGPTBackbone, checkpoint_id: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.checkpoint_id = checkpoint_id

    def predict_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        latent = self.backbone(
            batch["gene_ids"],
            batch["values"],
            batch["padding_mask"],
        )
        return {"latent": functional.normalize(latent, p=2, dim=1)}


def load_scgpt_model(
    checkpoint: str = DEFAULT_SCGPT_CHECKPOINT,
    *,
    device: str = "auto",
    cache_dir: str | Path | None = None,
) -> ScGPTEmbeddingModel:
    """Load a frozen scGPT checkpoint for embedding extraction."""

    checkpoint_dir, config, vocab = _load_scgpt_assets(checkpoint, cache_dir=cache_dir)
    model = ScGPTBackbone(
        ntoken=len(vocab),
        d_model=int(config["embsize"]),
        nhead=int(config["nheads"]),
        d_hid=int(config["d_hid"]),
        nlayers=int(config["nlayers"]),
        dropout=float(config["dropout"]),
        pad_token_id=vocab[str(config["pad_token"])],
    )
    state_dict = torch.load(checkpoint_dir / "best_model.pt", map_location="cpu")
    if isinstance(state_dict, dict):
        if "model_state_dict" in state_dict and isinstance(state_dict["model_state_dict"], dict):
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]
    _load_pretrained_weights(model, state_dict)
    model.eval()
    resolved_device = resolve_device(device)
    if resolved_device.type != "cuda":
        warnings.filterwarnings(
            "ignore",
            message=".*nested tensors is in prototype stage.*",
            category=UserWarning,
        )
    return ScGPTEmbeddingModel(backbone=model.to(resolved_device), checkpoint_id=checkpoint)
