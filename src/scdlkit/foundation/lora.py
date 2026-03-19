"""LoRA helpers for experimental scGPT annotation tuning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from scdlkit.foundation.scgpt import ScGPTBackbone

_ALLOWED_TARGET_MODULES = ("out_proj", "linear1", "linear2")


@dataclass(frozen=True, slots=True)
class ScGPTLoRAConfig:
    """Configuration for scGPT LoRA tuning.

    Parameters
    ----------
    rank
        Low-rank factorization dimension.
    alpha
        Scaling factor applied to the low-rank update.
    dropout
        Dropout applied to LoRA inputs for supported wrapped linear layers.
    target_modules
        Supported transformer-layer modules to wrap in ``v0.1.5``.
    """

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: tuple[str, ...] = _ALLOWED_TARGET_MODULES

    def __post_init__(self) -> None:
        if self.rank <= 0:
            msg = "LoRA rank must be positive."
            raise ValueError(msg)
        if self.alpha <= 0:
            msg = "LoRA alpha must be positive."
            raise ValueError(msg)
        if not 0.0 <= self.dropout < 1.0:
            msg = "LoRA dropout must be in the range [0, 1)."
            raise ValueError(msg)
        invalid = sorted(set(self.target_modules) - set(_ALLOWED_TARGET_MODULES))
        if invalid:
            msg = (
                "Unsupported LoRA target modules: "
                f"{', '.join(invalid)}. Supported values are {', '.join(_ALLOWED_TARGET_MODULES)}."
            )
            raise ValueError(msg)


class LoRALinear(nn.Module):
    """Linear layer wrapper with a trainable low-rank residual."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Parameter(torch.empty(rank, base_layer.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base_layer.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @property
    def weight(self) -> Tensor:
        update = torch.matmul(self.lora_b, self.lora_a) * self.scaling
        return self.base_layer.weight + update

    @property
    def bias(self) -> Tensor | None:
        return self.base_layer.bias

    def forward(self, inputs: Tensor) -> Tensor:
        base = self.base_layer(inputs)
        update = functional.linear(
            self.dropout(inputs),
            torch.matmul(self.lora_b, self.lora_a),
            bias=None,
        )
        return base + update * self.scaling


def apply_scgpt_lora(backbone: ScGPTBackbone, config: ScGPTLoRAConfig) -> None:
    """Inject LoRA modules into the supported scGPT transformer layers."""

    for layer in backbone.transformer_encoder.layers:
        if "out_proj" in config.target_modules:
            layer.self_attn.out_proj = LoRALinear(
                layer.self_attn.out_proj,
                rank=config.rank,
                alpha=config.alpha,
                dropout=0.0,
            )
        if "linear1" in config.target_modules:
            layer.linear1 = LoRALinear(
                layer.linear1,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
            )
        if "linear2" in config.target_modules:
            layer.linear2 = LoRALinear(
                layer.linear2,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
            )
