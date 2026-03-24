"""LoRA helpers for experimental scGPT annotation tuning."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional

from scdlkit.foundation.peft import LoRAConfig
from scdlkit.foundation.scgpt import ScGPTBackbone

ScGPTLoRAConfig = LoRAConfig


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


def apply_scgpt_lora(backbone: ScGPTBackbone, config: LoRAConfig) -> None:
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


__all__ = [
    "LoRALinear",
    "LoRAConfig",
    "ScGPTLoRAConfig",
    "apply_scgpt_lora",
]
