"""Adapter modules for experimental scGPT annotation tuning."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from scdlkit.foundation.peft import AdapterConfig
from scdlkit.foundation.scgpt import ScGPTBackbone


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported adapter activation '{name}'.")


class ResidualAdapter(nn.Module):
    """Bottleneck adapter block with a residual projection."""

    def __init__(self, *, d_model: int, config: AdapterConfig) -> None:
        super().__init__()
        self.down = nn.Linear(d_model, config.bottleneck_dim)
        self.activation = _activation(config.activation)
        self.dropout = nn.Dropout(config.dropout)
        self.up = nn.Linear(config.bottleneck_dim, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.dropout(self.activation(self.down(x))))


class AdapterTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer wrapper with attention and FF adapters."""

    def __init__(self, layer: Any, *, config: AdapterConfig, d_model: int) -> None:
        super().__init__()
        self.layer = layer
        self.self_attn = layer.self_attn
        self.linear1 = layer.linear1
        self.linear2 = layer.linear2
        self.dropout = layer.dropout
        self.dropout1 = layer.dropout1
        self.dropout2 = layer.dropout2
        self.norm1 = layer.norm1
        self.norm2 = layer.norm2
        self.activation = layer.activation
        self.norm_first = getattr(layer, "norm_first", False)
        self.attention_adapter = ResidualAdapter(d_model=d_model, config=config)
        self.feed_forward_adapter = ResidualAdapter(d_model=d_model, config=config)

    def _sa_block(
        self,
        x: Tensor,
        *,
        src_mask: Tensor | None,
        src_key_padding_mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        kwargs: dict[str, Any] = {
            "attn_mask": src_mask,
            "key_padding_mask": src_key_padding_mask,
            "need_weights": False,
        }
        if "is_causal" in self.layer.self_attn.forward.__code__.co_varnames:
            kwargs["is_causal"] = is_causal
        attention_output = self.layer.self_attn(x, x, x, **kwargs)[0]
        return self.layer.dropout1(attention_output)

    def _ff_block(self, x: Tensor) -> Tensor:
        hidden = self.layer.linear1(x)
        hidden = self.layer.activation(hidden)
        hidden = self.layer.dropout(hidden)
        hidden = self.layer.linear2(hidden)
        return self.layer.dropout2(hidden)

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        x = src
        if getattr(self.layer, "norm_first", False):
            attn = self._sa_block(
                self.layer.norm1(x),
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )
            x = x + attn + self.attention_adapter(attn)
            ff = self._ff_block(self.layer.norm2(x))
            x = x + ff + self.feed_forward_adapter(ff)
            return x

        attn = self._sa_block(
            x,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )
        x = self.layer.norm1(x + attn + self.attention_adapter(attn))
        ff = self._ff_block(x)
        x = self.layer.norm2(x + ff + self.feed_forward_adapter(ff))
        return x


def apply_scgpt_adapters(backbone: ScGPTBackbone, config: AdapterConfig) -> None:
    """Inject adapter-wrapped transformer layers into a scGPT backbone."""

    wrapped_layers = []
    for layer in backbone.transformer_encoder.layers:
        wrapped_layers.append(
            AdapterTransformerEncoderLayer(
                layer,
                config=config,
                d_model=backbone.d_model,
            )
        )
    backbone.transformer_encoder.layers = nn.ModuleList(wrapped_layers)
    if hasattr(backbone.transformer_encoder, "enable_nested_tensor"):
        backbone.transformer_encoder.enable_nested_tensor = False
    if hasattr(backbone.transformer_encoder, "use_nested_tensor"):
        backbone.transformer_encoder.use_nested_tensor = False


__all__ = [
    "AdapterTransformerEncoderLayer",
    "ResidualAdapter",
    "apply_scgpt_adapters",
]
