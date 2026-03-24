"""Prefix-tuning modules for experimental scGPT annotation tuning."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor, nn

from scdlkit.foundation.peft import PrefixTuningConfig
from scdlkit.foundation.scgpt import ScGPTBackbone


class PrefixTuningTransformerEncoder(nn.Module):
    """Transformer encoder wrapper with trainable layer-wise prefix prompts."""

    def __init__(
        self,
        encoder: nn.TransformerEncoder,
        *,
        d_model: int,
        config: PrefixTuningConfig,
    ) -> None:
        super().__init__()
        self.layers = encoder.layers
        self.norm = encoder.norm
        self.prefix_length = config.prefix_length
        self.prefix_dropout = nn.Dropout(config.dropout)
        self.prefix_embeddings = nn.ParameterList(
            [nn.Parameter(torch.empty(config.prefix_length, d_model)) for _ in self.layers]
        )
        for prefix in self.prefix_embeddings:
            nn.init.normal_(prefix, mean=0.0, std=config.init_std)

    def _expanded_mask(self, mask: Tensor | None, *, device: torch.device) -> Tensor | None:
        if mask is None:
            return None
        prefix_mask = torch.zeros(
            (mask.shape[0] + self.prefix_length, mask.shape[1] + self.prefix_length),
            dtype=mask.dtype,
            device=device,
        )
        prefix_mask[self.prefix_length :, self.prefix_length :] = mask
        return prefix_mask

    def forward(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        output = src
        batch_size = int(src.shape[0])
        for layer_index, layer in enumerate(self.layers):
            prefix = self.prefix_dropout(self.prefix_embeddings[layer_index]).to(
                device=output.device,
                dtype=output.dtype,
            )
            prefix = prefix.unsqueeze(0).expand(batch_size, -1, -1)
            layer_input = torch.cat([prefix, output], dim=1)

            expanded_mask = self._expanded_mask(mask, device=output.device)
            if src_key_padding_mask is not None:
                prefix_mask = torch.zeros(
                    (batch_size, self.prefix_length),
                    dtype=src_key_padding_mask.dtype,
                    device=src_key_padding_mask.device,
                )
                key_padding_mask = torch.cat([prefix_mask, src_key_padding_mask], dim=1)
            else:
                key_padding_mask = None

            kwargs: dict[str, Any] = {
                "src_mask": expanded_mask,
                "src_key_padding_mask": key_padding_mask,
            }
            if "is_causal" in layer.forward.__code__.co_varnames:
                kwargs["is_causal"] = is_causal
            try:
                layer_output = layer(layer_input, **kwargs)
            except TypeError:
                kwargs.pop("is_causal", None)
                layer_output = layer(layer_input, **kwargs)
            output = layer_output[:, self.prefix_length :, :]
        if self.norm is not None:
            output = self.norm(output)
        return output


def apply_scgpt_prefix_tuning(backbone: ScGPTBackbone, config: PrefixTuningConfig) -> None:
    """Replace the transformer encoder with a prefix-tuned wrapper."""

    backbone.transformer_encoder = cast(
        Any,
        PrefixTuningTransformerEncoder(
            backbone.transformer_encoder,
            d_model=backbone.d_model,
            config=config,
        ),
    )


__all__ = [
    "PrefixTuningTransformerEncoder",
    "apply_scgpt_prefix_tuning",
]
