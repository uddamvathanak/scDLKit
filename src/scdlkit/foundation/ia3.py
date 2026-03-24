"""IA3-style scaling modules for experimental scGPT annotation tuning."""

from __future__ import annotations

from torch import Tensor, nn

from scdlkit.foundation.peft import IA3Config
from scdlkit.foundation.scgpt import ScGPTBackbone


class IA3Linear(nn.Module):
    """Linear wrapper with learned multiplicative output scaling."""

    def __init__(self, base_layer: nn.Linear, *, init_scale: float) -> None:
        super().__init__()
        self.base_layer = base_layer
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False
        self.ia3_scale = nn.Parameter(
            base_layer.weight.new_full((base_layer.out_features,), float(init_scale))
        )

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.base_layer(inputs)
        view_shape = (1,) * (outputs.ndim - 1) + (-1,)
        return outputs * self.ia3_scale.view(*view_shape)

    @property
    def weight(self) -> Tensor:
        return self.base_layer.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base_layer.bias


def apply_scgpt_ia3(backbone: ScGPTBackbone, config: IA3Config) -> None:
    """Inject IA3-style scaling modules into a scGPT backbone."""

    for layer in backbone.transformer_encoder.layers:
        if "out_proj" in config.target_modules:
            layer.self_attn.out_proj = IA3Linear(
                layer.self_attn.out_proj,
                init_scale=config.init_scale,
            )
        if "linear1" in config.target_modules:
            layer.linear1 = IA3Linear(
                layer.linear1,
                init_scale=config.init_scale,
            )


__all__ = [
    "IA3Linear",
    "apply_scgpt_ia3",
]
