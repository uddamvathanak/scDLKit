"""Adapters for classification modules."""

from __future__ import annotations

import torch
from torch.nn import functional

from scdlkit.adapters.base import ForwardFn, LossFn, TorchModuleAdapter


class ClassificationModuleAdapter(TorchModuleAdapter):
    """Wrap a custom module for classification tasks."""

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        input_dim: int,
        forward_fn: ForwardFn | None = None,
        loss_fn: LossFn | None = None,
    ) -> None:
        super().__init__(
            module=module,
            input_dim=input_dim,
            supported_tasks=("classification",),
        )
        self._forward_fn = forward_fn
        self._loss_fn = loss_fn

    def _normalize_outputs(
        self,
        outputs: torch.Tensor | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if isinstance(outputs, torch.Tensor):
            normalized: dict[str, torch.Tensor] = {"logits": outputs}
        elif isinstance(outputs, dict):
            normalized = self._normalize_output_dict(outputs)
        else:
            msg = "Classification adapters must return a torch.Tensor or dict[str, torch.Tensor]."
            raise TypeError(msg)
        if "logits" not in normalized:
            msg = "Classification adapters must provide a 'logits' tensor."
            raise ValueError(msg)
        return normalized

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self._invoke_forward(x, self._forward_fn)
        return self._normalize_outputs(outputs)

    def compute_task_loss(
        self,
        task_name: str,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
        if task_name != "classification":
            msg = f"{self.__class__.__name__} does not support task '{task_name}'."
            raise ValueError(msg)
        outputs = self.forward(batch["x"])
        if self._loss_fn is not None:
            loss, stats = self._loss_fn(self.module, batch, outputs)
            return loss, self._normalize_stats(loss, stats), outputs
        loss = functional.cross_entropy(outputs["logits"], batch["y"])
        stats = {"loss": float(loss.detach().cpu())}
        return loss, stats, outputs


def wrap_classification_module(
    module: torch.nn.Module,
    *,
    input_dim: int,
    forward_fn: ForwardFn | None = None,
    loss_fn: LossFn | None = None,
) -> ClassificationModuleAdapter:
    """Create a classification adapter around a raw ``nn.Module``."""

    return ClassificationModuleAdapter(
        module=module,
        input_dim=input_dim,
        forward_fn=forward_fn,
        loss_fn=loss_fn,
    )
