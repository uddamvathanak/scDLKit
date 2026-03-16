"""Base adapter abstractions for custom PyTorch modules."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

import torch
from torch import nn

from scdlkit.models.base import BaseModel

ForwardFn = Callable[[nn.Module, torch.Tensor], torch.Tensor | dict[str, torch.Tensor]]
LossFn = Callable[
    [nn.Module, dict[str, torch.Tensor], dict[str, torch.Tensor]],
    tuple[torch.Tensor, dict[str, float]],
]


class TorchModuleAdapter(BaseModel):
    """Base wrapper around a raw ``nn.Module``."""

    supported_tasks: tuple[str, ...]

    def __init__(
        self,
        *,
        module: nn.Module,
        input_dim: int,
        supported_tasks: tuple[str, ...],
    ) -> None:
        super().__init__(input_dim=input_dim)
        self.module = module
        self.supported_tasks = tuple(supported_tasks)

    def _normalize_output_dict(
        self,
        outputs: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        normalized: dict[str, torch.Tensor] = {}
        for key, value in outputs.items():
            if not isinstance(value, torch.Tensor):
                msg = f"Adapter output '{key}' must be a torch.Tensor."
                raise TypeError(msg)
            normalized[key] = value
        return normalized

    def _invoke_forward(
        self,
        x: torch.Tensor,
        forward_fn: ForwardFn | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        callback = forward_fn or cast(ForwardFn, lambda module, tensor: module(tensor))
        return callback(self.module, x)

    @staticmethod
    def _normalize_stats(loss: torch.Tensor, stats: dict[str, float]) -> dict[str, float]:
        normalized = dict(stats)
        normalized.setdefault("loss", float(loss.detach().cpu()))
        return normalized

    def compute_task_loss(
        self,
        task_name: str,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
        msg = f"{self.__class__.__name__} does not implement task-aware loss for '{task_name}'."
        raise NotImplementedError(msg)
