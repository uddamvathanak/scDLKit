"""Task abstractions for training and evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseTask(ABC):
    """Task adapter for model-specific training behavior."""

    name: str
    metric_group: str
    requires_labels: bool = False

    @abstractmethod
    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
        """Compute the batch loss and return detached stats."""


_TASKS: dict[str, BaseTask] = {}


def register_task(task: BaseTask) -> BaseTask:
    _TASKS[task.name] = task
    return task


def get_task(name: str) -> BaseTask:
    try:
        return _TASKS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_TASKS))
        msg = f"Unknown task '{name}'. Available tasks: {available}"
        raise ValueError(msg) from exc
