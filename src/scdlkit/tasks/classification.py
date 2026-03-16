"""Classification task adapter."""

from __future__ import annotations

import torch
from torch.nn import functional

from scdlkit.tasks.base import BaseTask, register_task


class ClassificationTask(BaseTask):
    name = "classification"
    metric_group = "classification"
    requires_labels = True

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
        custom_compute_loss = getattr(model, "compute_task_loss", None)
        if callable(custom_compute_loss):
            return custom_compute_loss(self.name, batch)
        logits = model(batch["x"])["logits"]
        loss = functional.cross_entropy(logits, batch["y"])
        outputs = {"logits": logits}
        stats = {"loss": float(loss.detach().cpu())}
        return loss, stats, outputs


register_task(ClassificationTask())
