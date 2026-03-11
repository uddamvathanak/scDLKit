"""Reconstruction task adapter."""

from __future__ import annotations

from typing import cast

import torch
from torch.nn import functional

from scdlkit.tasks.base import BaseTask, register_task


class ReconstructionTask(BaseTask):
    name = "reconstruction"
    metric_group = "reconstruction"

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
        outputs = model(batch["x"])
        reconstruction = outputs["reconstruction"]
        recon_loss = functional.mse_loss(reconstruction, batch["x"])
        total_loss = recon_loss
        stats = {
            "loss": float(total_loss.detach().cpu()),
            "reconstruction_loss": float(recon_loss.detach().cpu()),
        }
        if "mu" in outputs and "logvar" in outputs and hasattr(model, "kl_weight"):
            mu = outputs["mu"]
            logvar = outputs["logvar"]
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            kl_weight = cast(float, model.kl_weight)
            total_loss = total_loss + float(kl_weight) * kl
            stats["kl_loss"] = float(kl.detach().cpu())
            stats["loss"] = float(total_loss.detach().cpu())
        return total_loss, stats, outputs


register_task(ReconstructionTask())
