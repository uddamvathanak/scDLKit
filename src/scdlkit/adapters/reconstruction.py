"""Adapters for reconstruction and representation modules."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.nn import functional

from scdlkit.adapters.base import ForwardFn, LossFn, TorchModuleAdapter


class ReconstructionModuleAdapter(TorchModuleAdapter):
    """Wrap a custom module for reconstruction and representation tasks."""

    _VALID_TASKS = ("reconstruction", "representation")

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        input_dim: int,
        supported_tasks: tuple[str, ...] = ("reconstruction",),
        forward_fn: ForwardFn | None = None,
        encode_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor] | None = None,
        loss_fn: LossFn | None = None,
    ) -> None:
        normalized_supported_tasks = tuple(supported_tasks)
        invalid_tasks = set(normalized_supported_tasks) - set(self._VALID_TASKS)
        if invalid_tasks:
            msg = (
                "ReconstructionModuleAdapter only supports "
                f"{self._VALID_TASKS}. Invalid tasks: {sorted(invalid_tasks)}"
            )
            raise ValueError(msg)
        if not normalized_supported_tasks:
            msg = "ReconstructionModuleAdapter requires at least one supported task."
            raise ValueError(msg)
        super().__init__(
            module=module,
            input_dim=input_dim,
            supported_tasks=normalized_supported_tasks,
        )
        self._forward_fn = forward_fn
        self._encode_fn = encode_fn
        self._loss_fn = loss_fn

    def _resolve_latent(self, x: torch.Tensor) -> torch.Tensor | None:
        if self._encode_fn is not None:
            return self._encode_fn(self.module, x)
        encode_method = getattr(self.module, "encode", None)
        if callable(encode_method):
            latent = encode_method(x)
            if not isinstance(latent, torch.Tensor):
                msg = "Wrapped module encode(x) must return a torch.Tensor."
                raise TypeError(msg)
            return latent
        return None

    def _normalize_outputs(
        self,
        outputs: torch.Tensor | dict[str, torch.Tensor],
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if isinstance(outputs, torch.Tensor):
            normalized: dict[str, torch.Tensor] = {"reconstruction": outputs}
        elif isinstance(outputs, dict):
            normalized = self._normalize_output_dict(outputs)
        else:
            msg = "Reconstruction adapters must return a torch.Tensor or dict[str, torch.Tensor]."
            raise TypeError(msg)
        if "reconstruction" not in normalized:
            msg = "Reconstruction adapters must provide a 'reconstruction' tensor."
            raise ValueError(msg)
        if "representation" in self.supported_tasks and "latent" not in normalized:
            latent = self._resolve_latent(x)
            if latent is None:
                msg = (
                    "Representation adapters must provide a latent path via "
                    "outputs['latent'], encode_fn, or module.encode(x)."
                )
                raise ValueError(msg)
            normalized["latent"] = latent
        return normalized

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self._invoke_forward(x, self._forward_fn)
        return self._normalize_outputs(outputs, x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(x)
        try:
            return outputs["latent"]
        except KeyError as exc:
            msg = f"{self.__class__.__name__} does not expose latent encodings."
            raise NotImplementedError(msg) from exc

    def compute_task_loss(
        self,
        task_name: str,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
        if task_name not in self.supported_tasks:
            msg = f"{self.__class__.__name__} does not support task '{task_name}'."
            raise ValueError(msg)
        outputs = self.forward(batch["x"])
        if self._loss_fn is not None:
            loss, stats = self._loss_fn(self.module, batch, outputs)
            return loss, self._normalize_stats(loss, stats), outputs
        loss = functional.mse_loss(outputs["reconstruction"], batch["x"])
        stats = {
            "loss": float(loss.detach().cpu()),
            "reconstruction_loss": float(loss.detach().cpu()),
        }
        return loss, stats, outputs


def wrap_reconstruction_module(
    module: torch.nn.Module,
    *,
    input_dim: int,
    supported_tasks: tuple[str, ...] = ("reconstruction",),
    forward_fn: ForwardFn | None = None,
    encode_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor] | None = None,
    loss_fn: LossFn | None = None,
) -> ReconstructionModuleAdapter:
    """Create a reconstruction/representation adapter around a raw ``nn.Module``."""

    return ReconstructionModuleAdapter(
        module=module,
        input_dim=input_dim,
        supported_tasks=supported_tasks,
        forward_fn=forward_fn,
        encode_fn=encode_fn,
        loss_fn=loss_fn,
    )
