"""Base interfaces for scDLKit models."""

from __future__ import annotations

import torch
from torch import nn


class BaseModel(nn.Module):
    """Common base class for all registered models."""

    supported_tasks: tuple[str, ...] = ()

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        msg = f"{self.__class__.__name__} does not expose latent encodings"
        raise NotImplementedError(msg)
