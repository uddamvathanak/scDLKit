"""Adapter utilities for wrapping custom PyTorch modules."""

from scdlkit.adapters.base import TorchModuleAdapter
from scdlkit.adapters.classification import (
    ClassificationModuleAdapter,
    wrap_classification_module,
)
from scdlkit.adapters.reconstruction import (
    ReconstructionModuleAdapter,
    wrap_reconstruction_module,
)

__all__ = [
    "TorchModuleAdapter",
    "ReconstructionModuleAdapter",
    "ClassificationModuleAdapter",
    "wrap_reconstruction_module",
    "wrap_classification_module",
]
