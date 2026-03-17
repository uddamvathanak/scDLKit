"""Public adapter helpers for custom PyTorch modules.

Use this namespace when you already have an ``nn.Module`` and want to train or
evaluate it through scDLKit's low-level :class:`~scdlkit.training.trainer.Trainer`
workflow without registering it as a built-in model.
"""

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
