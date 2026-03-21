"""Public adapter helpers for custom PyTorch modules.

Use this namespace when you already have an ``nn.Module`` and want to train or
evaluate it through scDLKit's low-level :class:`~scdlkit.training.trainer.Trainer`
workflow without registering it as a built-in model.

Public entrypoints
------------------
TorchModuleAdapter
    Common adapter base used by the task-specific wrappers.
ReconstructionModuleAdapter
    Wrap modules that expose latent and reconstruction-style outputs.
ClassificationModuleAdapter
    Wrap modules that expose class logits for encoded labels.
wrap_reconstruction_module
    Convenience function that returns a reconstruction adapter.
wrap_classification_module
    Convenience function that returns a classification adapter.
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
