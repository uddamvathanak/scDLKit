"""Experimental foundation-model helpers.

The current public scope is intentionally narrow:

- frozen scGPT embedding extraction
- human single-cell RNA data only
- Scanpy handoff through ``adata.obsm``

Treat this namespace as experimental until the embedding workflow and tutorial
story are more mature.
"""

from scdlkit.foundation.cache import ensure_scgpt_checkpoint, list_scgpt_checkpoints
from scdlkit.foundation.data import ScGPTPreparedData, prepare_scgpt_data
from scdlkit.foundation.scgpt import ScGPTEmbeddingModel, load_scgpt_model

__all__ = [
    "ScGPTEmbeddingModel",
    "ScGPTPreparedData",
    "ensure_scgpt_checkpoint",
    "list_scgpt_checkpoints",
    "load_scgpt_model",
    "prepare_scgpt_data",
]
