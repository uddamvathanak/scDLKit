"""Experimental foundation-model helpers.

The current public scope is intentionally narrow:

- frozen scGPT embedding extraction
- experimental scGPT annotation tuning for cell-type labels
- human single-cell RNA data only
- Scanpy handoff through ``adata.obsm``

Treat this namespace as experimental until the embedding and fine-tuning
workflow story is more mature.
"""

from scdlkit.foundation.annotation import (
    ScGPTAnnotationModel,
    load_scgpt_annotation_model,
)
from scdlkit.foundation.cache import ensure_scgpt_checkpoint, list_scgpt_checkpoints
from scdlkit.foundation.data import (
    ScGPTPreparedData,
    ScGPTSplitData,
    prepare_scgpt_data,
    split_scgpt_data,
)
from scdlkit.foundation.lora import ScGPTLoRAConfig
from scdlkit.foundation.scgpt import ScGPTEmbeddingModel, load_scgpt_model

__all__ = [
    "ScGPTAnnotationModel",
    "ScGPTLoRAConfig",
    "ScGPTEmbeddingModel",
    "ScGPTPreparedData",
    "ScGPTSplitData",
    "ensure_scgpt_checkpoint",
    "list_scgpt_checkpoints",
    "load_scgpt_annotation_model",
    "load_scgpt_model",
    "prepare_scgpt_data",
    "split_scgpt_data",
]
