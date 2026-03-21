"""Experimental foundation-model helpers.

The current public scope is intentionally narrow:

- frozen scGPT embedding extraction
- experimental scGPT annotation tuning for cell-type labels
- wrapper-first experimental scGPT dataset adaptation
- top-level ``scdlkit.adapt_annotation`` and ``scdlkit.AnnotationRunner`` are
  the easiest public entrypoints
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
    ScGPTAnnotationDataReport,
    ScGPTPreparedData,
    ScGPTSplitData,
    inspect_scgpt_annotation_data,
    prepare_scgpt_data,
    split_scgpt_data,
)
from scdlkit.foundation.lora import ScGPTLoRAConfig
from scdlkit.foundation.runner import (
    ScGPTAnnotationRunner,
    ScGPTAnnotationRunSummary,
    adapt_scgpt_annotation,
)
from scdlkit.foundation.scgpt import ScGPTEmbeddingModel, load_scgpt_model

__all__ = [
    "ScGPTAnnotationDataReport",
    "ScGPTAnnotationModel",
    "ScGPTAnnotationRunSummary",
    "ScGPTAnnotationRunner",
    "ScGPTLoRAConfig",
    "ScGPTEmbeddingModel",
    "ScGPTPreparedData",
    "ScGPTSplitData",
    "adapt_scgpt_annotation",
    "ensure_scgpt_checkpoint",
    "inspect_scgpt_annotation_data",
    "list_scgpt_checkpoints",
    "load_scgpt_annotation_model",
    "load_scgpt_model",
    "prepare_scgpt_data",
    "split_scgpt_data",
]
