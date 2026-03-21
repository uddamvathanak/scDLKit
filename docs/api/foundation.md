# Experimental Foundation Helpers

## What it is

Status: experimental.

This page documents the explicit lower-level scGPT path underneath
`scdlkit.adapt_annotation(...)`. It is the place to go when you want direct
control over frozen scGPT embeddings, tokenized datasets, split-aware
annotation training, or the underlying wrapper objects.

## When to use it

Use this page when you want to:

- extract frozen scGPT embeddings directly
- prepare tokenized scGPT data for your own workflow
- split tokenized data for annotation fine-tuning
- load a `Trainer`-compatible scGPT annotation model explicitly
- drop below the top-level beginner alias and inspect the scGPT-specific objects

## Minimal example

```python
from scdlkit.foundation import (
    load_scgpt_annotation_model,
    prepare_scgpt_data,
    split_scgpt_data,
)
from scdlkit import Trainer

prepared = prepare_scgpt_data(adata, label_key="cell_type")
split = split_scgpt_data(prepared)
model = load_scgpt_annotation_model(
    num_classes=len(prepared.label_categories or ()),
    label_categories=prepared.label_categories,
    tuning_strategy="head",
)
trainer = Trainer(model=model, task="classification", batch_size=prepared.batch_size)
trainer.fit(split.train, split.val)
```

## Parameters

- `load_scgpt_model(...)` loads the official `whole-human` checkpoint for frozen embeddings.
- `prepare_scgpt_data(...)` tokenizes compatible human `AnnData` and optionally encodes labels.
- `split_scgpt_data(...)` creates train, validation, and test subsets without re-tokenizing.
- `load_scgpt_annotation_model(...)` builds a `head` or `lora` scGPT classifier for `Trainer`.
- `ScGPTAnnotationRunner` and `adapt_scgpt_annotation(...)` expose the explicit wrapper-first foundation path.

## Input expectations

- input must be human scRNA-seq in `AnnData`.
- the checkpoint scope is currently limited to scGPT `whole-human`.
- expression values must be non-negative.
- annotation tuning requires a valid `label_key` with at least two label categories.
- sufficient gene overlap with the checkpoint vocabulary is required; otherwise preparation raises a clear error.

## Returns / outputs

- `ScGPTPreparedData` stores tokenized tensors plus checkpoint and label metadata.
- `ScGPTSplitData` stores split-aware token datasets for training and evaluation.
- `load_scgpt_model(...)` returns an embedding model for frozen inference.
- `load_scgpt_annotation_model(...)` returns a classification model ready for `Trainer(..., task="classification")`.
- `ScGPTAnnotationRunner` and `adapt_scgpt_annotation(...)` can emit reports, plots, predictions, and saved runner state.

## Failure modes / raises

- `ImportError` if the package was installed without `scdlkit[foundation]`.
- `ValueError` if labels are missing, the tuning strategy is unsupported, or the checkpoint vocabulary overlap is too small.
- `ValueError` if expression values are negative.
- `RuntimeError` if wrapper prediction or save/load methods are called in the wrong lifecycle stage.

## Notes / caveats

- The recommended beginner route is still [Experimental annotation quickstart API](./annotation.md).
- This page documents the lower-level implementation and is intentionally narrower than a general foundation-model framework.
- Supported scope remains:
  - human scRNA-seq only
  - scGPT `whole-human` only
  - annotation tuning only
  - `head` and `lora` as the trainable strategies

## Related tutorial(s)

- [Experimental scGPT PBMC embeddings](/_tutorials/scgpt_pbmc_embeddings)
- [Experimental scGPT cell-type annotation](/_tutorials/scgpt_cell_type_annotation)
- [Experimental scGPT dataset-specific annotation](/_tutorials/scgpt_dataset_specific_annotation)

```{eval-rst}
.. autoclass:: scdlkit.foundation.ScGPTPreparedData
   :members:
```

```{eval-rst}
.. autoclass:: scdlkit.foundation.ScGPTEmbeddingModel
   :members:
```

```{eval-rst}
.. autofunction:: scdlkit.foundation.ensure_scgpt_checkpoint
```

```{eval-rst}
.. autofunction:: scdlkit.foundation.list_scgpt_checkpoints
```

```{eval-rst}
.. autofunction:: scdlkit.foundation.load_scgpt_model
```

```{eval-rst}
.. autofunction:: scdlkit.foundation.prepare_scgpt_data
```

```{eval-rst}
.. autoclass:: scdlkit.foundation.ScGPTAnnotationDataReport
   :members:
```

```{eval-rst}
.. autoclass:: scdlkit.foundation.ScGPTSplitData
   :members:
```

```{eval-rst}
.. autoclass:: scdlkit.foundation.ScGPTLoRAConfig
   :members:
```

```{eval-rst}
.. autoclass:: scdlkit.foundation.ScGPTAnnotationModel
   :members:
```

```{eval-rst}
.. autoclass:: scdlkit.foundation.ScGPTAnnotationRunSummary
   :members:
```

```{eval-rst}
.. autoclass:: scdlkit.foundation.ScGPTAnnotationRunner
   :members:
```

```{eval-rst}
.. autofunction:: scdlkit.foundation.inspect_scgpt_annotation_data
```

```{eval-rst}
.. autofunction:: scdlkit.foundation.split_scgpt_data
```

```{eval-rst}
.. autofunction:: scdlkit.foundation.load_scgpt_annotation_model
```

```{eval-rst}
.. autofunction:: scdlkit.foundation.adapt_scgpt_annotation
```
