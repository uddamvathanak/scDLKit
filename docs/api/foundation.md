# Experimental Foundation Helpers

The foundation helpers are experimental and currently focused on one checkpoint family: scGPT `whole-human`.

Use this section when you want to:

- prepare human scRNA-seq data for the official checkpoint
- extract frozen embeddings
- split tokenized data for annotation fine-tuning
- compare head-only and LoRA tuning through `Trainer`
- use a wrapper-first adaptation path with minimal code

Do not treat this section as a stable general foundation-model abstraction yet.

## Start here

- use `load_scgpt_model(...)` when you only need frozen embeddings
- use `inspect_scgpt_annotation_data(...)` before tuning on a user dataset
- use `adapt_scgpt_annotation(...)` when you want the easiest public adaptation path
- use `ScGPTAnnotationRunner` when you want the wrapper with explicit inspect, fit, predict, annotate, save, and load steps
- use `prepare_scgpt_data(...)` and `split_scgpt_data(...)` when you need a labeled annotation workflow
- use `load_scgpt_annotation_model(...)` when you want a `Trainer`-compatible scGPT classifier

## Wrapper-first adaptation

```python
from scdlkit.foundation import adapt_scgpt_annotation

runner = adapt_scgpt_annotation(
    adata,
    label_key="cell_type",
    output_dir="artifacts/scgpt_annotation",
)
runner.annotate_adata(adata)
runner.save("artifacts/scgpt_annotation/best_model")
```

## Frozen embeddings

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

## Annotation fine-tuning

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

## Current experimental limits

- only `whole-human` is supported
- only human scRNA-seq is supported
- only annotation tuning is supported
- only `head` and `lora` tuning strategies are supported
- full-backbone fine-tuning is deferred
