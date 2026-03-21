# API reference

Use this page as the routing layer for the public API. The main question should
be easy to answer quickly:

- stable baseline workflow: [TaskRunner](./taskrunner.md)
- experimental labeled annotation workflow: [Experimental annotation quickstart API](./annotation.md)
- lower-level control: [Trainer](./trainer.md) plus [Data preparation](./data.md)
- custom extension: [Adapters](./adapters.md)

## What it is

Status: stable and experimental.

This index is the navigation page for the public API contract. Stable and
experimental surfaces are documented separately so users can see the current
scope without guessing.

## When to use it

Use this page when you need to answer one of these quickly:

- which entrypoint should I start with?
- where is the workflow tutorial for that entrypoint?
- where is the parameter and return contract for that API?

## Minimal example

Stable beginner route:

```python
import scanpy as sc
from scdlkit import TaskRunner

adata = sc.datasets.pbmc3k_processed()
runner = TaskRunner(model="vae", task="representation", label_key="louvain")
runner.fit(adata)
```

Experimental beginner route:

```python
from scdlkit import adapt_annotation

runner = adapt_annotation(adata, label_key="cell_type")
```

## Parameters

- Start with [TaskRunner](./taskrunner.md) for bundled stable baselines.
- Start with [Experimental annotation quickstart API](./annotation.md) for labeled annotation adaptation.
- Use [Trainer](./trainer.md) and [Data preparation](./data.md) when you need lower-level control.
- Use [Adapters](./adapters.md) when you already have a custom PyTorch module.

## Input expectations

- stable baseline pages assume processed `AnnData` and bundled models
- experimental annotation pages assume labeled human scRNA-seq and the scGPT `whole-human` checkpoint
- lower-level pages assume you are comfortable managing model and data objects more explicitly

## Returns / outputs

- `TaskRunner` and `adapt_annotation(...)` are the highest-level routes and return fitted workflow objects
- lower-level pages document datasets, raw prediction dictionaries, and wrapped modules in more detail

## Failure modes / raises

- stable pages call out model/task mismatches and missing `AnnData` columns
- experimental pages call out foundation-extra installation, label requirements, and checkpoint-compatibility failures
- lower-level pages call out task-contract and data-contract errors explicitly

## Notes / caveats

- `TaskRunner` remains the stable beginner path.
- `adapt_annotation(...)` is the easiest experimental annotation path.
- `scdlkit.foundation` remains the lower-level experimental surface underneath the beginner alias.

## Related tutorial(s)

- [Scanpy PBMC quickstart](/_tutorials/scanpy_pbmc_quickstart)
- [Custom model extension](/_tutorials/custom_model_extension)
- [Experimental scGPT dataset-specific annotation](/_tutorials/scgpt_dataset_specific_annotation)

## Start here

### High-level stable workflow

- [TaskRunner](./taskrunner.md)

Related tutorial:
- [Scanpy PBMC quickstart](/_tutorials/scanpy_pbmc_quickstart)

### Experimental beginner annotation

- [Experimental annotation quickstart API](./annotation.md)

Related tutorial:
- [Experimental scGPT dataset-specific annotation](/_tutorials/scgpt_dataset_specific_annotation)

### Low-level stable workflow

- [Trainer](./trainer.md)
- [Data preparation](./data.md)

### Extension surfaces

- [Adapters](./adapters.md)
- [Experimental foundation helpers](./foundation.md)

### Evaluation and reports

- [Evaluation and outputs](./evaluation.md)

### Built-in models

- [Built-in models](./models.md)

```{toctree}
:hidden:
:maxdepth: 1

taskrunner
annotation
trainer
data
adapters
foundation
evaluation
models
```
