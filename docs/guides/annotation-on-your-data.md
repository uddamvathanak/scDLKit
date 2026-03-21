# Experimental Annotation On Your Data

Use this guide when you have a labeled human `AnnData` object and want the easiest public path for experimental scGPT adaptation.

Related APIs:

- [Experimental annotation quickstart API](../api/annotation.md)
- [Experimental foundation helpers](../api/foundation.md)

The current scope is still narrow:

- human scRNA-seq only
- official scGPT `whole-human` checkpoint only
- annotation only
- strategies limited to:
  - frozen probe
  - head-only tuning
  - LoRA tuning

## Required input shape

Your dataset should provide:

- cells in `adata.obs`
- genes in `adata.var_names`
- non-negative expression values in `adata.X` or `adata.raw.X`
- a categorical label column in `adata.obs`

For the current experimental wrapper, the most important fields are:

- `adata.var_names`
- `adata.obs["your_label_key"]`

## Start with inspection

Run the inspection step before training:

```python
from scdlkit import inspect_annotation_data

report = inspect_annotation_data(
    adata,
    label_key="cell_type",
    checkpoint="whole-human",
)
```

Inspect these fields first:

- `num_genes_matched`
- `gene_overlap_ratio`
- `class_counts`
- `min_class_count`
- `warnings`

If the report shows low overlap or very small classes, the wrapper may still run, but the result should be treated with more caution.

## Fastest adaptation path

```python
from scdlkit import adapt_annotation

runner = adapt_annotation(
    adata,
    label_key="cell_type",
    output_dir="artifacts/scgpt_annotation",
)
```

This one call:

- inspects the dataset
- prepares and splits the tokenized data
- compares frozen probe and head-only tuning by default
- keeps the best fitted strategy
- writes the standard artifact bundle

LoRA remains available by explicit opt-in through `strategies=("frozen_probe", "head", "lora")`.

## Write results back into `AnnData`

```python
runner.annotate_adata(
    adata,
    obs_key="scgpt_label",
    embedding_key="X_scgpt_best",
)
```

This writes:

- predicted labels to `adata.obs["scgpt_label"]`
- label codes to `adata.obs["scgpt_label_code"]`
- max confidence to `adata.obs["scgpt_label_confidence"]`
- latent embedding to `adata.obsm["X_scgpt_best"]`

That keeps the downstream Scanpy handoff simple.

## Save and reload the best fitted runner

```python
save_dir = runner.save("artifacts/scgpt_annotation/best_model")

from scdlkit import AnnotationRunner

reloaded = AnnotationRunner.load(save_dir, device="auto")
```

The saved directory contains:

- `manifest.json`
- `model_state.pt`

The base `whole-human` checkpoint is not vendored into the saved artifact. Reloading resolves it from the local cache.

## When to drop down to the lower-level API

Use the wrapper first when you want:

- minimal code
- sensible defaults
- built-in strategy comparison
- straightforward `AnnData` write-back

Drop down to the `Trainer` path when you want:

- tighter control over epochs and optimizer settings
- custom evaluation code
- notebook-level debugging of the low-level training loop

## What remains experimental

- no full-backbone fine-tuning
- no non-human support
- no checkpoints beyond `whole-human`
- no perturbation, spatial, or multimodal workflows
- no claim that scGPT always beats classical baselines

The main product value of this path is not universal superiority. It is the ability to compare adaptation strategies on your own labeled dataset with a reproducible, Scanpy-compatible workflow.

Under the hood, this top-level beginner path is still backed by the experimental scGPT `whole-human` workflow in `scdlkit.foundation`.
