# Foundation Models

`scDLKit` now includes an experimental scGPT path for human scRNA-seq workflows.

Related APIs:

- [Experimental annotation quickstart API](../api/annotation.md)
- [Experimental foundation helpers](../api/foundation.md)

The public scope is still deliberately narrow:

- official `whole-human` checkpoint only
- human single-cell RNA only
- wrapper-first helpers for beginners, `Trainer` plus `scdlkit.foundation` helpers underneath
- frozen embeddings remain supported
- experimental cell-type annotation fine-tuning is now supported through:
  - frozen linear probe
  - head-only tuning
  - LoRA tuning
- no `TaskRunner` support yet
- no full-backbone fine-tuning yet

## Install

```bash
python -m pip install "scdlkit[foundation,tutorials]"
```

## What this path is for

Use the experimental foundation path when you want to:

- extract frozen cell embeddings from an official scGPT checkpoint
- compare those embeddings against `PCA` and scDLKit baselines
- fine-tune scGPT for a labeled cell-type annotation task
- decide whether your dataset needs:
  - only frozen embeddings
  - a trainable classification head
  - parameter-efficient LoRA tuning

This is the bridge between the baseline toolkit and later foundation-model adaptation work.

## Easiest wrapper-first path

If you want the smallest amount of code, start with the top-level experimental alias:

```python
from scdlkit import adapt_annotation

runner = adapt_annotation(
    adata,
    label_key="cell_type",
    output_dir="artifacts/scgpt_annotation",
)

runner.annotate_adata(adata)
runner.save("artifacts/scgpt_annotation/best_model")
```

This wrapper:

- inspects the labeled dataset
- compares frozen probe and head-only tuning by default
- keeps the best fitted strategy in memory
- writes standard report artifacts
- makes it easy to annotate `AnnData` and save the best fitted runner

LoRA remains available by explicit opt-in through `strategies=("frozen_probe", "head", "lora")`.

## Inspect before training

For user-supplied datasets, inspect first through the top-level beginner alias:

```python
from scdlkit import inspect_annotation_data

report = inspect_annotation_data(
    adata,
    label_key="cell_type",
    checkpoint="whole-human",
)
```

This is the recommended preflight step when you want to know whether gene overlap
or class balance is likely to make the adaptation path brittle.

## Frozen embedding API

```python
from scdlkit import Trainer
from scdlkit.foundation import load_scgpt_model, prepare_scgpt_data

prepared = prepare_scgpt_data(
    adata,
    checkpoint="whole-human",
    label_key="louvain",
    batch_size=64,
)

model = load_scgpt_model("whole-human", device="auto")
trainer = Trainer(
    model=model,
    task="representation",
    batch_size=prepared.batch_size,
    device="auto",
)

predictions = trainer.predict_dataset(prepared.dataset)
adata.obsm["X_scgpt_whole_human"] = predictions["latent"]
```

## Annotation fine-tuning API

```python
from scdlkit import Trainer
from scdlkit.foundation import (
    load_scgpt_annotation_model,
    prepare_scgpt_data,
    split_scgpt_data,
)

prepared = prepare_scgpt_data(
    adata,
    checkpoint="whole-human",
    label_key="louvain",
    batch_size=64,
)
split = split_scgpt_data(prepared, val_size=0.15, test_size=0.15, random_state=42)

model = load_scgpt_annotation_model(
    num_classes=len(prepared.label_categories or ()),
    checkpoint="whole-human",
    tuning_strategy="lora",
    label_categories=prepared.label_categories,
    device="auto",
)

trainer = Trainer(
    model=model,
    task="classification",
    batch_size=prepared.batch_size,
    device="auto",
    epochs=8,
)
trainer.fit(split.train, split.val)
predictions = trainer.predict_dataset(split.test)
adata.obsm["X_scgpt_lora"] = predictions["latent"]
```

## Wrapper class for advanced convenience

If you want the easy path but still need explicit control over the fitted object,
use the top-level runner alias directly:

```python
from scdlkit import AnnotationRunner

runner = AnnotationRunner(label_key="cell_type", output_dir="artifacts/scgpt_annotation")
runner.inspect(adata)
runner.fit_compare(adata)
runner.annotate_adata(adata)
runner.save("artifacts/scgpt_annotation/best_model")
```

The lower-level `Trainer` path and `scdlkit.foundation` helpers remain the advanced public surface underneath the wrapper.

## When to use each strategy

- frozen linear probe:
  - best first question
  - tells you whether the checkpoint already separates your labels
- head-only tuning:
  - cheapest trainable path
  - use when the frozen probe is useful but not good enough
- LoRA tuning:
  - first parameter-efficient adaptation path
  - use when you want more flexibility than a frozen backbone plus head

## Current limitations

- input preparation is a separate tokenized pipeline, not `prepare_data(...)`
- only the official `whole-human` checkpoint is supported
- user datasets must have labels for annotation fine-tuning
- gene overlap with the checkpoint vocabulary still gates compatibility
- full-backbone fine-tuning is intentionally deferred
- this release does not claim perturbation, spatial, or multimodal support

## Tutorials

- frozen embeddings: [Experimental scGPT PBMC embeddings](/_tutorials/scgpt_pbmc_embeddings)
- annotation tuning: [Experimental scGPT cell-type annotation](/_tutorials/scgpt_cell_type_annotation)
- dataset-specific wrapper workflow: [Experimental scGPT dataset-specific annotation](/_tutorials/scgpt_dataset_specific_annotation)
- beyond-PBMC wrapper workflow: [Experimental scGPT human-pancreas annotation](/_tutorials/scgpt_human_pancreas_annotation)
- benchmark framing: [Annotation benchmarks](./annotation-benchmarks.md)
- user-data guide: [Experimental annotation on your data](/guides/annotation-on-your-data)

## Positioning

This feature should be treated as experimental.

The goal is not to claim that foundation models always beat classical baselines. The goal is to give users a reproducible, Scanpy-compatible workflow to compare:

- `PCA + logistic regression`
- frozen scGPT linear probe
- head-only tuning
- LoRA tuning

That comparison story is the main product value of the current foundation release line.

The current beyond-PBMC evidence phase uses OpenProblems human pancreas to
show the same wrapper-first workflow on a second labeled human dataset.

Treat `scdlkit.foundation` as the explicit lower-level experimental namespace that sits underneath the easier top-level beginner aliases.
