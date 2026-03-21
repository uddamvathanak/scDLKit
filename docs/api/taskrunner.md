# TaskRunner

## What it is

Status: stable.

`TaskRunner` is the main beginner workflow in scDLKit. It keeps the common
baseline path compact:

1. start from a processed `AnnData`
2. choose a bundled scDLKit model
3. train and evaluate it
4. recover embeddings or reconstructed expression values
5. continue in Scanpy

## When to use it

Use `TaskRunner` when:

- you want the shortest stable path from `AnnData` to a model result
- you are using bundled scDLKit baselines such as `vae` or `transformer_ae`
- you want embeddings back for `adata.obsm`
- you want reconstructed or predicted expression values from reconstruction-capable models

Use [Trainer](./trainer.md) instead when you need lower-level control or custom wrapped models.

## Minimal example

```python
import scanpy as sc
from scdlkit import TaskRunner

adata = sc.datasets.pbmc3k_processed()

runner = TaskRunner(
    model="vae",
    task="representation",
    label_key="louvain",
    device="auto",
    epochs=20,
    batch_size=128,
    model_kwargs={"kl_weight": 1e-3},
)

runner.fit(adata)
adata.obsm["X_scdlkit_vae"] = runner.encode(adata)
reconstructed = runner.reconstruct(adata)
```

## Parameters

- `model`: built-in model name such as `vae`, `autoencoder`, `transformer_ae`, or `mlp_classifier`, or an instantiated scDLKit model.
- `task`: one of `representation`, `reconstruction`, or `classification`.
- `label_key`: optional `adata.obs` column used for supervised metrics or classification.
- `batch_key`: optional `adata.obs` column for batch-aware splits and metrics.
- `layer`, `use_hvg`, `normalize`, `log1p`, `scale`: preprocessing controls applied before training.
- `epochs`, `batch_size`, `lr`, `device`, `mixed_precision`: training and inference defaults.
- `output_dir`: optional report/checkpoint directory.

## Input expectations

- `adata` must be an `anndata.AnnData` object with cells in `obs` and genes in `var`.
- `adata.X` or the selected layer must be numeric and feature-consistent between training and inference.
- `label_key` must exist in `adata.obs` for classification tasks or supervised evaluation.
- `reconstruct(...)` and `encode(...)` require a fitted runner and a task that exposes those outputs.

## Returns / outputs

- `fit(...)` returns the fitted `TaskRunner`.
- `encode(...)` returns a `numpy.ndarray` latent matrix suitable for `adata.obsm`.
- `reconstruct(...)` returns a `numpy.ndarray` of reconstructed or predicted expression values.
- `evaluate(...)` returns a metric dictionary.
- `save_report(...)` writes a Markdown report and sibling CSV table.

## Failure modes / raises

- `ValueError` if the selected model does not support the requested task.
- `ValueError` if `label_key` or `batch_key` is missing from `adata.obs`.
- `RuntimeError` if inference or plotting is called before `fit(...)`.
- `ValueError` if you request latent or reconstruction outputs from a classification-only workflow.

## Notes / caveats

- `TaskRunner` is the stable bundled-model path; it is not the annotation fine-tuning surface.
- Classification models expose class predictions rather than reconstructed expression.
- For explicit lower-level control, use [Trainer](./trainer.md) and [Data preparation](./data.md).

## Related tutorial(s)

- [Scanpy PBMC quickstart](/_tutorials/scanpy_pbmc_quickstart)
- [Downstream Scanpy after scDLKit](/_tutorials/downstream_scanpy_after_scdlkit)
- [Reconstruction sanity check](/_tutorials/reconstruction_sanity_pbmc)
- [PBMC model comparison](/_tutorials/pbmc_model_comparison)

```{eval-rst}
.. autoclass:: scdlkit.runner.TaskRunner
   :members:
```
