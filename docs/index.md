# scDLKit

scDLKit helps you train, compare, and reuse deep-learning workflows for
single-cell data while staying inside the familiar `AnnData` and Scanpy
ecosystem.

Use it when you want:

- baseline autoencoder, VAE, transformer, and classifier workflows without
  writing the PyTorch loop yourself
- embeddings written back to `adata.obsm` for downstream Scanpy analysis
- reproducible metrics, reports, and tutorial artifacts
- experimental scGPT cell-type annotation and fine-tuning on labeled human
  scRNA-seq data

## Start with your task

````{grid} 1 2 2 2
:gutter: 3

```{grid-item-card} Train a baseline embedding
:link: _tutorials/scanpy_pbmc_quickstart
:link-type: doc

Fit a VAE or other bundled model, save the learned representation in
`adata.obsm`, then continue with neighbors, UMAP, Leiden clustering, and plots
in Scanpy.
```

```{grid-item-card} Fine-tune for cell-type annotation
:link: _tutorials/scgpt_human_pancreas_annotation
:link-type: doc

Use `adapt_annotation(...)` to compare frozen scGPT embeddings with a
head-tuned annotation model, annotate cells, and save the fitted runner.
```

```{grid-item-card} Compare model behavior
:link: _tutorials/pbmc_model_comparison
:link-type: doc

Run compact comparisons across classical and deep-learning baselines and review
the resulting metrics and plots.
```

```{grid-item-card} Check current scope
:link: roadmap
:link-type: doc

See what is available now, what is experimental, and what is planned for the
next research milestones.
```
````

## Stable baseline path

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
```

Continue in Scanpy:

```python
sc.pp.neighbors(adata, use_rep="X_scdlkit_vae")
sc.tl.umap(adata)
sc.pl.umap(adata, color="louvain")
```

Useful links:

- [Install scDLKit](./install.md)
- [TaskRunner API](./api/taskrunner.md)
- [Scanpy PBMC quickstart](/_tutorials/scanpy_pbmc_quickstart)
- [Downstream Scanpy after scDLKit](/_tutorials/downstream_scanpy_after_scdlkit)

## Experimental scGPT annotation path

```python
from scdlkit import adapt_annotation

runner = adapt_annotation(
    adata,
    label_key="cell_type",
    output_dir="artifacts/scgpt_annotation",
)

runner.annotate_adata(
    adata,
    obs_key="scgpt_label",
    embedding_key="X_scgpt_best",
)
runner.save("artifacts/scgpt_annotation/best_model")
```

This path is experimental and intentionally narrow in `0.1.7`:

- human scRNA-seq data only
- official scGPT `whole-human` checkpoint only
- labeled cell-type annotation only
- default quickstart comparison: `frozen_probe` and `head`

Useful links:

- [Main annotation tutorial: human-pancreas workflow](/_tutorials/scgpt_human_pancreas_annotation)
- [Experimental annotation API](./api/annotation.md)
- [Foundation-model guide](./guides/foundation-models.md)
- [Annotation benchmarks guide](./guides/annotation-benchmarks.md)

## What is included now

- AnnData-native data handling for common supervised and representation tasks
- bundled baseline models for embeddings, reconstruction, and classification
- model comparison, evaluation metrics, plots, and Markdown/CSV reports
- Scanpy handoff through `adata.obsm`
- experimental scGPT annotation adaptation with save/load support
- executed public tutorials and a [tutorial status page](./tutorials/status.md)

## Workflow snapshots

```{figure} _static/pbmc_vae_latent_umap.png
:alt: Latent UMAP from the Scanpy PBMC quickstart

Quickstart embedding colored by PBMC reference labels.
```

```{figure} _static/pbmc_downstream_leiden_umap.png
:alt: Leiden UMAP from the downstream Scanpy tutorial

Leiden clustering on the same embedding after handing control back to Scanpy.
```

## What is next

The next milestone is integration / representation transfer: comparing whether
methods reduce batch effects while preserving biological cell-type structure.
That work is tracked separately and is not part of the `0.1.7` release.

See the [roadmap](./roadmap.md) for planned annotation, integration,
perturbation, and spatial directions.

```{toctree}
:hidden:
:maxdepth: 2

install
tutorials/index
guides/scanpy-integration
guides/data
guides/models
guides/training
guides/custom-models
guides/foundation-models
guides/annotation-benchmarks
guides/evaluation
guides/visualization
guides/comparison
guides/annotation-on-your-data
api/index
contributing
roadmap
```
