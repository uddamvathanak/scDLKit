# Scanpy integration

scDLKit is designed to fit naturally into a Scanpy-centered workflow.

The most important framing is simple:

- Scanpy still owns the single-cell analysis workflow.
- scDLKit provides the model-training, evaluation, comparison, and output-handoff layer.

## Minimal integration pattern

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

For this single-cell baseline, use a light VAE KL term so PBMC populations remain visibly separable in the latent UMAP. The quickstart notebook exposes both a `quickstart` and a `full` profile; the latter simply runs longer with the same code path when you want a stronger qualitative result.

## Continue with Scanpy

Once the latent representation is in `adata.obsm`, use it like any other Scanpy embedding:

```python
sc.pp.neighbors(adata, use_rep="X_scdlkit_vae")
sc.tl.umap(adata)
sc.pl.umap(adata, color="louvain")
```

For reconstruction-capable models, you can also retrieve reconstructed expression directly:

```python
reconstructed = runner.reconstruct(adata)
```

## Workflow map

| Workflow step | Owned by | Where to learn it |
| --- | --- | --- |
| Raw QC, filtering, normalization, HVG selection | Scanpy | Official Scanpy preprocessing and clustering tutorials |
| Train a baseline model on processed PBMC | scDLKit | [Scanpy PBMC quickstart](/_tutorials/scanpy_pbmc_quickstart) |
| Push latent embeddings into `adata.obsm` | scDLKit + Scanpy handoff | [Scanpy PBMC quickstart](/_tutorials/scanpy_pbmc_quickstart) |
| Cluster and interpret the scDLKit embedding | Scanpy on top of scDLKit output | [Downstream Scanpy after scDLKit](/_tutorials/downstream_scanpy_after_scdlkit) |
| Inspect reconstructed expression | scDLKit | [Reconstruction sanity check](/_tutorials/reconstruction_sanity_pbmc) |
| Compare baseline models | scDLKit | [PBMC model comparison](/_tutorials/pbmc_model_comparison) |
| Wrap a custom PyTorch module | scDLKit | [Custom model extension](/_tutorials/custom_model_extension) |
| Try the experimental frozen foundation path | scDLKit | [Experimental scGPT PBMC embeddings](/_tutorials/scgpt_pbmc_embeddings) |

## Positioning

scDLKit is not a replacement for Scanpy.

It is the model-training and evaluation layer you can drop into a standard single-cell analysis workflow when you want:

- a baseline autoencoder or VAE
- a quick benchmark before building a custom method
- a consistent way to compare latent representations
- a tutorial-backed way to inspect reconstructed outputs when the model supports them

Today that scope is still gene-expression-first. Spatial and multimodal workflows are intentionally deferred until the current baseline toolkit is better benchmarked.
