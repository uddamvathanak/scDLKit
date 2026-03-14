# Scanpy integration

scDLKit is designed to fit naturally into a Scanpy-centered workflow.

Use Scanpy for:

- loading and storing single-cell data in `AnnData`
- neighborhood graph construction
- UMAP and related visualization
- downstream exploratory analysis

Use scDLKit for:

- training baseline deep-learning models
- evaluating reconstruction, representation, or classification metrics
- comparing multiple baselines quickly

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

For this single-cell baseline, use a light VAE KL term so PBMC populations remain
visibly separable in the latent UMAP. The quickstart notebook exposes both a
`quickstart` and a `full` profile; the latter simply runs longer with the same
code path when you want a stronger qualitative result.

## Continue with Scanpy

Once the latent representation is in `adata.obsm`, use it like any other Scanpy embedding:

```python
sc.pp.neighbors(adata, use_rep="X_scdlkit_vae")
sc.tl.umap(adata)
sc.pl.umap(adata, color="louvain")
```

## Positioning

scDLKit is not a replacement for Scanpy.

It is the model-training and evaluation layer you can drop into a standard single-cell analysis workflow when you want:

- a baseline autoencoder or VAE
- a quick benchmark before building a custom method
- a consistent way to compare latent representations

Today that scope is still gene-expression-first. Spatial and multimodal workflows are intentionally deferred until the current baseline toolkit is better benchmarked.
