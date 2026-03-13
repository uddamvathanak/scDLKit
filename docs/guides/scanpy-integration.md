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
    epochs=10,
    batch_size=128,
)

runner.fit(adata)
adata.obsm["X_scdlkit_vae"] = runner.encode(adata)
```

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
