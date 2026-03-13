# Visualization

scDLKit includes quick baseline visualizations so researchers do not need to rebuild plotting code for every sanity check.

## Available plots

- loss curves
- latent UMAP
- latent PCA
- reconstruction scatter
- classification confusion matrix
- model-comparison bar chart

## Typical representation workflow

```python
runner.plot_losses()
runner.plot_latent(method="umap", color="label")
```

## Scanpy after scDLKit

For the main PBMC notebook, the recommended path is:

1. train and encode with scDLKit
2. store the embedding in `adata.obsm`
3. run `sc.pp.neighbors` and `sc.tl.umap`
4. visualize with Scanpy plotting
