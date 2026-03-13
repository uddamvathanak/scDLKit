# Tutorials

The tutorial path is intentionally Scanpy-first and model-focused.

Start with Scanpy for the dataset object and downstream neighborhood analysis, then use scDLKit for training, evaluation, and model comparison.

````{grid} 1 2 2 2
:gutter: 2

```{grid-item-card} Scanpy PBMC quickstart
:link: /_tutorials/scanpy_pbmc_quickstart
:link-type: doc

Learn the main scDLKit workflow on `pbmc3k_processed`, then store the latent representation in `adata.obsm` for Scanpy analysis.
```

```{grid-item-card} PBMC model comparison
:link: /_tutorials/pbmc_model_comparison
:link-type: doc

Compare AutoEncoder, VAE, and Transformer AE baselines on the same PBMC workflow.
```

```{grid-item-card} PBMC classification
:link: /_tutorials/pbmc_classification
:link-type: doc

Run the classification baseline and inspect accuracy, macro F1, and a confusion matrix.
```

```{grid-item-card} Synthetic smoke tutorial
:link: /_tutorials/synthetic_smoke
:link-type: doc

Use the minimal synthetic notebook only when you want the smallest dependency path or a smoke run.
```
````

## Learning order

1. Scanpy PBMC quickstart
2. PBMC model comparison
3. PBMC classification
4. Synthetic smoke tutorial

```{toctree}
:hidden:
:maxdepth: 1

/_tutorials/scanpy_pbmc_quickstart
/_tutorials/pbmc_model_comparison
/_tutorials/pbmc_classification
/_tutorials/synthetic_smoke
```
