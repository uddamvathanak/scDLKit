# scDLKit

AnnData-native deep-learning baselines for single-cell workflows.

scDLKit is designed to sit alongside Scanpy, not replace it. Use Scanpy for loading and exploratory single-cell analysis, then use scDLKit to train, compare, and evaluate baseline deep-learning models with a small, reproducible API.

````{grid} 1 2 2 2
:gutter: 3

```{grid-item-card} Install
:link: install
:link-type: doc

Set up the CPU or GPU tutorial path from PyPI, including the new `tutorials` extra.
```

```{grid-item-card} Scanpy PBMC quickstart
:link: _tutorials/scanpy_pbmc_quickstart
:link-type: doc

Start with the primary notebook tutorial built on `scanpy.datasets.pbmc3k_processed()`.
```

```{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

Open the notebook walkthroughs for representation learning, model comparison, and classification.
```

```{grid-item-card} Scanpy integration
:link: guides/scanpy-integration
:link-type: doc

See how to store scDLKit latent embeddings in `adata.obsm` and continue with standard Scanpy analysis.
```
````

## Why scDLKit

- AnnData-native model training and evaluation
- baseline-first deep-learning workflows for single-cell data
- one shared CPU/GPU path with `device="auto"`
- reproducible reports, plots, and tutorial notebooks
- clean separation between model workflows and Scanpy analysis

## Example outputs

```{figure} _static/first_run_loss_curve.png
:alt: Training loss curve from the synthetic smoke example

Loss curve from the lightweight synthetic smoke example.
```

```{figure} _static/first_run_latent_pca.png
:alt: Latent PCA from the synthetic smoke example

Latent embedding produced by the first end-to-end scDLKit walkthrough.
```

```{figure} _static/pbmc_benchmark_comparison.png
:alt: PBMC comparison plot from the benchmark tutorial

Benchmark comparison chart from the PBMC model-comparison tutorial.
```

## Recommended learning path

1. Install the tutorial dependencies from PyPI.
2. Run the Scanpy PBMC quickstart notebook.
3. Continue with the model comparison notebook.
4. Use the classification notebook once you want a supervised baseline.
5. Keep the synthetic notebook only as a minimal smoke or fallback path.

```{toctree}
:hidden:
:maxdepth: 2

install
tutorials/index
guides/scanpy-integration
guides/data
guides/models
guides/training
guides/evaluation
guides/visualization
guides/comparison
api/index
contributing
roadmap
```
