# scDLKit

AnnData-native deep-learning baselines for single-cell workflows.

scDLKit is designed to sit alongside Scanpy, not replace it. The intended flow is:

1. use Scanpy to load and manage the dataset
2. use scDLKit to train or compare a model
3. write embeddings back into `adata.obsm`
4. continue with clustering, visualization, and interpretation in Scanpy

## Quickstart-first

The first workflow should be obvious:

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

Then return to Scanpy:

```python
sc.pp.neighbors(adata, use_rep="X_scdlkit_vae")
sc.tl.umap(adata)
sc.pl.umap(adata, color="louvain")
```

For reconstruction-capable models, scDLKit can also expose predicted or reconstructed expression values directly:

```python
reconstructed = runner.reconstruct(adata)
```

If your goal is labeled cell-type annotation rather than only an embedding, there is now a second quickstart path:

```python
from scdlkit.foundation import adapt_scgpt_annotation

runner = adapt_scgpt_annotation(
    adata,
    label_key="cell_type",
    output_dir="artifacts/scgpt_annotation",
)
runner.annotate_adata(adata)
runner.save("artifacts/scgpt_annotation/best_model")
```

This wrapper-first path is still experimental, but it is the shortest public route for researchers who want to compare frozen and fine-tuned scGPT strategies on a labeled dataset.

## Learning path

Follow this order if you want the tutorial set to build up like a coherent workflow rather than a pile of notebooks:

1. [Scanpy PBMC quickstart](/_tutorials/scanpy_pbmc_quickstart)
2. [Downstream Scanpy after scDLKit](/_tutorials/downstream_scanpy_after_scdlkit)
3. [PBMC model comparison](/_tutorials/pbmc_model_comparison)
4. [Reconstruction sanity check](/_tutorials/reconstruction_sanity_pbmc)
5. [PBMC classification](/_tutorials/pbmc_classification)
6. [Custom model extension](/_tutorials/custom_model_extension)
7. [Experimental scGPT PBMC embeddings](/_tutorials/scgpt_pbmc_embeddings)
8. [Experimental scGPT cell-type annotation](/_tutorials/scgpt_cell_type_annotation)
9. [Experimental scGPT dataset-specific annotation](/_tutorials/scgpt_dataset_specific_annotation)
10. [Synthetic smoke tutorial](/_tutorials/synthetic_smoke)

If you already have labels and your main question is annotation adaptation, the shorter researcher path is:

1. [Scanpy PBMC quickstart](/_tutorials/scanpy_pbmc_quickstart)
2. [Experimental scGPT cell-type annotation](/_tutorials/scgpt_cell_type_annotation)
3. [Experimental scGPT dataset-specific annotation](/_tutorials/scgpt_dataset_specific_annotation)

````{grid} 1 2 2 2
:gutter: 3

```{grid-item-card} Start Here
:link: _tutorials/scanpy_pbmc_quickstart
:link-type: doc

Train the baseline VAE, save the embedding into `adata.obsm`, and keep the rest of the workflow in Scanpy.
```

```{grid-item-card} Fine-Tune On Labels
:link: _tutorials/scgpt_dataset_specific_annotation
:link-type: doc

Use the wrapper-first experimental scGPT path to inspect a labeled dataset, compare frozen and tuned strategies, annotate `AnnData`, and save the best fitted runner.
```

```{grid-item-card} Interpret the embedding
:link: _tutorials/downstream_scanpy_after_scdlkit
:link-type: doc

Take the learned embedding through neighbors, UMAP, Leiden, marker ranking, and a careful coarse annotation pass.
```

```{grid-item-card} Validate the baselines
:link: _tutorials/pbmc_model_comparison
:link-type: doc

Compare `PCA`, AutoEncoder, VAE, and Transformer AE before deciding whether a deeper model is buying anything useful.
```

```{grid-item-card} Inspect reconstructed expression
:link: _tutorials/reconstruction_sanity_pbmc
:link-type: doc

Use a dedicated reconstruction tutorial to inspect predicted or reconstructed gene-expression outputs without overloading the main quickstart.
```

```{grid-item-card} API reference
:link: api/index
:link-type: doc

Start with the public workflow APIs first, then drop into lower-level, custom-model, and experimental surfaces as needed.
```

```{grid-item-card} Experimental scGPT annotation
:link: _tutorials/scgpt_cell_type_annotation
:link-type: doc

Compare `PCA + logistic regression`, frozen scGPT, head-only tuning, and LoRA tuning on labeled PBMC data.
```

```{grid-item-card} Easy scGPT adaptation
:link: _tutorials/scgpt_dataset_specific_annotation
:link-type: doc

Use the new wrapper-first path to inspect a labeled dataset, compare strategies, annotate `AnnData`, and save the best fitted runner.
```

```{grid-item-card} Scanpy integration map
:link: guides/scanpy-integration
:link-type: doc

See what Scanpy still owns, what scDLKit adds, and how the two tutorial ecosystems fit together.
```
````

## Example outputs

```{figure} _static/pbmc_vae_latent_umap.png
:alt: Latent UMAP from the Scanpy PBMC quickstart

Latent UMAP from the Scanpy PBMC quickstart. A healthy quickstart run should separate the major PBMC populations into broad regions rather than collapsing into a single mixed cloud.
```

```{figure} _static/pbmc_benchmark_comparison.png
:alt: PBMC comparison plot from the benchmark tutorial

Benchmark comparison chart from the PBMC model-comparison tutorial.
```

```{figure} _static/pbmc_downstream_leiden_umap.png
:alt: Leiden UMAP from the downstream Scanpy tutorial

Leiden-clustered UMAP from the downstream Scanpy workflow. This is a more realistic view of how researchers inspect cell-type structure after the scDLKit embedding step.
```

## Positioning

- Scanpy still owns raw-data preprocessing, QC, and most exploratory analysis.
- scDLKit owns the model-training, evaluation, comparison, and output-handoff layer.
- The main public scope is still gene-expression-first.
- Experimental foundation-model content remains clearly separated from the stable beginner path.

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
guides/evaluation
guides/visualization
guides/comparison
guides/annotation-on-your-data
api/index
contributing
roadmap
```
