# scDLKit

scDLKit is moving from a baseline toolkit identity toward a publication-first
research program with a software artifact attached to it.

## Available now

Today the public repo supports two main entrypoints:

- stable baseline workflows through `TaskRunner`
- experimental labeled annotation adaptation through `adapt_annotation(...)`

The current implemented scope is still narrower than the paper target:

- stable deep-learning baselines for single-cell workflows
- Scanpy handoff through `adata.obsm`
- experimental scGPT annotation adaptation on labeled human scRNA-seq
- beyond-PBMC annotation evidence on human pancreas

## Paper target

The paper target is:

**scDLKit is a minimal-code, AnnData-native framework for parameter-efficient adaptation and reproducible benchmarking of single-cell and spatial foundation models.**

That target expands the repo in two directions:

- model breadth:
  - `scGPT`
  - `scFoundation`
  - `CellFM`
  - `Nicheformer`
- task breadth:
  - annotation
  - integration
  - perturbation
  - spatial

Use the [roadmap](./roadmap.md) when you want the full distinction between
paper target and current implementation truth.

## Main research task map

````{grid} 1 2 2 2
:gutter: 3

```{grid-item-card} Cell type annotation
:link: _tutorials/scgpt_human_pancreas_annotation
:link-type: doc

Status: `Pilot`

Main question:
Can scDLKit already support a credible low-code adaptation story on labeled
human data?
```

```{grid-item-card} Integration / representation transfer
:link: /roadmap#integration-pillar
:link-type: url

Status: `Planned`

Main question:
Can adapted representations transfer across studies and batches under a
standardized benchmark?
```

```{grid-item-card} Perturbation-response prediction
:link: /roadmap#perturbation-pillar
:link-type: url

Status: `Planned`

Main question:
Can the framework benchmark adaptation strategies on perturbation-response
tasks?
```

```{grid-item-card} Spatial domain / niche classification
:link: /roadmap#spatial-pillar
:link-type: url

Status: `Planned`

Main question:
Can scDLKit support a real spatial pillar anchored by Nicheformer rather than a
future placeholder?
```
````

## Current entrypoints

### Stable baseline path

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

Use this when you want the stable baseline workflow.

Related docs:

- [TaskRunner API](./api/taskrunner.md)
- [Scanpy PBMC quickstart](/_tutorials/scanpy_pbmc_quickstart)
- [Downstream Scanpy after scDLKit](/_tutorials/downstream_scanpy_after_scdlkit)

### Experimental annotation path

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

Use this when you want the current low-code research-facing adaptation path.

Related docs:

- [Experimental annotation quickstart API](./api/annotation.md)
- [Experimental scGPT human-pancreas annotation](/_tutorials/scgpt_human_pancreas_annotation)
- [Experimental scGPT dataset-specific annotation](/_tutorials/scgpt_dataset_specific_annotation)
- [Roadmap](./roadmap.md)

## Supporting workflows

- [Tutorial map](./tutorials/index.md)
- [Scanpy integration guide](./guides/scanpy-integration.md)
- [Annotation benchmarks guide](./guides/annotation-benchmarks.md)
- [Foundation-model guide](./guides/foundation-models.md)

## Workflow snapshots

```{figure} _static/pbmc_vae_latent_umap.png
:alt: Latent UMAP from the Scanpy PBMC quickstart

Quickstart embedding colored by the PBMC reference labels.
```

```{figure} _static/pbmc_downstream_leiden_umap.png
:alt: Leiden UMAP from the downstream Scanpy tutorial

Leiden clustering on the same embedding after handing control back to Scanpy.
```

## Current scope

- Scanpy still owns raw-data preprocessing, QC, and most exploratory analysis.
- scDLKit currently owns model training, evaluation, comparison, and output
  handoff.
- the current public implementation is still gene-expression-first
- the paper target is broader than the current implementation and must remain
  labeled as such

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
