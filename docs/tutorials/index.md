# Tutorials

scDLKit now presents its public tutorial surface around four research tasks.

The goal is to make the paper identity obvious without pretending that every
task is already implemented at equal maturity.

## Main research tasks

````{grid} 1 2 2 2
:gutter: 2

```{grid-item-card} Cell type annotation
:link: /_tutorials/scgpt_human_pancreas_annotation
:link-type: doc

Status: `Pilot`

Question answered:
Can scDLKit already support a credible low-code adaptation story on labeled
human data?

Model and task scope:
Current pilot uses the experimental scGPT adaptation path on labeled human
single-cell RNA data.

Figure role:
Primary supervised adaptation and PEFT comparison figure family.
```

```{grid-item-card} Integration / representation transfer
:link: /roadmap#integration-pillar
:link-type: url

Status: `Planned`

Question answered:
Can adapted representations preserve biological structure while reducing
technical variation across studies or batches?

Model and task scope:
Planned as a paper task with a formal integration metric pipeline.

Figure role:
Cross-study transfer and batch-mixing figure family.
```

```{grid-item-card} Perturbation-response prediction
:link: /roadmap#perturbation-pillar
:link-type: url

Status: `Planned`

Question answered:
Can the adapted model recover perturbation-response structure under a unified
benchmark interface?

Model and task scope:
Planned as a dedicated perturbation benchmark pillar.

Figure role:
Response-prediction and low-label efficiency figure family.
```

```{grid-item-card} Spatial domain / niche classification
:link: /roadmap#spatial-pillar
:link-type: url

Status: `Planned`

Question answered:
Can scDLKit support spatial task adaptation as a real paper pillar rather than
an appendix claim?

Model and task scope:
Planned around the Nicheformer-facing spatial benchmark story.

Figure role:
Spatial qualitative and task-performance figure family.
```
````

## Supporting workflows

These are still valuable and still public, but they are not the main paper-task
surface.

````{grid} 1 2 2 2
:gutter: 2

```{grid-item-card} Scanpy PBMC quickstart
:link: /_tutorials/scanpy_pbmc_quickstart
:link-type: doc

Audience:
Users starting the stable baseline workflow.

Outputs:
Latent embedding, report, loss curve, latent UMAP.
```

```{grid-item-card} Downstream Scanpy after scDLKit
:link: /_tutorials/downstream_scanpy_after_scdlkit
:link-type: doc

Audience:
Users who want the Scanpy interpretation layer after the model step.

Outputs:
Latent UMAP, Leiden UMAP, markers, downstream report.
```

```{grid-item-card} PBMC model comparison
:link: /_tutorials/pbmc_model_comparison
:link-type: doc

Audience:
Users validating whether deeper baselines help beyond `PCA`.

Outputs:
Benchmark CSV, comparison figure, baseline reference UMAPs.
```
````

## Advanced / appendix workflows

These remain documented and tested, but they are no longer equal to the four
main research task tracks.

### Advanced appendix

- [Reconstruction sanity check](/_tutorials/reconstruction_sanity_pbmc)
- [Custom model extension](/_tutorials/custom_model_extension)
- [Experimental scGPT PBMC embeddings](/_tutorials/scgpt_pbmc_embeddings)

### Experimental detail appendix

- [Experimental scGPT cell-type annotation](/_tutorials/scgpt_cell_type_annotation)
- [Experimental scGPT dataset-specific annotation](/_tutorials/scgpt_dataset_specific_annotation)
- [Experimental scGPT human-pancreas annotation](/_tutorials/scgpt_human_pancreas_annotation)

### Maintainer and smoke path

- [Synthetic smoke tutorial](/_tutorials/synthetic_smoke)

## Reading order

If you want the current stable baseline path first:

1. [Scanpy PBMC quickstart](/_tutorials/scanpy_pbmc_quickstart)
2. [Downstream Scanpy after scDLKit](/_tutorials/downstream_scanpy_after_scdlkit)
3. [PBMC model comparison](/_tutorials/pbmc_model_comparison)

If you want the current research-facing annotation path first:

1. [Experimental scGPT human-pancreas annotation](/_tutorials/scgpt_human_pancreas_annotation)
2. [Experimental scGPT dataset-specific annotation](/_tutorials/scgpt_dataset_specific_annotation)
3. [Experimental scGPT cell-type annotation](/_tutorials/scgpt_cell_type_annotation)

```{toctree}
:hidden:
:maxdepth: 1

/_tutorials/scanpy_pbmc_quickstart
/_tutorials/downstream_scanpy_after_scdlkit
/_tutorials/pbmc_model_comparison
/_tutorials/pbmc_classification
/_tutorials/reconstruction_sanity_pbmc
/_tutorials/custom_model_extension
/_tutorials/scgpt_pbmc_embeddings
/_tutorials/scgpt_cell_type_annotation
/_tutorials/scgpt_dataset_specific_annotation
/_tutorials/scgpt_human_pancreas_annotation
/_tutorials/synthetic_smoke
```
