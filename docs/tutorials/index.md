# Tutorials

The tutorial path is intentionally Scanpy-first and model-focused.

The goal is not to replace Scanpy’s preprocessing tutorials. The goal is to show where scDLKit enters the workflow, what outputs it creates, and how those outputs move back into normal Scanpy analysis.

## Core path

````{grid} 1 2 2 2
:gutter: 2

```{grid-item-card} Scanpy PBMC quickstart
:link: /_tutorials/scanpy_pbmc_quickstart
:link-type: doc

Audience:
Researchers and analysts starting the main baseline workflow.

Question answered:
How do I train a first model and get an embedding back into `adata.obsm`?

Outputs:
Latent embedding, report, loss curve, latent UMAP.
```

```{grid-item-card} Downstream Scanpy after scDLKit
:link: /_tutorials/downstream_scanpy_after_scdlkit
:link-type: doc

Audience:
Readers who want the missing interpretation layer after the embedding step.

Question answered:
What should I do in Scanpy after scDLKit has produced an embedding?

Outputs:
Latent UMAP, Leiden UMAP, marker dotplot, ranked markers, downstream report.
```
````

## Validation path

````{grid} 1 2 2 2
:gutter: 2

```{grid-item-card} PBMC model comparison
:link: /_tutorials/pbmc_model_comparison
:link-type: doc

Audience:
Users deciding whether a deeper baseline is worth it.

Question answered:
Does `PCA` already solve enough of the problem, or does a deeper model buy useful structure?

Outputs:
Benchmark CSV, comparison figure, PCA reference UMAP, best baseline UMAP.
```

```{grid-item-card} Reconstruction sanity check
:link: /_tutorials/reconstruction_sanity_pbmc
:link-type: doc

Audience:
Users who need to inspect predicted or reconstructed gene-expression values.

Question answered:
How do I retrieve reconstructed expression, and what is a reasonable sanity check for it?

Outputs:
Report, loss curve, reconstruction scatter, gene-panel reconstruction summary.
```
````

## Extension path

````{grid} 1 2 2 2
:gutter: 2

```{grid-item-card} PBMC classification
:link: /_tutorials/pbmc_classification
:link-type: doc

Audience:
Users who want a simple supervised baseline.

Question answered:
What does a lightweight classifier baseline look like in the same toolkit?

Outputs:
Report, loss curve, confusion matrix.
```

```{grid-item-card} Custom model extension
:link: /_tutorials/custom_model_extension
:link-type: doc

Audience:
Users bringing their own PyTorch module.

Question answered:
How do I validate a wrapped custom model inside scDLKit before building on top of it?

Outputs:
Report, loss curve, latent UMAP.
```

```{grid-item-card} Experimental scGPT PBMC embeddings
:link: /_tutorials/scgpt_pbmc_embeddings
:link-type: doc

Audience:
Users who want an experimental foundation-model reference.

Question answered:
How do I extract frozen scGPT embeddings and return them to Scanpy?

Outputs:
Report, latent UMAP, frozen linear-probe confusion matrix, embedding summary.
```

```{grid-item-card} Experimental scGPT cell-type annotation
:link: /_tutorials/scgpt_cell_type_annotation
:link-type: doc

Audience:
Users who want to compare frozen and fine-tuned scGPT strategies on a labeled dataset.

Question answered:
Do I need only frozen scGPT embeddings, a trainable classification head, or LoRA tuning?

Outputs:
Report, strategy metrics table, frozen UMAP, LoRA UMAP, best-strategy confusion matrix.
```

```{grid-item-card} Experimental scGPT dataset-specific annotation
:link: /_tutorials/scgpt_dataset_specific_annotation
:link-type: doc

Audience:
Users who want the easiest wrapper-first path for adapting scGPT to a labeled dataset with minimal code.

Question answered:
How do I inspect my dataset, compare strategies, annotate `AnnData`, and save the best fitted runner in one workflow?

Outputs:
Report, strategy metrics table, frozen UMAP, best-strategy UMAP, saved runner manifest, saved runner weights.
```
````

## Fallback path

````{grid} 1 1 1 1
:gutter: 2

```{grid-item-card} Synthetic smoke tutorial
:link: /_tutorials/synthetic_smoke
:link-type: doc

Audience:
Users who want the smallest possible end-to-end example.

Question answered:
Can I smoke-test the basic workflow with minimal setup?

Outputs:
Report, loss curve, latent PCA.
```
````

## Recommended order

1. Scanpy PBMC quickstart
2. Downstream Scanpy after scDLKit
3. PBMC model comparison
4. Reconstruction sanity check
5. PBMC classification
6. Custom model extension
7. Experimental scGPT PBMC embeddings
8. Experimental scGPT cell-type annotation
9. Experimental scGPT dataset-specific annotation
10. Synthetic smoke tutorial

```{toctree}
:hidden:
:maxdepth: 1

/_tutorials/scanpy_pbmc_quickstart
/_tutorials/downstream_scanpy_after_scdlkit
/_tutorials/pbmc_model_comparison
/_tutorials/reconstruction_sanity_pbmc
/_tutorials/pbmc_classification
/_tutorials/custom_model_extension
/_tutorials/scgpt_pbmc_embeddings
/_tutorials/scgpt_cell_type_annotation
/_tutorials/scgpt_dataset_specific_annotation
/_tutorials/synthetic_smoke
```
