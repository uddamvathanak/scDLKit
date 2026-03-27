# Codebase Context

## Top-level structure

```
scDLKit/
  src/scdlkit/         # Main library source
  scripts/             # Benchmark and quality suite runners
  tests/               # Unit and integration tests
  examples/            # Jupyter notebook tutorials
  planning/            # Milestone checklists, research docs
  artifacts/           # Benchmark output artifacts (gitignored)
  docs/                # Documentation site source
  data/                # Cached datasets
  logo/                # Project branding assets
```

## Source: `src/scdlkit/`

### Core training

- `training/trainer.py` - Plain PyTorch training loop with early stopping, checkpointing, LR scheduling (StepLR), and AMP support. Central to all model adaptation.
- `training/callbacks.py` - Early stopping state dataclass.
- `runner.py` - High-level `TaskRunner` for quick model comparison.

### Tasks

- `tasks/base.py` - `BaseTask` interface (compute_loss, evaluate).
- `tasks/classification.py` - Cross-entropy classification task.
- `tasks/reconstruction.py` - MSE reconstruction task.
- `tasks/representation.py` - Representation extraction (no training loss).

### Models

- `models/autoencoder.py` - Vanilla autoencoder.
- `models/vae.py` - Variational autoencoder.
- `models/classifier.py` - MLP classifier.
- `models/transformer.py` - Transformer autoencoder.
- `models/blocks.py` - Shared building blocks (encoder/decoder).
- `models/base.py` - Base model mixin.
- `models/denoising.py` - Denoising autoencoder.
- `models/registry.py` - Model name registry.

### Foundation model support

- `foundation/scgpt.py` - scGPT embedding model wrapper.
- `foundation/annotation.py` - scGPT annotation model with classifier head.
- `foundation/runner.py` - `ScGPTAnnotationRunner` - high-level experimental wrapper for annotation adaptation. Supports frozen_probe, head, full_finetune, lora, adapter, prefix_tuning, ia3 strategies.
- `foundation/data.py` - scGPT data preparation, tokenization, splitting.
- `foundation/cache.py` - Checkpoint download and caching.
- `foundation/peft.py` - PEFT strategy resolution and serialization.
- `foundation/lora.py` - LoRA implementation for scGPT.
- `foundation/adapters.py` - Bottleneck adapter layers.
- `foundation/prefix_tuning.py` - Prefix tuning for attention.
- `foundation/ia3.py` - IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations).
- `foundation/base.py` - Base foundation model interface.

### Evaluation

- `evaluation/metrics.py` - Metric helpers: classification (accuracy, macro/weighted F1, balanced accuracy, precision, recall, Cohen's kappa, MCC, AUROC, per-class F1/precision/recall, confusion matrix), reconstruction (MSE, MAE, Pearson, Spearman), representation (silhouette, ARI, NMI, KNN label consistency).
- `evaluation/evaluator.py` - `evaluate_predictions` dispatcher.
- `evaluation/compare.py` - Multi-model comparison utilities.
- `evaluation/report.py` - Markdown and CSV report generation.

### Data

- `data/datasets.py` - `AnnDataset` wrapping AnnData for PyTorch.
- `data/prepare.py` - Data preparation pipeline.
- `data/schemas.py` - `SplitData` schema for train/val/test.
- `data/splits.py` - Stratified splitting utilities.
- `_datasets/openproblems.py` - OpenProblems dataset loader.

### Visualization

- `visualization/classification.py` - Confusion matrix heatmap.
- `visualization/latent.py` - UMAP/embedding plots.
- `visualization/reconstruction.py` - Reconstruction scatter plots.
- `visualization/compare.py` - Multi-model comparison plots.
- `visualization/training.py` - Training history plots.

### Benchmarks

- `benchmarks/annotation_spec.py` - Locked annotation benchmark specification: datasets (PBMC68k, Human Pancreas), regimes (full_label, low_label, cross_study) with 5-fold stratified CV, task spec with primary/secondary/efficiency metrics.

### Adapters

- `adapters/base.py` - Base adapter interface.
- `adapters/classification.py` - Classification adapter.
- `adapters/reconstruction.py` - Reconstruction adapter.

## Scripts

- `scripts/run_annotation_benchmark.py` - Full annotation benchmark: 8 strategies x 2 datasets x 3 regimes x 5-fold CV = 344 runs. Uses stratified k-fold cross-validation (default 5 folds). Generates publication-quality figures (performance bars with error bars, label efficiency curves with confidence bands, cross-study heatmap, Pareto scatter, radar chart, per-class F1 heatmap). Supports `--aggregate-only` (rebuild from row.json) and `--figures-only` (regenerate from all_results.csv).
- `scripts/run_quality_suite.py` - Internal quality gates for CI and release checks. Defines training profiles (CI: 1 epoch, full: 10-15 epochs with patience=5).
- `scripts/run_foundation_smoke.py` - Foundation model smoke tests.

## Planning

- `planning/checklists/01-annotation.md` - Milestone 1 checklist (annotation pillar).
- `planning/research/annotation_benchmark_design.md` - Research rationale for benchmark design based on published scGPT/PEFT papers.
- `planning/research/annotation_benchmark_methods.md` - Plain-language methods documentation for the annotation benchmark (method-by-method explanation, randomness and reproducibility, manuscript language template).
- `planning/research/bioinformatics_original_paper_target.md` - Target journal analysis: Bioinformatics Original Paper (7 pages, ~5000 words), requirements, style study, manuscript structure, submission checklist.

## Examples

- `examples/scgpt_human_pancreas_annotation.ipynb` - Main annotation tutorial (human pancreas).
- `examples/scgpt_cell_type_annotation.ipynb` - scGPT annotation tutorial.
- `examples/scgpt_pbmc_embeddings.ipynb` - scGPT PBMC embedding tutorial.
