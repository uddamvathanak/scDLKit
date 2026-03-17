# Evaluation

Evaluation is built into the workflow rather than left to ad hoc notebook code.

The release process also uses an internal quality suite so the toolkit is evaluated
against itself on small Scanpy built-ins before public tutorial defaults are changed.
The primary release-gate datasets are `pbmc3k_processed` and `paul15`, with `PCA`
kept as the classical reference baseline in comparison work. The experimental
foundation-model pilot adds `pbmc68k_reduced` and compares frozen scGPT embeddings
against `PCA` rather than treating the foundation path as automatically better.

## Core metrics

Representation and reconstruction workflows can report:

- `mse`
- `mae`
- `pearson`
- `spearman`
- `silhouette`
- `knn_label_consistency`
- `ari`
- `nmi`
- `runtime_sec` in comparison and benchmark summaries
- `probe_accuracy` for frozen linear probes on embedding benchmarks
- `probe_macro_f1` for frozen linear probes on embedding benchmarks

Classification workflows can report:

- `accuracy`
- `macro_f1`
- `confusion_matrix`

## Example

```python
metrics = runner.evaluate()
metrics
```

## Reports

You can export a Markdown report and scalar metrics table:

```python
runner.save_report("artifacts/report.md")
```

For reconstruction-capable models, evaluation often goes together with direct
inspection of reconstructed outputs:

```python
reconstructed = runner.reconstruct(adata)
```

That output is now covered in the dedicated reconstruction sanity-check tutorial
rather than being overloaded into the main embedding quickstart.

For benchmark work, treat `PCA` as the classical reference baseline rather than comparing deep-learning models only against each other.
