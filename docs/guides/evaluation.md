# Evaluation

Evaluation is built into the workflow rather than left to ad hoc notebook code.

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
