# Evaluation and outputs

## What it is

Status: stable.

This page documents the stable evaluation and export helpers that make scDLKit
results comparable and reportable:

- `compare_models(...)`
- `evaluate_predictions(...)`
- `save_markdown_report(...)`
- `save_metrics_table(...)`

## When to use it

Use these helpers when:

- you want consistent metrics across tasks
- you need a quick benchmark table for several bundled models
- you want Markdown and CSV outputs that match the tutorial artifacts

## Minimal example

```python
from scdlkit import compare_models
from scdlkit.evaluation import evaluate_predictions, save_markdown_report

benchmark = compare_models(
    adata,
    models=["autoencoder", "vae", "transformer_ae"],
    task="representation",
    shared_kwargs={"label_key": "louvain", "epochs": 10},
    output_dir="artifacts/pbmc_compare",
)

metrics = evaluate_predictions("classification", {"y": labels, "logits": logits})
save_markdown_report(metrics, path="artifacts/report.md", title="Classification report")
```

## Parameters

- `compare_models(...)` expects a shared `AnnData`, a list of bundled model names, a task, and optional shared runner kwargs.
- `evaluate_predictions(...)` expects a task name plus a prediction dictionary with task-specific keys.
- `save_markdown_report(...)` and `save_metrics_table(...)` expect metric dictionaries and output paths.

## Input expectations

- classification evaluation requires encoded labels under `y` and logits under `logits`.
- reconstruction evaluation requires `x` and `reconstruction`.
- representation evaluation requires `latent` and benefits from `y` or `batch` when present.
- report helpers serialize scalar metrics directly and include structured values as-is in Markdown.

## Returns / outputs

- `compare_models(...)` returns a `BenchmarkResult` with a metrics frame, fitted runners, and optional artifact paths.
- `evaluate_predictions(...)` returns a task-specific metric dictionary.
- `save_markdown_report(...)` writes a Markdown report.
- `save_metrics_table(...)` writes a CSV with scalar metrics.

## Failure modes / raises

- `ValueError` if the prediction payload does not satisfy the selected task contract.
- downstream file-writing errors propagate from the filesystem if the destination path is invalid.

## Notes / caveats

- `compare_models(...)` is the stable bundled-model comparison path, not the scGPT adaptation benchmark.
- prediction payloads usually come from `Trainer.predict_dataset(...)` or a compatible wrapper.
- the tutorial artifacts in `artifacts/` are built on these helpers.

## Related tutorial(s)

- [PBMC model comparison](/_tutorials/pbmc_model_comparison)
- [PBMC classification](/_tutorials/pbmc_classification)
- [Experimental scGPT dataset-specific annotation](/_tutorials/scgpt_dataset_specific_annotation)

```{eval-rst}
.. autofunction:: scdlkit.compare_models
```

```{eval-rst}
.. autofunction:: scdlkit.evaluation.evaluate_predictions
```

```{eval-rst}
.. autofunction:: scdlkit.evaluation.save_markdown_report
```

```{eval-rst}
.. autofunction:: scdlkit.evaluation.save_metrics_table
```
