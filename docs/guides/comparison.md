# Comparison

Use `compare_models` when you want a quick side-by-side baseline benchmark with consistent training and reporting settings.

## Example

```python
from scdlkit import compare_models

result = compare_models(
    adata,
    models=["autoencoder", "vae", "transformer_ae"],
    task="representation",
    shared_kwargs={
        "epochs": 5,
        "batch_size": 128,
        "label_key": "louvain",
        "device": "auto",
    },
    output_dir="artifacts/pbmc_compare",
)
```

## Outputs

When `output_dir` is set, comparison can write:

- `benchmark_metrics.csv`
- `benchmark_report.md`
- `benchmark_comparison.png`

This makes it easy to include baseline evidence in research notes or portfolio material.
