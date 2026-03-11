# Comparison

Use `compare_models()` to benchmark multiple baselines against the same task configuration.

```python
from scdlkit import compare_models

benchmark = compare_models(
    adata,
    models=["autoencoder", "vae", "transformer_ae"],
    task="representation",
    shared_kwargs={"epochs": 10, "label_key": "cell_type"},
    output_dir="artifacts/compare",
)
```

This writes:

- `benchmark_metrics.csv`
- `benchmark_report.md`
- `benchmark_comparison.png`
