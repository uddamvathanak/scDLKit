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
        "epochs": 10,
        "batch_size": 128,
        "label_key": "louvain",
        "device": "auto",
    },
    output_dir="artifacts/pbmc_compare",
)
```

The public PBMC comparison notebook goes one step further and adds `PCA` as an explicit classical reference baseline and uses a compact Transformer AE configuration so the comparison stays practical on CPU. That is the right comparison structure for a baseline-first toolkit: first ask whether deep learning beats a reasonable classical method, then compare model families within scDLKit.

## Outputs

When `output_dir` is set, comparison can write:

- `benchmark_metrics.csv`
- `benchmark_report.md`
- `benchmark_comparison.png`

`benchmark_metrics.csv` now also records `runtime_sec`, which makes it easier to compare qualitative gains against training cost.

For CPU-oriented notebook runs, the comparison tutorial uses a compact Transformer AE setting with `patch_size=48`, `d_model=64`, `n_heads=2`, and `n_layers=1`.

This makes it easy to include baseline evidence in research notes or portfolio material.

The same comparison structure also feeds the internal release hardening work:
small built-in Scanpy datasets, a `PCA` reference, and explicit runtime tracking
before tutorial defaults are changed.
