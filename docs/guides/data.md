# Data guide

scDLKit works directly with `AnnData`.

## What scDLKit expects

- an `AnnData` object with a usable expression matrix in `adata.X` or a named layer
- optional label information in `adata.obs`
- optional batch information in `adata.obs`

## `prepare_data`

Use `prepare_data` when you want lower-level control over preprocessing and split construction:

```python
from scdlkit import prepare_data

prepared = prepare_data(
    adata,
    layer="X",
    label_key="louvain",
    batch_key="batch",
    use_hvg=True,
    n_top_genes=2000,
    normalize=True,
    log1p=True,
    batch_aware_split=True,
)
```

## Scanpy-backed preprocessing

When you request `normalize`, `log1p`, or `use_hvg`, scDLKit uses Scanpy-backed preprocessing. Install it with:

```bash
python -m pip install "scdlkit[scanpy]"
```

## Recommended practice

For the public tutorials, keep preprocessing simple and standard:

- use `scanpy.datasets.pbmc3k_processed()` for the example notebooks
- use `louvain` as the label field for representation and classification demos
- focus on model behavior rather than broad biological analysis
