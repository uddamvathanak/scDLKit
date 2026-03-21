# Data preparation

## What it is

Status: stable.

This page documents the lower-level preprocessing helpers that turn `AnnData`
into split objects for `Trainer` and related workflows:

- `prepare_data(...)`
- `transform_adata(...)`

## When to use it

Use these helpers when:

- you are building on `Trainer` directly
- you need explicit control over labels, batches, or split fractions
- you want to transform a second `AnnData` with the same fitted preprocessing metadata

## Minimal example

```python
from scdlkit import prepare_data
from scdlkit.data import transform_adata

prepared = prepare_data(adata, label_key="louvain", normalize=True, log1p=True)
held_out = transform_adata(
    other_adata,
    prepared.preprocessing,
    label_encoder=prepared.label_encoder,
    batch_encoder=prepared.batch_encoder,
)
```

## Parameters

- `prepare_data(...)` controls matrix selection and preprocessing through `layer`, `use_hvg`, `normalize`, `log1p`, and `scale`.
- `label_key` and `batch_key` define optional encoded supervision and batch metadata from `adata.obs`.
- `val_size`, `test_size`, `batch_aware_split`, and `random_state` define split behavior.
- `transform_adata(...)` expects the stored `preprocessing` metadata plus optional fitted label and batch encoders.

## Input expectations

- input must be an `anndata.AnnData` object with features in `var_names`.
- if `label_key` or `batch_key` is provided, the column must exist in `adata.obs`.
- the transformed dataset must contain the same feature names as the fitted preprocessing metadata.
- Scanpy-backed operations require the `scanpy` extra.

## Returns / outputs

- `prepare_data(...)` returns `PreparedData` with train, validation, and test `SplitData` plus preprocessing metadata.
- `transform_adata(...)` returns a transformed `SplitData` that can be passed to `Trainer.predict_dataset(...)`.

## Failure modes / raises

- `ValueError` if `label_key`, `batch_key`, or the selected layer is missing.
- `ValueError` if transformed data is missing required features or contains unseen labels.
- `ImportError` if Scanpy-backed preprocessing is requested without `scdlkit[scanpy]`.

## Notes / caveats

- These helpers are the stable lower-level path behind `TaskRunner`.
- They are not the scGPT tokenization entrypoint; use [Experimental foundation helpers](./foundation.md) for that path.
- `transform_adata(...)` applies the previously stored feature order and optional scaler before inference.

## Related tutorial(s)

- [Scanpy PBMC quickstart](/_tutorials/scanpy_pbmc_quickstart)
- [Custom model extension](/_tutorials/custom_model_extension)
- [Downstream Scanpy after scDLKit](/_tutorials/downstream_scanpy_after_scdlkit)

```{eval-rst}
.. autofunction:: scdlkit.prepare_data
```

```{eval-rst}
.. autofunction:: scdlkit.data.prepare.transform_adata
```
