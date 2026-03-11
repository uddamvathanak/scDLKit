# Data Preparation

`prepare_data()` accepts `AnnData` and returns a `PreparedData` object containing train, validation, and test splits plus preprocessing metadata.

Supported features:

- `layer` selection
- optional HVG selection
- optional normalization, log transform, and scaling
- label extraction via `label_key`
- batch extraction via `batch_key`
- batch-aware splitting
- sparse-safe access through `AnnDataset`

If you request Scanpy-backed preprocessing without Scanpy installed, scDLKit raises an informative install error for `scdlkit[scanpy]`.
