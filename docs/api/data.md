# Data preparation

Use the data helpers when you want lower-level control over preprocessing and split construction.

Typical use cases:

- custom `Trainer` workflows
- adapter-based custom models
- explicit control over labels, batches, and split behavior

```{eval-rst}
.. autofunction:: scdlkit.prepare_data
```

```{eval-rst}
.. autofunction:: scdlkit.data.prepare.transform_adata
```
