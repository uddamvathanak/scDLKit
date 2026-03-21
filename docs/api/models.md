# Built-in models

## What it is

Status: stable.

This page documents the bundled scDLKit model inventory and the registry used by
`TaskRunner` and `create_model(...)`.

## When to use it

Use this page when:

- you want to know which built-in models are available
- you need constructor parameters for a bundled baseline
- you want to check which task a model is intended to support

Most users should still start with [TaskRunner](./taskrunner.md) or the notebook tutorials rather than constructing models manually.

## Minimal example

```python
from scdlkit import create_model

model = create_model(
    "vae",
    input_dim=2000,
    latent_dim=32,
    hidden_dims=(512, 256),
    kl_weight=1e-3,
)
```

## Parameters

- `autoencoder`, `vae`, `denoising_autoencoder`, and `transformer_ae` are the bundled representation and reconstruction baselines.
- `mlp_classifier` is the bundled supervised classification baseline.
- constructor parameters vary by model family and are documented below through autodoc.

## Input expectations

- bundled models expect feature matrices with cells on the batch dimension and genes/features on the last dimension.
- `mlp_classifier` expects encoded class labels during training.
- most users should let `TaskRunner` or `prepare_data(...)` handle preprocessing and split construction.

## Returns / outputs

- encoder-style models expose latent outputs for representation workflows.
- reconstruction-capable models expose reconstructed expression outputs.
- classification models expose logits.
- `create_model(...)` returns an instantiated bundled model ready for `TaskRunner` or `Trainer`.

## Failure modes / raises

- `create_model(...)` raises when the requested model name is unknown or required constructor arguments are missing.
- task mismatches raise when a model is used with an unsupported task.

## Notes / caveats

- `TaskRunner` is the recommended stable path for bundled models.
- the tutorial suite is the best place to see these models in realistic single-cell workflows.
- the experimental scGPT path is intentionally documented elsewhere under [Experimental foundation helpers](./foundation.md).

## Related tutorial(s)

- [Scanpy PBMC quickstart](/_tutorials/scanpy_pbmc_quickstart)
- [PBMC model comparison](/_tutorials/pbmc_model_comparison)
- [PBMC classification](/_tutorials/pbmc_classification)

```{eval-rst}
.. automodule:: scdlkit.models.registry
   :members:
```

```{eval-rst}
.. automodule:: scdlkit.models.autoencoder
   :members:
```

```{eval-rst}
.. automodule:: scdlkit.models.vae
   :members:
```

```{eval-rst}
.. automodule:: scdlkit.models.denoising
   :members:
```

```{eval-rst}
.. automodule:: scdlkit.models.transformer
   :members:
```

```{eval-rst}
.. automodule:: scdlkit.models.classifier
   :members:
```
