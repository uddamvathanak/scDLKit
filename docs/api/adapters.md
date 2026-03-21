# Adapters

## What it is

Status: stable.

Adapters are the supported stable path for bringing a raw PyTorch `nn.Module`
into scDLKit without registering it as a bundled model.

## When to use it

Use adapters when:

- you already have your own PyTorch module
- you want to train it through [Trainer](./trainer.md)
- you want scDLKit evaluation, reporting, and Scanpy handoff patterns without rewriting task glue

## Minimal example

```python
from scdlkit import Trainer, prepare_data
from scdlkit.adapters import wrap_reconstruction_module

prepared = prepare_data(adata, label_key="louvain")
adapter = wrap_reconstruction_module(custom_autoencoder)
trainer = Trainer(model=adapter, task="representation", device="auto")
trainer.fit(prepared.train, prepared.val)
```

## Parameters

- `wrap_reconstruction_module(...)` expects a module that can produce latent and reconstruction-style outputs.
- `wrap_classification_module(...)` expects a module that can produce class logits for encoded labels.
- the adapter classes expose the task-aware surface that `Trainer` expects.

## Input expectations

- wrapped modules must be ordinary PyTorch modules.
- reconstruction-style modules must expose the outputs needed for representation or reconstruction tasks.
- classification-style modules must expose logits compatible with encoded class labels.
- you are still responsible for data preparation through `prepare_data(...)` or equivalent lower-level helpers.

## Returns / outputs

- `wrap_reconstruction_module(...)` returns a `ReconstructionModuleAdapter`.
- `wrap_classification_module(...)` returns a `ClassificationModuleAdapter`.
- adapter objects can be passed directly to `Trainer`.

## Failure modes / raises

- wrapped modules that do not satisfy the expected task contract will fail during training or prediction.
- task mismatches still raise errors through `Trainer` or the task-specific loss path.

## Notes / caveats

- Adapters are for extension, not for replacing the bundled high-level workflow.
- [Custom model extension](/_tutorials/custom_model_extension) is the reference notebook for validating an adapter end to end.
- Use bundled baselines with [TaskRunner](./taskrunner.md) unless you genuinely need a custom module path.

## Related tutorial(s)

- [Custom model extension](/_tutorials/custom_model_extension)
- [PBMC classification](/_tutorials/pbmc_classification)

```{eval-rst}
.. autoclass:: scdlkit.adapters.TorchModuleAdapter
   :members:
```

```{eval-rst}
.. autoclass:: scdlkit.adapters.ReconstructionModuleAdapter
   :members:
```

```{eval-rst}
.. autoclass:: scdlkit.adapters.ClassificationModuleAdapter
   :members:
```

```{eval-rst}
.. autofunction:: scdlkit.adapters.wrap_reconstruction_module
```

```{eval-rst}
.. autofunction:: scdlkit.adapters.wrap_classification_module
```
