# Trainer

## What it is

Status: stable.

`Trainer` is the lower-level training and inference loop behind the high-level
workflow surfaces. It accepts an already-constructed model plus a task adapter
and runs batched training or prediction.

## When to use it

Use `Trainer` when:

- `TaskRunner` is too opinionated for your workflow
- you are wrapping a raw PyTorch module with the adapter APIs
- you want explicit dataset or split control
- you are driving the experimental scGPT path underneath `adapt_annotation(...)`

## Minimal example

```python
from scdlkit import Trainer, create_model, prepare_data

prepared = prepare_data(adata, label_key="louvain")
model = create_model("vae", input_dim=prepared.input_dim, latent_dim=32)

trainer = Trainer(
    model=model,
    task="representation",
    device="auto",
    epochs=20,
    batch_size=128,
)

trainer.fit(prepared.train, prepared.val)
predictions = trainer.predict_dataset(prepared.test or prepared.val or prepared.train)
```

## Parameters

- `model`: a PyTorch module or a scDLKit-compatible wrapped model.
- `task`: task name or instantiated task adapter.
- `epochs`, `batch_size`, `lr`, `device`, `mixed_precision`, `early_stopping_patience`, `checkpoint`, `seed`: training loop controls.

## Input expectations

- `fit(...)` accepts either `SplitData` objects or datasets yielding batch dictionaries.
- `predict_dataset(...)` expects the same batch-dictionary contract used during training.
- the model must support the selected task through `supported_tasks` or task-specific methods.
- inference-only models can be passed to `predict_dataset(...)`, but `fit(...)` requires `supports_training=True`.

## Returns / outputs

- `fit(...)` returns the fitted `Trainer`.
- `predict_dataset(...)` returns a dictionary of concatenated arrays such as `latent`, `reconstruction`, `logits`, `x`, `y`, or `batch`.
- `history_frame` exposes the training history as a pandas `DataFrame`.
- `save_checkpoint(...)` writes the best checkpointed model state to disk.

## Failure modes / raises

- `ValueError` if the supplied model does not support the requested task.
- `NotImplementedError` if `fit(...)` is called on an inference-only model.
- `RuntimeError` if you try to save a checkpoint before one exists.

## Notes / caveats

- `Trainer` does not perform AnnData preprocessing for you; pair it with [Data preparation](./data.md) or the foundation helpers.
- `TaskRunner` remains the recommended stable beginner path for bundled models.
- `predict_dataset(...)` is the public bridge used by the experimental foundation workflows.

## Related tutorial(s)

- [Custom model extension](/_tutorials/custom_model_extension)
- [Experimental scGPT cell-type annotation](/_tutorials/scgpt_cell_type_annotation)
- [Experimental scGPT dataset-specific annotation](/_tutorials/scgpt_dataset_specific_annotation)

```{eval-rst}
.. autoclass:: scdlkit.training.trainer.Trainer
   :members:
```
